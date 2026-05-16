import argparse
import csv
import inspect
import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

from pipeline.features import (
    build_prompt,
    apply_feature_transform,
    build_completion_text,
    fit_feature_transform,
    load_cleaned_dataset,
    save_feature_state,
)
from pipeline.rhyme import (
    _extract_line_end_spans,
    _map_spans_to_token_positions,
    compute_rhyme_loss,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_storage_root() -> Path:
    drive_root = Path("/content/drive/MyDrive")
    if drive_root.exists():
        return drive_root / PROJECT_ROOT.name
    return PROJECT_ROOT


STORAGE_ROOT = resolve_storage_root()


def resolve_output_dir(output_dir_arg: str) -> Path:
    base_output_dir = Path(output_dir_arg).resolve()
    dated_output_dir = base_output_dir / datetime.now().strftime("%Y-%m-%d")
    dated_output_dir.mkdir(parents=True, exist_ok=True)
    return dated_output_dir


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_dataset(df, seed, val_ratio):
    train_df, val_df = train_test_split(df, test_size=val_ratio, random_state=seed, shuffle=True)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def infer_lora_targets(model):
    model_type = getattr(model.config, "model_type", "")
    if model_type == "gpt2":
        return ["c_attn", "c_proj"]
    if model_type in {"llama", "mistral", "qwen2", "qwen3"}:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    return ["c_attn", "c_proj"]


class PromptCompletionDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length, max_rhyme_positions):
        self.examples = []

        for row in rows:
            prompt = build_prompt(row)
            completion = build_completion_text(row["LYRICS"])
            sequence = f"{prompt}{completion}"

            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
            offset_mapping = tokenizer(
                sequence,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
                return_offsets_mapping=True,
            )["offset_mapping"]

            spans = _extract_line_end_spans(sequence)
            rhyme_positions, rhyme_targets = _map_spans_to_token_positions(
                offset_mapping=offset_mapping,
                spans=spans,
                max_rhyme_positions=max_rhyme_positions,
            )

            input_ids = (prompt_ids + completion_ids)[:max_length]
            attention_mask = [1] * len(input_ids)
            labels = ([-100] * len(prompt_ids) + completion_ids)[:max_length]

            self.examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                    "rhyme_positions": rhyme_positions,
                    "rhyme_targets": rhyme_targets,
                }
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class SFTDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        input_features = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            }
            for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        batch = self.tokenizer.pad(input_features, padding=True, return_tensors="pt")
        labels = self.tokenizer.pad(label_features, padding=True, return_tensors="pt")["input_ids"]
        labels = labels.masked_fill(batch["attention_mask"] == 0, -100)
        batch["labels"] = labels
        batch["rhyme_positions"] = torch.stack([feature["rhyme_positions"] for feature in features])
        batch["rhyme_targets"] = torch.stack([feature["rhyme_targets"] for feature in features])
        return batch


class RhymeAwareTrainer(Trainer):
    def __init__(self, *args, rhyme_loss_weight: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.rhyme_loss_weight = rhyme_loss_weight

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        rhyme_positions = inputs.pop("rhyme_positions", None)
        rhyme_targets = inputs.pop("rhyme_targets", None)

        outputs = model(**inputs, output_hidden_states=True)
        loss = outputs.loss

        if (
            self.rhyme_loss_weight > 0.0
            and rhyme_positions is not None
            and rhyme_targets is not None
            and outputs.hidden_states
        ):
            rhyme_loss = compute_rhyme_loss(
                hidden_states=outputs.hidden_states[-1],
                rhyme_positions=rhyme_positions,
                rhyme_targets=rhyme_targets,
            )
            loss = loss + (self.rhyme_loss_weight * rhyme_loss)

        return (loss, outputs) if return_outputs else loss


class MetricsLoggerCallback(TrainerCallback):
    def __init__(self, experiment_id: str, metrics_dir: Path, plots_dir: Path):
        self.experiment_id = experiment_id
        self.metrics_dir = Path(metrics_dir).resolve()
        self.plots_dir = Path(plots_dir).resolve()
        self.train_rows = []
        self.eval_rows = []

        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if "loss" not in logs:
            return

        self.train_rows.append(
            {
                "step": state.global_step,
                "epoch": logs.get("epoch", state.epoch),
                "loss": logs.get("loss"),
                "grad_norm": logs.get("grad_norm"),
                "learning_rate": logs.get("learning_rate"),
            }
        )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return

        self.eval_rows.append(
            {
                "step": state.global_step,
                "epoch": metrics.get("epoch", state.epoch),
                "eval_loss": metrics.get("eval_loss"),
                "eval_runtime": metrics.get("eval_runtime"),
                "eval_samples_per_second": metrics.get("eval_samples_per_second"),
                "eval_steps_per_second": metrics.get("eval_steps_per_second"),
            }
        )

    def on_train_end(self, args, state, control, **kwargs):
        train_csv = self.metrics_dir / f"{self.experiment_id}-train.csv"
        eval_csv = self.metrics_dir / f"{self.experiment_id}-eval.csv"
        plot_path = self.plots_dir / f"{self.experiment_id}-loss.png"

        self._write_csv(
            train_csv,
            ["step", "epoch", "loss", "grad_norm", "learning_rate"],
            self.train_rows,
        )
        self._write_csv(
            eval_csv,
            ["step", "epoch", "eval_loss", "eval_runtime", "eval_samples_per_second", "eval_steps_per_second"],
            self.eval_rows,
        )
        self._write_plot(plot_path)

    def _write_csv(self, path: Path, fieldnames, rows):
        with path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _write_plot(self, path: Path):
        plt.figure(figsize=(8, 5))

        train_x = [row["step"] for row in self.train_rows if row.get("loss") is not None]
        train_y = [row["loss"] for row in self.train_rows if row.get("loss") is not None]
        if train_x and train_y:
            plt.plot(train_x, train_y, label="train_loss", marker="o", linewidth=1.5, markersize=3)

        eval_x = [row["step"] for row in self.eval_rows if row.get("eval_loss") is not None]
        eval_y = [row["eval_loss"] for row in self.eval_rows if row.get("eval_loss") is not None]
        if eval_x and eval_y:
            plt.plot(eval_x, eval_y, label="eval_loss", marker="s", linewidth=1.5, markersize=4)

        plt.xlabel("Global Step")
        plt.ylabel("Loss")
        plt.title(f"{self.experiment_id} Loss Curve")
        plt.grid(True, alpha=0.3)
        if train_x or eval_x:
            plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train a conditional Korean lyric SFT model with LoRA.")
    parser.add_argument("--experiment-id", default="adhoc", help="Identifier used for metrics CSV and plots.")
    parser.add_argument("--csv", required=True, help="Path to the training CSV.")
    parser.add_argument(
        "--output-dir",
        default=str(STORAGE_ROOT / "outputs" / "sft-run"),
        help="Directory where checkpoints will be saved.",
    )
    parser.add_argument(
        "--metrics-dir",
        default=str(STORAGE_ROOT / "experiments" / "metrics"),
        help="Directory where training/eval CSV logs will be saved.",
    )
    parser.add_argument(
        "--plots-dir",
        default=str(STORAGE_ROOT / "experiments" / "plots"),
        help="Directory where training plots will be saved.",
    )
    parser.add_argument("--model-name", default="gpt2", help="Base Hugging Face model name or local model path.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--rhyme-loss-weight", type=float, default=0.1)
    parser.add_argument("--max-rhyme-positions", type=int, default=16)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    csv_path = Path(args.csv).resolve()
    output_dir = resolve_output_dir(args.output_dir)

    df = load_cleaned_dataset(csv_path)
    train_df, val_df = split_dataset(df, seed=args.seed, val_ratio=args.val_ratio)
    feature_state = fit_feature_transform(train_df)
    train_df = apply_feature_transform(train_df, feature_state)
    val_df = apply_feature_transform(val_df, feature_state)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, local_files_only=args.local_files_only)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer must define either a pad token or an eos token.")
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=infer_lora_targets(model),
    )
    model = get_peft_model(model, lora_config)

    train_dataset = PromptCompletionDataset(
        train_df.to_dict("records"),
        tokenizer,
        args.max_length,
        args.max_rhyme_positions,
    )
    val_dataset = PromptCompletionDataset(
        val_df.to_dict("records"),
        tokenizer,
        args.max_length,
        args.max_rhyme_positions,
    )

    training_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "report_to": [],
        "remove_unused_columns": False,
    }
    training_arg_names = set(inspect.signature(TrainingArguments.__init__).parameters)
    if "evaluation_strategy" in training_arg_names:
        training_kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in training_arg_names:
        training_kwargs["eval_strategy"] = "epoch"
    training_args = TrainingArguments(**training_kwargs)

    trainer = RhymeAwareTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=SFTDataCollator(tokenizer),
        tokenizer=tokenizer,
        rhyme_loss_weight=args.rhyme_loss_weight,
        callbacks=[
            MetricsLoggerCallback(
                experiment_id=args.experiment_id,
                metrics_dir=Path(args.metrics_dir),
                plots_dir=Path(args.plots_dir),
            )
        ],
    )

    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Train rows: {len(train_df)}")
    print(f"Validation rows: {len(val_df)}")
    print(f"Saving checkpoints under {output_dir}")
    print(f"Rhyme loss weight: {args.rhyme_loss_weight}")

    trainer.train()

    model.save_pretrained(output_dir / "final_adapter")
    tokenizer.save_pretrained(output_dir / "final_tokenizer")
    save_feature_state(output_dir, feature_state)
    print(f"Saved LoRA adapter and tokenizer to {output_dir}")


if __name__ == "__main__":
    main()
