import argparse
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipeline.features import FEATURE_STATE_FILENAME, build_prompt, load_feature_state, transform_inference_features


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def classify_bpm(bpm: float) -> str:
    if bpm <= 110:
        return "붐뱁"
    return "트랩"


def infer_density_class(avg_chars_per_line: float) -> str:
    if avg_chars_per_line < 10:
        return "적은 밀도"
    if avg_chars_per_line < 16:
        return "보통 밀도"
    return "높은 밀도"


def infer_avg_chars_per_line(bpm: float) -> float:
    if bpm < 90:
        return 9.0
    if bpm < 110:
        return 11.0
    if bpm < 130:
        return 14.0
    if bpm < 150:
        return 17.0
    return 19.0


def build_feature_prompt(args):
    avg_chars_per_line = args.avg_chars_per_line if args.avg_chars_per_line is not None else infer_avg_chars_per_line(args.bpm)
    raw_values = {
        "BPM": args.bpm,
        "LINE_COUNT": args.line_count,
        "AVG_CHARS_PER_LINE": avg_chars_per_line,
        "ARTIST": args.artist,
        "BPM_CLASS": args.bpm_class or classify_bpm(args.bpm),
        "DENSITY_CLASS": args.density_class or infer_density_class(avg_chars_per_line),
    }
    transformed_values = transform_inference_features(raw_values, args.feature_state)
    return build_prompt(transformed_values)

def format_generated_lyrics(text: str, line_count: int) -> str:
    raw_lines = [line.strip() for line in text.replace("\r", "\n").split("\n") if line.strip()]
    return "\n".join(raw_lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate lyrics from either a LoRA SFT adapter checkpoint or a base model.")
    parser.add_argument("--adapter-dir", help="Path to the saved LoRA adapter directory.")
    parser.add_argument("--base-model", help="Base model name/path. Required when using --base-model-only.")
    parser.add_argument("--base-model-only", action="store_true", help="Run generation with the base model only, without loading a LoRA adapter.")
    parser.add_argument("--tokenizer-dir", help="Optional tokenizer directory. Defaults to adapter parent/final_tokenizer, or the base model in --base-model-only mode.")
    parser.add_argument("--feature-config", help="Optional feature transform JSON. Defaults to the adapter parent dir.")
    parser.add_argument("--artist", default="unknown", help="Artist conditioning token.")
    parser.add_argument("--bpm", type=float, default=120.0)
    parser.add_argument("--bpm-class", help="Optional BPM class override such as 붐뱁 or 트랩.")
    parser.add_argument("--density-class", help="Optional density class override such as 적은 밀도, 보통 밀도, or 높은 밀도.")
    parser.add_argument("--line-count", type=float, default=8.0, help="Target verse line count used in the prompt.")
    parser.add_argument("--avg-chars-per-line", type=float, help="Optional average visible chars per line target.")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--min-length", type=int, default=64)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    adapter_dir = Path(args.adapter_dir).resolve() if args.adapter_dir else None

    if args.base_model_only:
        if adapter_dir is not None:
            raise ValueError("--adapter-dir cannot be used together with --base-model-only.")
        if not args.base_model:
            raise ValueError("--base-model is required when using --base-model-only.")
    elif adapter_dir is None:
        raise ValueError("--adapter-dir is required unless --base-model-only is set.")

    if args.tokenizer_dir:
        tokenizer_dir = Path(args.tokenizer_dir).resolve()
    elif adapter_dir is not None:
        tokenizer_dir = adapter_dir.parent / "final_tokenizer"
    else:
        tokenizer_dir = args.base_model

    if args.feature_config:
        feature_config = Path(args.feature_config).resolve()
    elif adapter_dir is not None:
        feature_config = adapter_dir.parent / FEATURE_STATE_FILENAME
    else:
        raise ValueError("--feature-config is required when using --base-model-only.")

    args.feature_state = load_feature_state(feature_config)

    if args.base_model_only:
        base_model_name = args.base_model
    else:
        peft_config = PeftConfig.from_pretrained(adapter_dir)
        base_model_name = args.base_model or peft_config.base_model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(base_model_name, local_files_only=args.local_files_only)
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, adapter_dir, local_files_only=args.local_files_only)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer must define either a pad token or an eos token.")
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    device = get_device()
    model.to(device)
    model.eval()

    prompt = build_feature_prompt(args)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    output = model.generate(
        **inputs,
        max_length=args.max_length,
        min_length=args.min_length,
        num_beams=args.num_beams,
        no_repeat_ngram_size=2,
        early_stopping=True,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    generated = decoded.split("<LYRICS>:", maxsplit=1)[-1].strip()
    formatted = format_generated_lyrics(generated, line_count=int(args.line_count))
    print(formatted)


if __name__ == "__main__":
    main()
