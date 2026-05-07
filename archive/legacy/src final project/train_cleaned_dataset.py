import argparse
import random
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, get_linear_schedule_with_warmup

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rhyme_loss import compute_rhyme_loss, normalize_lyrics_for_rhyme, prepare_rhyme_supervision


NUMERICAL_COLUMNS = [
    "DURATION_MS",
    "POPULARITY",
    "DANCEABILITY",
    "ENERGY",
    "LOUDNESS",
    "ACOUSTICNESS",
    "INSTRUMENTALNESS",
    "LIVENESS",
    "VALENCE",
    "TEMPO",
    "EXPLICIT",
]

CATEGORICAL_COLUMNS = ["ARTIST", "TRACK_GENRE"]

FEATURE_WEIGHTS = {
    "TEMPO": 4.5,
    "VALENCE": 4.0,
    "POPULARITY": 4.0,
    "DANCEABILITY": 3.5,
    "ENERGY": 4.5,
    "LOUDNESS": 3.0,
    "ACOUSTICNESS": 3.5,
    "INSTRUMENTALNESS": 3.5,
    "LIVENESS": 3.0,
    "EXPLICIT": 2.5,
    "DURATION_MS": 1.0,
}

# These defaults come from the existing LyricGeneration training data distribution.
DEFAULT_FEATURES = {
    "POPULARITY": 46.0,
    "DURATION_MS": 211361.0,
    "EXPLICIT": 0.0,
    "ACOUSTICNESS": 0.1275,
    "INSTRUMENTALNESS": 0.0000757,
    "LIVENESS": 0.122,
    "TEMPO": 120.034,
    "TRACK_GENRE": "k-rap",
}

MAX_LENGTH = 512
MAX_RHYME_POSITIONS = 16


class LyricsFeatureDataset(Dataset):
    def __init__(self, encodings, rhyme_positions, rhyme_targets):
        self.encodings = encodings
        self.rhyme_positions = rhyme_positions
        self.rhyme_targets = rhyme_targets

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item["rhyme_positions"] = self.rhyme_positions[idx].clone().detach()
        item["rhyme_targets"] = self.rhyme_targets[idx].clone().detach()
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_input_sequences(df):
    sequences = []
    for _, row in df.iterrows():
        feature_string = " ".join(
            [f"<{col}: {row[col]:.2f}>" for col in NUMERICAL_COLUMNS]
            + [f"<{col}: {row[col]}>" for col in CATEGORICAL_COLUMNS]
        )
        sequences.append(f"{feature_string} <LYRICS>: {row['LYRICS']}")
    return sequences


def apply_feature_weights(df):
    weighted = df.copy()
    for column, weight in FEATURE_WEIGHTS.items():
        weighted[column] = weighted[column] * weight
    return weighted


def load_cleaned_dataset(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = df.columns.str.upper()

    required_source_columns = {"TITLE", "ARTIST", "LYRICS", "BPM", "ENERGY", "DANCEABILITY", "LOUDNESS", "VALENCE"}
    missing_source_columns = sorted(required_source_columns - set(df.columns))
    if missing_source_columns:
        raise ValueError(f"Missing required columns in cleaned dataset: {missing_source_columns}")

    prepared = pd.DataFrame()
    prepared["TITLE"] = df["TITLE"].astype(str).str.strip()
    prepared["ARTIST"] = df["ARTIST"].astype(str).str.strip()
    prepared["LYRICS"] = df["LYRICS"].fillna("").apply(normalize_lyrics_for_rhyme)
    prepared["DANCEABILITY"] = pd.to_numeric(df["DANCEABILITY"], errors="coerce")
    prepared["ENERGY"] = pd.to_numeric(df["ENERGY"], errors="coerce")
    prepared["LOUDNESS"] = pd.to_numeric(df["LOUDNESS"], errors="coerce")
    prepared["VALENCE"] = pd.to_numeric(df["VALENCE"], errors="coerce")
    prepared["TEMPO"] = pd.to_numeric(df["BPM"], errors="coerce")

    prepared["POPULARITY"] = DEFAULT_FEATURES["POPULARITY"]
    prepared["DURATION_MS"] = DEFAULT_FEATURES["DURATION_MS"]
    prepared["EXPLICIT"] = DEFAULT_FEATURES["EXPLICIT"]
    prepared["ACOUSTICNESS"] = DEFAULT_FEATURES["ACOUSTICNESS"]
    prepared["INSTRUMENTALNESS"] = DEFAULT_FEATURES["INSTRUMENTALNESS"]
    prepared["LIVENESS"] = DEFAULT_FEATURES["LIVENESS"]
    prepared["TRACK_GENRE"] = DEFAULT_FEATURES["TRACK_GENRE"]

    prepared["TEMPO"] = prepared["TEMPO"].fillna(DEFAULT_FEATURES["TEMPO"])

    for column in NUMERICAL_COLUMNS:
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
            prepared[column] = prepared[column].fillna(DEFAULT_FEATURES.get(column, prepared[column].median()))

    prepared = prepared[prepared["LYRICS"].str.len() > 0].reset_index(drop=True)
    return prepared


def split_and_scale(df, seed, val_ratio):
    train_df, val_df = train_test_split(df, test_size=val_ratio, random_state=seed, shuffle=True)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    scaler = StandardScaler()
    train_df.loc[:, NUMERICAL_COLUMNS] = scaler.fit_transform(train_df[NUMERICAL_COLUMNS])
    val_df.loc[:, NUMERICAL_COLUMNS] = scaler.transform(val_df[NUMERICAL_COLUMNS])

    train_df = apply_feature_weights(train_df)
    val_df = apply_feature_weights(val_df)
    return train_df, val_df, scaler


def build_dataloaders(train_df, val_df, tokenizer, batch_size):
    train_sequences = create_input_sequences(train_df)
    val_sequences = create_input_sequences(val_df)

    train_encodings, train_rhyme_positions, train_rhyme_targets = prepare_rhyme_supervision(
        train_sequences,
        tokenizer,
        max_length=MAX_LENGTH,
        max_rhyme_positions=MAX_RHYME_POSITIONS,
    )
    val_encodings, val_rhyme_positions, val_rhyme_targets = prepare_rhyme_supervision(
        val_sequences,
        tokenizer,
        max_length=MAX_LENGTH,
        max_rhyme_positions=MAX_RHYME_POSITIONS,
    )

    train_dataset = LyricsFeatureDataset(train_encodings, train_rhyme_positions, train_rhyme_targets)
    val_dataset = LyricsFeatureDataset(val_encodings, val_rhyme_positions, val_rhyme_targets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def evaluate(model, data_loader, device, rhyme_loss_weight):
    model.eval()
    total_loss = 0.0
    total_rhyme_loss = 0.0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            rhyme_positions = batch["rhyme_positions"].to(device)
            rhyme_targets = batch["rhyme_targets"].to(device)

            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
                output_hidden_states=True,
            )
            rhyme_loss = compute_rhyme_loss(outputs.hidden_states[-1], rhyme_positions, rhyme_targets)
            loss = outputs.loss + (rhyme_loss_weight * rhyme_loss)

            total_loss += loss.item()
            total_rhyme_loss += rhyme_loss.item()

    return total_loss / max(1, len(data_loader)), total_rhyme_loss / max(1, len(data_loader))


def train(args):
    set_seed(args.seed)

    base_dir = SCRIPT_DIR
    csv_path = (base_dir / args.csv).resolve() if not Path(args.csv).is_absolute() else Path(args.csv)
    output_dir = (base_dir / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_cleaned_dataset(csv_path)
    train_df, val_df, _ = split_and_scale(df, seed=args.seed, val_ratio=args.val_ratio)

    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Train rows: {len(train_df)}")
    print(f"Validation rows: {len(val_df)}")

    tokenizer = GPT2TokenizerFast.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    model = GPT2LMHeadModel.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    train_loader, val_loader = build_dataloaders(train_df, val_df, tokenizer, args.batch_size)

    device = get_device()
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    print(f"Device: {device}")
    print(f"Steps per epoch: {len(train_loader)}")

    best_val_loss = None
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        epoch_rhyme_losses = []

        for batch_idx, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            rhyme_positions = batch["rhyme_positions"].to(device)
            rhyme_targets = batch["rhyme_targets"].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
                output_hidden_states=True,
            )
            rhyme_loss = compute_rhyme_loss(outputs.hidden_states[-1], rhyme_positions, rhyme_targets)
            loss = outputs.loss + (args.rhyme_loss_weight * rhyme_loss)

            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss.item())
            epoch_rhyme_losses.append(rhyme_loss.item())

            if batch_idx % args.log_every == 0 or batch_idx == len(train_loader):
                print(
                    f"Epoch {epoch + 1}/{args.epochs} "
                    f"Batch {batch_idx}/{len(train_loader)} "
                    f"Loss {loss.item():.4f} "
                    f"RhymeLoss {rhyme_loss.item():.4f}"
                )

        avg_train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        avg_train_rhyme_loss = float(np.mean(epoch_rhyme_losses)) if epoch_rhyme_losses else 0.0
        avg_val_loss, avg_val_rhyme_loss = evaluate(model, val_loader, device, args.rhyme_loss_weight)

        print(
            f"Epoch {epoch + 1} complete | "
            f"TrainLoss {avg_train_loss:.4f} | "
            f"TrainRhymeLoss {avg_train_rhyme_loss:.4f} | "
            f"ValLoss {avg_val_loss:.4f} | "
            f"ValRhymeLoss {avg_val_rhyme_loss:.4f}"
        )

        if best_val_loss is None or avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(output_dir / "best_model")
            tokenizer.save_pretrained(output_dir / "best_tokenizer")
            print(f"Saved best checkpoint to {output_dir}")

    model.save_pretrained(output_dir / "final_model")
    tokenizer.save_pretrained(output_dir / "final_tokenizer")
    print(f"Saved final checkpoint to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the lyric generation model directly from FINAL_DATASET_CLEANED.csv."
    )
    parser.add_argument(
        "--csv",
        default="TRAIN_KOR_DATA.csv",
        help="Path to the training CSV. Relative paths resolve from src final project.",
    )
    parser.add_argument(
        "--output-dir",
        default="cleaned_dataset_training_output",
        help="Directory where checkpoints will be saved.",
    )
    parser.add_argument("--model-name", default="gpt2", help="Base Hugging Face model name or local model path.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Optimizer learning rate.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--rhyme-loss-weight", type=float, default=0.05, help="Rhyme loss weight.")
    parser.add_argument("--log-every", type=int, default=10, help="Log every N batches.")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only load model/tokenizer from the local Hugging Face cache.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
