import argparse
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


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


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_feature_prompt(args):
    numerical_values = {
        "DURATION_MS": args.duration_ms,
        "POPULARITY": args.popularity,
        "DANCEABILITY": args.danceability,
        "ENERGY": args.energy,
        "LOUDNESS": args.loudness,
        "ACOUSTICNESS": args.acousticness,
        "INSTRUMENTALNESS": args.instrumentalness,
        "LIVENESS": args.liveness,
        "VALENCE": args.valence,
        "TEMPO": args.tempo,
        "EXPLICIT": args.explicit,
    }
    categorical_values = {
        "ARTIST": args.artist,
        "TRACK_GENRE": args.track_genre,
    }

    feature_string = " ".join(
        [f"<{column}: {numerical_values[column]:.2f}>" for column in NUMERICAL_COLUMNS]
        + [f"<{column}: {categorical_values[column]}>" for column in CATEGORICAL_COLUMNS]
    )
    return f"{feature_string} <LYRICS>:"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Korean lyrics from a fine-tuned checkpoint.")
    parser.add_argument("--model-dir", required=True, help="Path to the saved model directory.")
    parser.add_argument("--tokenizer-dir", help="Optional tokenizer directory. Defaults to --model-dir.")
    parser.add_argument("--artist", default="unknown", help="Artist conditioning token.")
    parser.add_argument("--track-genre", default="k-rap", help="Genre conditioning token.")
    parser.add_argument("--duration-ms", type=float, default=211361.0)
    parser.add_argument("--popularity", type=float, default=46.0)
    parser.add_argument("--danceability", type=float, default=0.0)
    parser.add_argument("--energy", type=float, default=0.0)
    parser.add_argument("--loudness", type=float, default=0.0)
    parser.add_argument("--acousticness", type=float, default=0.0)
    parser.add_argument("--instrumentalness", type=float, default=0.0)
    parser.add_argument("--liveness", type=float, default=0.0)
    parser.add_argument("--valence", type=float, default=0.0)
    parser.add_argument("--tempo", type=float, default=120.0)
    parser.add_argument("--explicit", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--min-length", type=int, default=64)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.92)
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    tokenizer_dir = Path(args.tokenizer_dir).resolve() if args.tokenizer_dir else model_dir

    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

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
    print(generated)


if __name__ == "__main__":
    main()
