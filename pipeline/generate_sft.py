import argparse
import re
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


def build_feature_prompt(args):
    raw_values = {
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
        "ARTIST": args.artist,
        "TRACK_GENRE": args.track_genre,
    }
    transformed_values = transform_inference_features(raw_values, args.feature_state)
    return build_prompt(transformed_values)


def _split_line(line: str):
    segments = [segment.strip() for segment in re.split(r"(?<=[,?!.;])\s+|,\s+|(?<=\))\s+", line) if segment.strip()]
    return segments or [line.strip()]


def _split_longest_line(lines):
    if not lines:
        return lines

    longest_index = max(range(len(lines)), key=lambda idx: len(lines[idx].split()))
    words = lines[longest_index].split()
    if len(words) < 2:
        return lines

    midpoint = max(1, len(words) // 2)
    replacement = [" ".join(words[:midpoint]).strip(), " ".join(words[midpoint:]).strip()]
    return lines[:longest_index] + replacement + lines[longest_index + 1 :]


def format_generated_lyrics(text: str, line_count: int) -> str:
    raw_lines = [line.strip() for line in text.replace("\r", "\n").split("\n") if line.strip()]

    lines = []
    for raw_line in raw_lines:
        lines.extend(_split_line(raw_line))

    if not lines:
        return ""

    while len(lines) < line_count:
        updated_lines = _split_longest_line(lines)
        if updated_lines == lines:
            break
        lines = [line for line in updated_lines if line.strip()]

    if len(lines) > line_count:
        lines = lines[:line_count]

    return "\n".join(line.strip() for line in lines if line.strip())


def parse_args():
    parser = argparse.ArgumentParser(description="Generate lyrics from a LoRA SFT adapter checkpoint.")
    parser.add_argument("--adapter-dir", required=True, help="Path to the saved LoRA adapter directory.")
    parser.add_argument("--base-model", help="Optional base model name/path. Defaults to adapter config.")
    parser.add_argument("--tokenizer-dir", help="Optional tokenizer directory. Defaults to adapter parent/final_tokenizer.")
    parser.add_argument("--feature-config", help="Optional feature transform JSON. Defaults to the adapter parent dir.")
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
    parser.add_argument("--line-count", type=int, default=8, help="Number of lyric lines/bars to format in the output.")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    adapter_dir = Path(args.adapter_dir).resolve()
    tokenizer_dir = Path(args.tokenizer_dir).resolve() if args.tokenizer_dir else adapter_dir.parent / "final_tokenizer"
    feature_config = Path(args.feature_config).resolve() if args.feature_config else adapter_dir.parent / FEATURE_STATE_FILENAME
    args.feature_state = load_feature_state(feature_config)

    peft_config = PeftConfig.from_pretrained(adapter_dir)
    base_model_name = args.base_model or peft_config.base_model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(base_model_name, local_files_only=args.local_files_only)
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
    formatted = format_generated_lyrics(generated, line_count=args.line_count)
    print(formatted)


if __name__ == "__main__":
    main()
