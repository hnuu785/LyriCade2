import argparse
import re
from pathlib import Path

import pandas as pd


SECTION_HEADER_RE = re.compile(r"^\[(?P<label>[^\]]+)\]\s*$")
VERSE_LABEL_RE = re.compile(r"\bverse\b", re.IGNORECASE)


def normalize_lyrics(text: str) -> str:
    if pd.isna(text):
        return ""
    lyrics = str(text).replace("\r\n", "\n").replace("\r", "\n")
    lyrics = lyrics.replace("\u2005", " ").replace("\u205f", " ").replace("\xa0", " ")
    lyrics = re.sub(r"[ \t]+\n", "\n", lyrics)
    lyrics = re.sub(r"\n{3,}", "\n\n", lyrics)
    return lyrics.strip()


def split_sections(lyrics: str):
    sections = []
    current_label = None
    current_lines = []

    for line in lyrics.split("\n"):
        stripped = line.strip()
        header_match = SECTION_HEADER_RE.match(stripped)
        if header_match:
            if current_label is not None or current_lines:
                sections.append((current_label, "\n".join(current_lines).strip()))
            current_label = header_match.group("label").strip()
            current_lines = []
            continue
        current_lines.append(line)

    if current_label is not None or current_lines:
        sections.append((current_label, "\n".join(current_lines).strip()))

    return sections


def is_verse_label(label: str | None) -> bool:
    return bool(label and VERSE_LABEL_RE.search(label))


def extract_verses(lyrics: str):
    normalized = normalize_lyrics(lyrics)
    if not normalized:
        return "", "empty"

    sections = split_sections(normalized)
    tagged_sections = [section for section in sections if section[0] is not None]

    if tagged_sections:
        verse_blocks = [content for label, content in tagged_sections if is_verse_label(label) and content.strip()]
        if verse_blocks:
            return "\n\n".join(verse_blocks).strip(), "tagged_verse"
        return "", "tagged_no_verse"

    return normalized, "untagged_full_lyrics"


def build_verse_dataset(df: pd.DataFrame, lyrics_column: str) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        verse_lyrics, method = extract_verses(row.get(lyrics_column, ""))
        updated_row = row.copy()
        updated_row["lyrics_verse_only"] = verse_lyrics
        updated_row["verse_extraction_method"] = method
        updated_row["verse_line_count"] = verse_lyrics.count("\n") + 1 if verse_lyrics else 0
        records.append(updated_row)
    return pd.DataFrame(records)


def build_strict_verse_dataset(verse_df: pd.DataFrame, lyrics_column: str) -> pd.DataFrame:
    strict_df = verse_df[verse_df["verse_extraction_method"] == "tagged_verse"].copy()
    strict_df[lyrics_column] = strict_df["lyrics_verse_only"]
    return strict_df.reset_index(drop=True)


def build_fallback_verse_dataset(verse_df: pd.DataFrame, lyrics_column: str) -> pd.DataFrame:
    fallback_df = verse_df[verse_df["verse_extraction_method"] != "tagged_no_verse"].copy()
    fallback_df[lyrics_column] = fallback_df["lyrics_verse_only"]
    return fallback_df.reset_index(drop=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract verse-only lyrics from a Korean lyrics CSV.")
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument("--output", required=True, help="Annotated output CSV path.")
    parser.add_argument("--strict-output", help="Strict output path that keeps only explicitly tagged verses.")
    parser.add_argument("--fallback-output", help="Fallback output path that keeps untagged lyrics as-is.")
    parser.add_argument("--lyrics-column", default="lyrics", help="Lyrics column name in the input CSV.")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    strict_output_path = Path(args.strict_output).resolve() if args.strict_output else None
    fallback_output_path = Path(args.fallback_output).resolve() if args.fallback_output else None

    df = pd.read_csv(input_path, encoding="utf-8-sig")
    verse_df = build_verse_dataset(df, lyrics_column=args.lyrics_column)
    verse_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Saved annotated verse dataset to {output_path}")
    print(f"Annotated rows: {len(verse_df)}")
    print("Extraction summary:")
    for key, value in sorted(verse_df["verse_extraction_method"].value_counts(dropna=False).to_dict().items()):
        print(f"  {key}: {value}")

    if strict_output_path:
        strict_df = build_strict_verse_dataset(verse_df, lyrics_column=args.lyrics_column)
        strict_df.to_csv(strict_output_path, index=False, encoding="utf-8-sig")
        print(f"Saved strict verse-only dataset to {strict_output_path}")
        print(f"Strict rows: {len(strict_df)}")

    if fallback_output_path:
        fallback_df = build_fallback_verse_dataset(verse_df, lyrics_column=args.lyrics_column)
        fallback_df.to_csv(fallback_output_path, index=False, encoding="utf-8-sig")
        print(f"Saved fallback verse dataset to {fallback_output_path}")
        print(f"Fallback rows: {len(fallback_df)}")


if __name__ == "__main__":
    main()
