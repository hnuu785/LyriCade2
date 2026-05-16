import argparse
import csv
import re
from pathlib import Path


REQUIRED_COLUMNS = ["title", "artist", "lyrics", "bpm"]
OUTPUT_COLUMNS = [
    "title",
    "artist",
    "lyrics",
    "bpm",
    "bpm_class",
    "line_count",
    "avg_chars_per_line",
    "density_class",
]
HANGUL_CHAR_RE = re.compile(r"[가-힣]")


def classify_bpm(bpm: float) -> str:
    if bpm <= 110:
        return "붐뱁"
    return "트랩"


def count_line_chars(line: str) -> int:
    hangul_chars = len(HANGUL_CHAR_RE.findall(line))
    if hangul_chars > 0:
        return hangul_chars
    return len(line.replace(" ", ""))


def classify_density(avg_chars_per_line: float) -> str:
    if avg_chars_per_line < 10:
        return "적은 밀도"
    if avg_chars_per_line < 16:
        return "보통 밀도"
    return "높은 밀도"


def build_structure_fields(lyrics_text: str, bpm_value: float) -> dict:
    normalized = (lyrics_text or "").replace("\r", "\n")
    lines = [line.strip() for line in normalized.split("\n") if line.strip()]
    line_count = len(lines)
    avg_chars_per_line = 0.0
    if lines:
        avg_chars_per_line = sum(count_line_chars(line) for line in lines) / len(lines)

    return {
        "bpm_class": classify_bpm(bpm_value),
        "line_count": line_count,
        "avg_chars_per_line": round(avg_chars_per_line, 2),
        "density_class": classify_density(avg_chars_per_line),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Create a slim training CSV with only title, artist, lyrics, and bpm.")
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    with input_path.open("r", encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is None:
            raise ValueError("Input CSV is missing a header row.")

        lowered = {column.lower(): column for column in reader.fieldnames}
        missing = [column for column in REQUIRED_COLUMNS if column not in lowered]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        output_rows = []
        for row in reader:
            slim_row = {column: row[lowered[column]] for column in REQUIRED_COLUMNS}
            bpm_value = float(slim_row["bpm"])
            structure_fields = build_structure_fields(slim_row["lyrics"], bpm_value)
            output_rows.append({**slim_row, **structure_fields})

    missing = [column for column in REQUIRED_COLUMNS if column not in lowered]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Saved slim dataset to {output_path}")
    print(f"Rows: {len(output_rows)}")


if __name__ == "__main__":
    main()
