import argparse
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

HANGUL_RE = re.compile(r"[가-힣]")
ASCII_WORD_RE = re.compile(r"[A-Za-z]+")
LINE_RE = re.compile(r"[^\n]+")


def parse_args():
    parser = argparse.ArgumentParser(description="Optional helper: attach heuristic evaluation to a structured experiment run.")
    parser.add_argument("--run", required=True, help="Path to experiments/runs/EXP-XXX.json")
    return parser.parse_args()


def read_run(path):
    with Path(path).resolve().open("r", encoding="utf-8") as fp:
        return json.load(fp)


def score_sample(text, conditions):
    stripped = text.strip()
    lines = [line.strip() for line in LINE_RE.findall(stripped) if line.strip()]
    ascii_words = ASCII_WORD_RE.findall(stripped)
    hangul_chars = HANGUL_RE.findall(stripped)

    line_count = len(lines)
    word_count = len(stripped.split())
    english_word_count = len(ascii_words)
    hangul_char_count = len(hangul_chars)
    has_mixing = english_word_count > 0 and hangul_char_count > 0

    fluency = 6 if line_count >= 4 else 3
    lyricness = min(10, 3 + line_count)
    bilingual_mixing = 8 if has_mixing else 1

    conditioning_fidelity = 2
    genre = conditions.get("track_genre", "").lower()
    artist = conditions.get("artist", "").lower()
    lowered = stripped.lower()
    if "rap" in genre and english_word_count > 0:
        conditioning_fidelity += 1
    if artist and artist in lowered:
        conditioning_fidelity += 1
    conditioning_fidelity = min(conditioning_fidelity, 10)

    rap_feel = 6 if line_count >= 6 and english_word_count > 0 else 3
    korean_quality = 6 if hangul_char_count > 20 else 3
    english_quality = 5 if english_word_count >= 4 else 1
    originality = 5 if word_count >= 20 else 3

    return {
        "line_count": line_count,
        "word_count": word_count,
        "english_word_count": english_word_count,
        "hangul_char_count": hangul_char_count,
        "scores": {
            "fluency": fluency,
            "lyricness": lyricness,
            "rap_feel": rap_feel,
            "korean_quality": korean_quality,
            "english_quality": english_quality,
            "bilingual_mixing": bilingual_mixing,
            "conditioning_fidelity": conditioning_fidelity,
            "originality": originality,
        },
    }


def summarize_run(run_data):
    train_metrics = run_data.get("train", {}).get("metrics", {})
    eval_loss = train_metrics.get("eval_loss")
    train_loss = train_metrics.get("train_loss")
    samples = run_data.get("generation", {}).get("samples", [])

    sample_evaluations = []
    for sample in samples:
        sample_evaluations.append(
            {
                "conditions": sample.get("conditions", {}),
                "text": sample.get("text", ""),
                "analysis": score_sample(sample.get("text", ""), sample.get("conditions", {})),
            }
        )

    if sample_evaluations:
        avg_scores = {}
        score_keys = sample_evaluations[0]["analysis"]["scores"].keys()
        for key in score_keys:
            avg_scores[key] = round(
                sum(item["analysis"]["scores"][key] for item in sample_evaluations) / len(sample_evaluations),
                2,
            )
    else:
        avg_scores = {}

    summary_parts = []
    if train_loss is not None and eval_loss is not None:
        summary_parts.append(f"train_loss={train_loss:.4f}, eval_loss={eval_loss:.4f}")
    if avg_scores:
        summary_parts.append(
            " | ".join(
                [
                    f"lyricness={avg_scores['lyricness']}",
                    f"conditioning={avg_scores['conditioning_fidelity']}",
                    f"bilingual={avg_scores['bilingual_mixing']}",
                ]
            )
        )

    result = "promising"
    if avg_scores:
        if avg_scores.get("conditioning_fidelity", 0) <= 2 and avg_scores.get("bilingual_mixing", 0) <= 2:
            result = "weak-conditioning"
        elif avg_scores.get("lyricness", 0) < 5:
            result = "weak-lyricness"

    next_hypothesis = "prompt format을 강화하고 epoch를 늘려 conditioning fidelity와 bilingual mixing을 개선한다."
    if avg_scores and avg_scores.get("bilingual_mixing", 0) <= 2:
        next_hypothesis = "prompt에 bilingual style 지시를 추가해 한국어/영어 혼합 출력을 유도한다."

    return {
        "result": result,
        "summary": "; ".join(summary_parts),
        "sample_evaluations": sample_evaluations,
        "aggregate_scores": avg_scores,
        "next_hypothesis": next_hypothesis,
    }


def write_run(path, run_data):
    with Path(path).resolve().open("w", encoding="utf-8") as fp:
        json.dump(run_data, fp, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    run_path = Path(args.run).resolve()
    run_data = read_run(run_path)
    run_data["evaluation"] = summarize_run(run_data)
    write_run(run_path, run_data)
    print(f"Updated evaluation in {run_path}")
    print(json.dumps(run_data["evaluation"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
