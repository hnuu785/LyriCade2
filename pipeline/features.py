import json
from pathlib import Path

import pandas as pd

from pipeline.rhyme import normalize_lyrics_for_rhyme


NUMERICAL_COLUMNS = [
    "BPM",
    "LINE_COUNT",
    "AVG_CHARS_PER_LINE",
]
RAW_NUMERICAL_COLUMNS = ["BPM", "LINE_COUNT", "AVG_CHARS_PER_LINE"]
SCALED_NUMERICAL_COLUMNS = [column for column in NUMERICAL_COLUMNS if column not in RAW_NUMERICAL_COLUMNS]

CATEGORICAL_COLUMNS = ["ARTIST", "BPM_CLASS", "DENSITY_CLASS"]

FEATURE_WEIGHTS = {}

DEFAULT_FEATURES = {
    "BPM": 92.0,
    "LINE_COUNT": 8.0,
    "AVG_CHARS_PER_LINE": 10.0,
    "BPM_CLASS": "붐뱁",
    "DENSITY_CLASS": "낮은 밀도",
}

REQUIRED_COLUMNS = {"TITLE", "ARTIST", "LYRICS", "BPM", "BPM_CLASS", "LINE_COUNT", "AVG_CHARS_PER_LINE", "DENSITY_CLASS"}
FEATURE_STATE_FILENAME = "feature_transform.json"
STYLE_TAG = "한국어 영어 혼용 랩 가사"
BAR_RULE_TAG = "한 줄은 한 마디로 끊기"
LYRICS_MARKER = "<가사>:"


def load_cleaned_dataset(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = df.columns.str.upper()

    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    prepared = pd.DataFrame()
    prepared["TITLE"] = df["TITLE"].astype(str).str.strip()
    prepared["ARTIST"] = df["ARTIST"].astype(str).str.strip()
    prepared["LYRICS"] = df["LYRICS"].fillna("").apply(normalize_lyrics_for_rhyme)
    prepared["BPM"] = pd.to_numeric(df["BPM"], errors="coerce")
    prepared["LINE_COUNT"] = pd.to_numeric(df["LINE_COUNT"], errors="coerce")
    prepared["AVG_CHARS_PER_LINE"] = pd.to_numeric(df["AVG_CHARS_PER_LINE"], errors="coerce")
    prepared["BPM_CLASS"] = df["BPM_CLASS"].astype(str).str.strip()
    prepared["DENSITY_CLASS"] = df["DENSITY_CLASS"].astype(str).str.strip()

    prepared["BPM"] = prepared["BPM"].fillna(DEFAULT_FEATURES["BPM"])
    prepared["LINE_COUNT"] = prepared["LINE_COUNT"].fillna(DEFAULT_FEATURES["LINE_COUNT"])
    prepared["AVG_CHARS_PER_LINE"] = prepared["AVG_CHARS_PER_LINE"].fillna(DEFAULT_FEATURES["AVG_CHARS_PER_LINE"])

    for column in NUMERICAL_COLUMNS:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
        prepared[column] = prepared[column].fillna(DEFAULT_FEATURES.get(column, prepared[column].median()))

    prepared = prepared[prepared["LYRICS"].str.len() > 0].reset_index(drop=True)
    return prepared


def fit_feature_transform(train_df):
    means = {}
    scales = {}

    for column in SCALED_NUMERICAL_COLUMNS:
        mean = float(train_df[column].mean())
        scale = float(train_df[column].std(ddof=0))
        if scale == 0.0:
            scale = 1.0
        means[column] = mean
        scales[column] = scale

    return {
        "numerical_columns": NUMERICAL_COLUMNS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "feature_weights": FEATURE_WEIGHTS,
        "defaults": DEFAULT_FEATURES,
        "raw_numerical_columns": RAW_NUMERICAL_COLUMNS,
        "scaled_numerical_columns": SCALED_NUMERICAL_COLUMNS,
        "means": means,
        "scales": scales,
        "style_tag": STYLE_TAG,
        "bar_rule_tag": BAR_RULE_TAG,
        "prompt_marker": LYRICS_MARKER,
    }


def apply_feature_transform(df, feature_state):
    transformed = df.copy()
    for column in feature_state.get("scaled_numerical_columns", feature_state["numerical_columns"]):
        mean = feature_state["means"][column]
        scale = feature_state["scales"][column]
        value = (transformed[column] - mean) / scale
        transformed[column] = value * feature_state["feature_weights"].get(column, 1.0)
    return transformed


def build_prompt(feature_values):
    bpm_value = feature_values["BPM"]
    line_count = int(round(feature_values["LINE_COUNT"]))
    density_value = feature_values["DENSITY_CLASS"]
    prompt_lines = [
        f"<아티스트: {feature_values['ARTIST']}>",
        f"<BPM: {bpm_value:.0f}>",
        f"<비트 스타일: {feature_values['BPM_CLASS']}>",
        f"<줄 수: {line_count}줄>",
        f"<{BAR_RULE_TAG}>",
        f"<한 마디 글자 수: {density_value}>",
        f"<가사 스타일: {STYLE_TAG}>",
        LYRICS_MARKER,
    ]
    return "\n".join(prompt_lines)


def build_training_sequence(row):
    return f"{build_prompt(row)} {row['LYRICS']}".strip()


def build_completion_text(lyrics: str) -> str:
    return f" {lyrics}".rstrip()


def transform_inference_features(raw_values, feature_state):
    transformed = {}
    raw_numerical_columns = feature_state.get("raw_numerical_columns", [])
    scaled_numerical_columns = feature_state.get("scaled_numerical_columns", feature_state["numerical_columns"])

    for column in raw_numerical_columns:
        transformed[column] = float(raw_values.get(column, feature_state["defaults"].get(column, 0.0)))

    for column in scaled_numerical_columns:
        raw_value = float(raw_values.get(column, feature_state["defaults"].get(column, 0.0)))
        mean = feature_state["means"][column]
        scale = feature_state["scales"][column]
        transformed[column] = ((raw_value - mean) / scale) * feature_state["feature_weights"].get(column, 1.0)

    for column in feature_state["categorical_columns"]:
        transformed[column] = str(raw_values.get(column, feature_state["defaults"].get(column, "unknown")))

    return transformed


def save_feature_state(output_dir, feature_state):
    output_path = Path(output_dir).resolve() / FEATURE_STATE_FILENAME
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(feature_state, fp, ensure_ascii=True, indent=2, sort_keys=True)
    return output_path


def load_feature_state(path):
    with Path(path).resolve().open("r", encoding="utf-8") as fp:
        return json.load(fp)
