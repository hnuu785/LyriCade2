import json
from pathlib import Path

import pandas as pd

from pipeline.rhyme import normalize_lyrics_for_rhyme


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

REQUIRED_COLUMNS = {"TITLE", "ARTIST", "LYRICS", "BPM", "ENERGY", "DANCEABILITY", "LOUDNESS", "VALENCE"}
FEATURE_STATE_FILENAME = "feature_transform.json"
STYLE_TAG = "mostly korean k-rap verse with occasional english phrases"
FORMAT_TAG = "8 rap bars, one bar per line"
LYRICS_MARKER = "<LYRICS>:"


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
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
        prepared[column] = prepared[column].fillna(DEFAULT_FEATURES.get(column, prepared[column].median()))

    prepared = prepared[prepared["LYRICS"].str.len() > 0].reset_index(drop=True)
    return prepared


def fit_feature_transform(train_df):
    means = {}
    scales = {}

    for column in NUMERICAL_COLUMNS:
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
        "means": means,
        "scales": scales,
        "style_tag": STYLE_TAG,
        "format_tag": FORMAT_TAG,
        "prompt_marker": LYRICS_MARKER,
    }


def apply_feature_transform(df, feature_state):
    transformed = df.copy()
    for column in feature_state["numerical_columns"]:
        mean = feature_state["means"][column]
        scale = feature_state["scales"][column]
        value = (transformed[column] - mean) / scale
        transformed[column] = value * feature_state["feature_weights"].get(column, 1.0)
    return transformed


def build_prompt(feature_values):
    prompt_lines = [f"<{column}: {feature_values[column]:.2f}>" for column in NUMERICAL_COLUMNS]
    prompt_lines.extend(f"<{column}: {feature_values[column]}>" for column in CATEGORICAL_COLUMNS)
    prompt_lines.append(f"<STYLE: {STYLE_TAG}>")
    prompt_lines.append(f"<FORMAT: {FORMAT_TAG}>")
    prompt_lines.append(LYRICS_MARKER)
    return "\n".join(prompt_lines)


def build_training_sequence(row):
    return f"{build_prompt(row)} {row['LYRICS']}".strip()


def build_completion_text(lyrics: str) -> str:
    return f" {lyrics}".rstrip()


def transform_inference_features(raw_values, feature_state):
    transformed = {}
    for column in feature_state["numerical_columns"]:
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
