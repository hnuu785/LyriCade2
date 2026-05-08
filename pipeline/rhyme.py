import math
import re
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F


WORD_RE = re.compile(r"[가-힣]+")
LYRICS_MARKER = "<LYRICS>:"
HANGUL_BASE = 0xAC00
HANGUL_END = 0xD7A3
NUM_JUNGSEONG = 21
NUM_JONGSEONG = 28

VOWEL_FAMILY = {
    0: 0,
    1: 0,
    2: 1,
    3: 1,
    4: 2,
    5: 2,
    6: 3,
    7: 3,
    8: 4,
    9: 5,
    10: 5,
    11: 4,
    12: 6,
    13: 7,
    14: 8,
    15: 8,
    16: 7,
    17: 9,
    18: 10,
    19: 11,
    20: 12,
}

CODA_FAMILY = {
    0: 0,
    1: 1,
    2: 2,
    3: 1,
    4: 3,
    5: 4,
    6: 4,
    7: 5,
    8: 6,
    9: 7,
    10: 7,
    11: 7,
    12: 7,
    13: 7,
    14: 7,
    15: 7,
    16: 8,
    17: 9,
    18: 10,
    19: 11,
    20: 12,
    21: 13,
    22: 14,
    23: 15,
    24: 16,
    25: 17,
    26: 18,
    27: 19,
}


def normalize_lyrics_for_rhyme(text) -> str:
    if text is None:
        return ""
    if isinstance(text, float) and math.isnan(text):
        return ""

    lyrics = str(text).replace("\r\n", "\n").replace("\r", "\n")
    lyrics = re.sub(r"\s{3,}", "\n", lyrics)
    if "\n" not in lyrics:
        lyrics = re.sub(r"\s{2,}", "\n", lyrics)
    lyrics = re.sub(r"[ \t]+\n", "\n", lyrics)
    lyrics = re.sub(r"\n{2,}", "\n", lyrics)
    return lyrics.strip()


def _is_hangul_syllable(char: str) -> bool:
    if not char:
        return False
    codepoint = ord(char)
    return HANGUL_BASE <= codepoint <= HANGUL_END


def _decompose_syllable(char: str) -> Tuple[int, int]:
    offset = ord(char) - HANGUL_BASE
    jungseong = (offset % (NUM_JUNGSEONG * NUM_JONGSEONG)) // NUM_JONGSEONG
    jongseong = offset % NUM_JONGSEONG
    return jungseong, jongseong


def _word_to_korean_rhyme_features(word: str) -> List[Tuple[int, int]]:
    syllables = [char for char in word if _is_hangul_syllable(char)]
    if not syllables:
        return []

    features = []
    for syllable in syllables[-2:]:
        jungseong, jongseong = _decompose_syllable(syllable)
        features.append((VOWEL_FAMILY[jungseong], CODA_FAMILY[jongseong]))
    return features


def _rhyme_similarity(features_a: List[Tuple[int, int]], features_b: List[Tuple[int, int]]) -> float:
    if not features_a or not features_b:
        return 0.0

    last_a = features_a[-1]
    last_b = features_b[-1]
    score = 0.0

    if last_a[0] == last_b[0]:
        score += 0.75
    elif abs(last_a[0] - last_b[0]) == 1:
        score += 0.35

    if last_a[1] == last_b[1]:
        score += 0.15
    elif last_a[1] != 0 and last_b[1] != 0:
        score += 0.05

    if len(features_a) > 1 and len(features_b) > 1:
        prev_a = features_a[-2]
        prev_b = features_b[-2]
        if prev_a[0] == prev_b[0]:
            score += 0.20
        elif abs(prev_a[0] - prev_b[0]) == 1:
            score += 0.10
        if prev_a[1] == prev_b[1]:
            score += 0.05

    return min(score, 1.0)


def _extract_line_end_spans(sequence: str, marker: str = LYRICS_MARKER) -> List[Tuple[int, int, List[Tuple[int, int]]]]:
    marker_index = sequence.find(marker)
    if marker_index == -1:
        lyrics_start = 0
        lyrics = sequence
    else:
        lyrics_start = marker_index + len(marker)
        if lyrics_start < len(sequence) and sequence[lyrics_start] == " ":
            lyrics_start += 1
        lyrics = sequence[lyrics_start:]

    if not lyrics.strip():
        return []

    spans = []
    for line_match in re.finditer(r"[^\n]+", lyrics):
        words = list(WORD_RE.finditer(line_match.group(0)))
        if not words:
            continue

        end_word = words[-1]
        features = _word_to_korean_rhyme_features(end_word.group(0))
        if not features:
            continue

        line_offset = line_match.start()
        start = lyrics_start + line_offset + end_word.start()
        end = lyrics_start + line_offset + end_word.end()
        spans.append((start, end, features))

    return spans


def _map_spans_to_token_positions(
    offset_mapping: Sequence[Tuple[int, int]],
    spans: Sequence[Tuple[int, int, List[Tuple[int, int]]]],
    max_rhyme_positions: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    token_positions = []
    rhyme_features = []

    for start, end, features in spans:
        matching_tokens = []
        for token_index, (token_start, token_end) in enumerate(offset_mapping):
            if token_start == token_end:
                continue
            if token_end > start and token_start < end:
                matching_tokens.append(token_index)

        if matching_tokens:
            token_positions.append(matching_tokens[-1])
            rhyme_features.append(features)

    deduped_positions = []
    deduped_features = []
    seen_positions = set()
    for position, features in zip(token_positions, rhyme_features):
        if position in seen_positions:
            continue
        seen_positions.add(position)
        deduped_positions.append(position)
        deduped_features.append(features)

    deduped_positions = deduped_positions[:max_rhyme_positions]
    deduped_features = deduped_features[:max_rhyme_positions]

    padded_positions = torch.full((max_rhyme_positions,), -1, dtype=torch.long)
    target_matrix = torch.full((max_rhyme_positions, max_rhyme_positions), -1.0, dtype=torch.float32)

    if deduped_positions:
        padded_positions[: len(deduped_positions)] = torch.tensor(deduped_positions, dtype=torch.long)
        for row_idx, row_features in enumerate(deduped_features):
            for col_idx, col_features in enumerate(deduped_features):
                if row_idx == col_idx:
                    target_matrix[row_idx, col_idx] = 1.0
                else:
                    target_matrix[row_idx, col_idx] = _rhyme_similarity(row_features, col_features)

    return padded_positions, target_matrix


def prepare_rhyme_supervision(
    sequences: Sequence[str],
    tokenizer,
    max_length: int = 512,
    max_rhyme_positions: int = 16,
):
    encodings = tokenizer(
        list(sequences),
        padding="longest",
        truncation=True,
        return_tensors="pt",
        max_length=max_length,
        return_offsets_mapping=True,
    )

    rhyme_positions = []
    rhyme_targets = []
    for sequence, offset_mapping in zip(sequences, encodings["offset_mapping"].tolist()):
        spans = _extract_line_end_spans(sequence)
        positions, targets = _map_spans_to_token_positions(offset_mapping, spans, max_rhyme_positions)
        rhyme_positions.append(positions)
        rhyme_targets.append(targets)

    encodings.pop("offset_mapping")
    return encodings, torch.stack(rhyme_positions), torch.stack(rhyme_targets)


def compute_rhyme_loss(
    hidden_states: torch.Tensor,
    rhyme_positions: torch.Tensor,
    rhyme_targets: torch.Tensor,
) -> torch.Tensor:
    sample_losses = []

    for sample_hidden_states, sample_positions, sample_targets in zip(hidden_states, rhyme_positions, rhyme_targets):
        valid_mask = sample_positions >= 0
        positions = sample_positions[valid_mask]
        if positions.numel() < 2:
            continue

        in_range_mask = positions < sample_hidden_states.size(0)
        positions = positions[in_range_mask]
        if positions.numel() < 2:
            continue

        valid_count = positions.numel()
        target_matrix = sample_targets[:valid_count, :valid_count]

        rhyme_vectors = F.normalize(sample_hidden_states[positions], dim=-1)
        similarity_matrix = rhyme_vectors @ rhyme_vectors.T

        pair_mask = ~torch.eye(valid_count, dtype=torch.bool, device=sample_hidden_states.device)
        masked_targets = target_matrix.to(sample_hidden_states.device)
        valid_target_mask = (masked_targets >= 0) & pair_mask
        if not valid_target_mask.any():
            continue

        pairwise_error = (similarity_matrix - masked_targets) ** 2
        weighted_error = pairwise_error[valid_target_mask] * (0.25 + masked_targets[valid_target_mask])
        sample_losses.append(weighted_error.mean())

    if not sample_losses:
        return hidden_states.new_tensor(0.0)

    return torch.stack(sample_losses).mean()
