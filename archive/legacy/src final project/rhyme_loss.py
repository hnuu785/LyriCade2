import math
import re
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F


WORD_RE = re.compile(r"[가-힣]+")
LYRICS_MARKER = "<LYRICS>: "
HANGUL_BASE = 0xAC00
HANGUL_END = 0xD7A3
NUM_JUNGSEONG = 21
NUM_JONGSEONG = 28

# Relaxed vowel families work better for Korean rap than exact vowel identity.
VOWEL_FAMILY = {
    0: 0,   # ㅏ
    1: 0,   # ㅐ
    2: 1,   # ㅑ
    3: 1,   # ㅒ
    4: 2,   # ㅓ
    5: 2,   # ㅔ
    6: 3,   # ㅕ
    7: 3,   # ㅖ
    8: 4,   # ㅗ
    9: 5,   # ㅘ
    10: 5,  # ㅙ
    11: 4,  # ㅚ
    12: 6,  # ㅛ
    13: 7,  # ㅜ
    14: 8,  # ㅝ
    15: 8,  # ㅞ
    16: 7,  # ㅟ
    17: 9,  # ㅠ
    18: 10, # ㅡ
    19: 11, # ㅢ
    20: 12, # ㅣ
}

# Final consonants are grouped by rhyme-relevant phonetic closeness.
CODA_FAMILY = {
    0: 0,   # no coda
    1: 1,   # ㄱ
    2: 2,   # ㄲ
    3: 1,   # ㄳ
    4: 3,   # ㄴ
    5: 4,   # ㄵ
    6: 4,   # ㄶ
    7: 5,   # ㄷ
    8: 6,   # ㄹ
    9: 7,   # ㄺ
    10: 7,  # ㄻ
    11: 7,  # ㄼ
    12: 7,  # ㄽ
    13: 7,  # ㄾ
    14: 7,  # ㄿ
    15: 7,  # ㅀ
    16: 8,  # ㅁ
    17: 9,  # ㅂ
    18: 10, # ㅄ
    19: 11, # ㅅ
    20: 12, # ㅆ
    21: 13, # ㅇ
    22: 14, # ㅈ
    23: 15, # ㅊ
    24: 16, # ㅋ
    25: 17, # ㅌ
    26: 18, # ㅍ
    27: 19, # ㅎ
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


def _vowel_family(jungseong: int) -> int:
    return VOWEL_FAMILY[jungseong]


def _coda_family(jongseong: int) -> int:
    return CODA_FAMILY[jongseong]


def _word_to_korean_rhyme_features(word: str) -> List[Tuple[int, int]]:
    syllables = [char for char in word if _is_hangul_syllable(char)]
    if not syllables:
        return []

    features = []
    for syllable in syllables[-2:]:
        jungseong, jongseong = _decompose_syllable(syllable)
        features.append((_vowel_family(jungseong), _coda_family(jongseong)))
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
        lyrics = sequence[lyrics_start:]

    normalized_lyrics = normalize_lyrics_for_rhyme(lyrics)
    if not normalized_lyrics:
        return []

    spans = []
    for line_match in re.finditer(r"[^\n]+", normalized_lyrics):
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
    sequences: Sequence[str], tokenizer, max_length: int = 512, max_rhyme_positions: int = 16
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
