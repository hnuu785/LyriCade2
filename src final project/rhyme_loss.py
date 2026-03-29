import math
import re
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F


WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
LYRICS_MARKER = "<LYRICS>: "


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


def _fallback_rhyme_key(word: str) -> str:
    cleaned = re.sub(r"[^a-z']", "", word.lower())
    if not cleaned:
        return ""

    cleaned = re.sub(r"n'$", "ng", cleaned)

    if len(cleaned) > 4:
        cleaned = re.sub(r"(ing|edly|ed|es|s)$", "", cleaned)
    if len(cleaned) > 3 and cleaned.endswith("e"):
        cleaned = cleaned[:-1]

    vowels = "aeiouy"
    last_vowel = max((idx for idx, char in enumerate(cleaned) if char in vowels), default=-1)
    if last_vowel == -1:
        return cleaned[-3:]

    start = last_vowel
    while start > 0 and cleaned[start - 1] in vowels:
        start -= 1

    rhyme_key = cleaned[start:]
    if len(rhyme_key) < 2 and len(cleaned) >= 3:
        rhyme_key = cleaned[-3:]
    return rhyme_key[-5:]


def word_to_rhyme_key(word: str) -> str:
    try:
        import pronouncing

        phones = pronouncing.phones_for_word(word.lower())
        if phones:
            rhyme_part = pronouncing.rhyming_part(phones[0])
            if rhyme_part:
                return rhyme_part
    except Exception:
        pass

    return _fallback_rhyme_key(word)


def _extract_line_end_spans(sequence: str, marker: str = LYRICS_MARKER) -> List[Tuple[int, int, str]]:
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
        line_offset = line_match.start()
        start = lyrics_start + line_offset + end_word.start()
        end = lyrics_start + line_offset + end_word.end()
        rhyme_key = word_to_rhyme_key(end_word.group(0))
        if rhyme_key:
            spans.append((start, end, rhyme_key))

    return spans


def _map_spans_to_token_positions(
    offset_mapping: Sequence[Tuple[int, int]], spans: Sequence[Tuple[int, int, str]], max_rhyme_positions: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    token_positions = []
    rhyme_keys = []

    for start, end, rhyme_key in spans:
        matching_tokens = []
        for token_index, (token_start, token_end) in enumerate(offset_mapping):
            if token_start == token_end:
                continue
            if token_end > start and token_start < end:
                matching_tokens.append(token_index)

        if matching_tokens:
            token_positions.append(matching_tokens[-1])
            rhyme_keys.append(rhyme_key)

    deduped_positions = []
    deduped_keys = []
    seen_positions = set()
    for position, rhyme_key in zip(token_positions, rhyme_keys):
        if position in seen_positions:
            continue
        seen_positions.add(position)
        deduped_positions.append(position)
        deduped_keys.append(rhyme_key)

    deduped_positions = deduped_positions[:max_rhyme_positions]
    deduped_keys = deduped_keys[:max_rhyme_positions]

    group_lookup = {}
    next_group_id = 0
    group_ids = []
    for rhyme_key in deduped_keys:
        if rhyme_key not in group_lookup:
            group_lookup[rhyme_key] = next_group_id
            next_group_id += 1
        group_ids.append(group_lookup[rhyme_key])

    padded_positions = torch.full((max_rhyme_positions,), -1, dtype=torch.long)
    padded_group_ids = torch.full((max_rhyme_positions,), -1, dtype=torch.long)

    if deduped_positions:
        padded_positions[: len(deduped_positions)] = torch.tensor(deduped_positions, dtype=torch.long)
        padded_group_ids[: len(group_ids)] = torch.tensor(group_ids, dtype=torch.long)

    return padded_positions, padded_group_ids


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
    rhyme_group_ids = []

    for sequence, offset_mapping in zip(sequences, encodings["offset_mapping"].tolist()):
        spans = _extract_line_end_spans(sequence)
        positions, groups = _map_spans_to_token_positions(offset_mapping, spans, max_rhyme_positions)
        rhyme_positions.append(positions)
        rhyme_group_ids.append(groups)

    encodings.pop("offset_mapping")
    return encodings, torch.stack(rhyme_positions), torch.stack(rhyme_group_ids)


def compute_rhyme_loss(
    hidden_states: torch.Tensor,
    rhyme_positions: torch.Tensor,
    rhyme_group_ids: torch.Tensor,
    similarity_scale: float = 10.0,
) -> torch.Tensor:
    sample_losses = []

    for sample_hidden_states, sample_positions, sample_groups in zip(hidden_states, rhyme_positions, rhyme_group_ids):
        valid_mask = sample_positions >= 0
        positions = sample_positions[valid_mask]
        groups = sample_groups[valid_mask]

        if positions.numel() < 2:
            continue

        in_range_mask = positions < sample_hidden_states.size(0)
        positions = positions[in_range_mask]
        groups = groups[in_range_mask]
        if positions.numel() < 2:
            continue

        pair_count = positions.numel()
        pair_mask = ~torch.eye(pair_count, dtype=torch.bool, device=sample_hidden_states.device)
        positive_mask = (groups[:, None] == groups[None, :]) & pair_mask
        negative_mask = (groups[:, None] != groups[None, :]) & pair_mask

        rhyme_vectors = F.normalize(sample_hidden_states[positions], dim=-1)
        similarity_logits = similarity_scale * (rhyme_vectors @ rhyme_vectors.T)

        loss_terms = []
        if positive_mask.any():
            loss_terms.append(F.softplus(-similarity_logits[positive_mask]).mean())
        if negative_mask.any():
            loss_terms.append(F.softplus(similarity_logits[negative_mask]).mean())
        if loss_terms:
            sample_losses.append(torch.stack(loss_terms).mean())

    if not sample_losses:
        return hidden_states.new_tensor(0.0)

    return torch.stack(sample_losses).mean()
