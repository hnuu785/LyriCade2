# Korean Lyric Fine-Tuning Pipeline

이 저장소는 한국어 가사 생성을 위한 LLM 파인튜닝 실험에 집중하도록 정리했다.  
기존 데모 앱, 오디오 특성 추출 노트북, 학기 프로젝트 산출물은 실행 경로에서 분리하고, 현재는 아래 세 단계만 유지한다.

1. 데이터 정제 및 verse 추출
2. feature-conditioned GPT 계열 모델 파인튜닝
3. 저장된 체크포인트로 샘플 가사 생성

## Active Structure

```text
pipeline/
  prepare_dataset.py
  rhyme.py
  train.py
  generate.py
datasets/
requirements.txt
README.md
```

`archive/` 는 기존 실험 자산을 옮겨두는 보관 위치다. 학습 파이프라인 실행에는 필요하지 않다.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1) Prepare Dataset

입력 CSV는 최소한 아래 컬럼을 가져야 한다.

- `title`
- `artist`
- `lyrics`
- `bpm`
- `energy`
- `danceability`
- `loudness`
- `valence`

verse 구간만 남긴 학습용 CSV를 만들려면:

```bash
python -m pipeline.prepare_dataset \
  --input /absolute/path/to/train_kor_data.csv \
  --output /absolute/path/to/train_kor_data_annotated.csv \
  --strict-output /absolute/path/to/train_kor_data_verse_only.csv \
  --fallback-output /absolute/path/to/train_kor_data_verse_fallback.csv
```

- `strict-output`: `[Verse]` 같은 태그가 있는 가사만 사용
- `fallback-output`: 태그가 없는 가사는 원문 전체를 유지

## 2) Train

```bash
python -m pipeline.train \
  --csv /absolute/path/to/train_kor_data_verse_only.csv \
  --output-dir /absolute/path/to/outputs/korean-lyrics-gpt2 \
  --model-name gpt2 \
  --epochs 2 \
  --batch-size 4
```

현재 학습 스크립트 특징:

- 숫자형 음악 feature를 프롬프트 토큰으로 직렬화
- 한국어 라임 유사도를 반영한 보조 loss 사용
- `best_model`, `best_tokenizer`, `final_model`, `final_tokenizer` 저장

## 3) Generate

```bash
python -m pipeline.generate \
  --model-dir /absolute/path/to/outputs/korean-lyrics-gpt2/final_model \
  --tokenizer-dir /absolute/path/to/outputs/korean-lyrics-gpt2/final_tokenizer \
  --artist dynamicduo \
  --track-genre k-rap \
  --tempo 118 \
  --energy 0.4 \
  --valence 0.1
```

## Notes

- 현재 기본 베이스 모델은 `gpt2`다. 한국어 성능이 목적이면 이후에는 한국어 tokenizer/model 계열로 교체하는 편이 맞다.
- 기존 실험 코드가 필요하면 `archive/` 에서 다시 꺼내면 된다.
