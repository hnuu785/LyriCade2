# Korean Lyric Fine-Tuning Pipeline

이 저장소는 한국어 가사 생성을 위한 LLM 파인튜닝 실험에 집중하도록 정리했다.  
기존 데모 앱, 오디오 특성 추출 노트북, 학기 프로젝트 산출물은 실행 경로에서 분리하고, 현재는 아래 세 단계만 유지한다.

1. 데이터 정제 및 verse 추출
2. feature-conditioned causal LM 파인튜닝
3. 저장된 체크포인트로 샘플 가사 생성

## Active Structure

```text
pipeline/
  prepare_dataset.py
  features.py
  rhyme.py
  train_sft.py
  generate_sft.py
datasets/
docs/
experiments/
scripts/
requirements.txt
README.md
```

`archive/` 는 기존 실험 자산을 옮겨두는 보관 위치다. 학습 파이프라인 실행에는 필요하지 않다.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

macOS 메모:

- Apple Silicon/macOS 환경에서는 `torch` wheel 설치가 오래 걸리거나 환경에 따라 별도 안내가 필요할 수 있다.
- `pip install -r requirements.txt` 중 `torch` 설치가 실패하면 먼저 PyTorch 공식 설치 가이드를 따라 `torch`를 설치한 뒤, 나머지 패키지를 다시 설치하는 편이 안전하다.
- 가상환경을 쓰지 않으면 시스템 Python과 패키지가 섞이기 쉬우니 `.venv` 사용을 권장한다.

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

현재 기본 권장 학습 경로는 `pipeline.train_sft` 이다.  
기본 베이스라인 모델은 `Qwen/Qwen2.5-7B-Instruct` 로 정했다.

이 선택 배경과 공통 실험 원칙은 [docs/experiment-overview.md](/Users/cho/Developer/PMTMProject/LyricGeneration/docs/experiment-overview.md) 에 정리했다.

### Recommended: LoRA SFT with Qwen

```bash
python -m pipeline.train_sft \
  --csv /absolute/path/to/train_kor_data_verse_only.csv \
  --output-dir /absolute/path/to/outputs/qwen-lyrics-sft \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --epochs 2 \
  --batch-size 2
```

실제 저장 경로는 `--output-dir/YYYY-MM-DD/` 형태다. 예를 들어 2026년 5월 7일에 실행하면
`/absolute/path/to/outputs/qwen-lyrics-sft/2026-05-07/` 아래에 체크포인트가 저장된다.

현재 권장 학습 스크립트 특징:

- Qwen 계열 causal LM + LoRA SFT
- prompt/completion 분리 라벨링
- 숫자형 음악 feature를 프롬프트 토큰으로 직렬화
- train/inference 공통 feature transform 저장
- `final_adapter`, `final_tokenizer`, `feature_transform.json` 저장

## 3) Generate

### Generate from LoRA SFT

```bash
python -m pipeline.generate_sft \
  --adapter-dir /absolute/path/to/outputs/qwen-lyrics-sft/2026-05-07/final_adapter \
  --tokenizer-dir /absolute/path/to/outputs/qwen-lyrics-sft/2026-05-07/final_tokenizer \
  --artist dynamicduo \
  --track-genre k-rap \
  --tempo 118 \
  --energy 0.4 \
  --valence 0.1
```

## Notes

- 현재 기본 베이스 모델은 `Qwen/Qwen2.5-7B-Instruct` 다.
- 프로젝트 목표는 한국어 전용 모델보다 한국어/영어 혼합 가사 생성에 맞는 범용 다국어 모델을 쓰는 것이다.
- `pipeline.train_sft` / `pipeline.generate_sft` 가 현재 학습/생성 기본 경로다.
- 자동 실험 누적을 위한 기계 판독용 기록은 `experiments/`, 실행/기록 스크립트는 `scripts/` 아래에 둔다.
- 기존 실험 코드가 필요하면 `archive/` 에서 다시 꺼내면 된다.
