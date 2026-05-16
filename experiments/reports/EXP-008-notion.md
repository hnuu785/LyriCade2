## EXP-008

- 날짜: 2026-05-16
- 실험 ID: `EXP-008`
- 목적: Add rhyme-aware auxiliary loss
- 가설: 

### 변경한 것

- 모델: `Qwen/Qwen2.5-1.5B-Instruct`
- 데이터: `kor_verse_datasets.csv` 1134행
- 학습 설정: epoch 3, batch size 1, gradient accumulation 2, max length 512
- 라이밍 설정: rhyme loss weight 0.1, max rhyme positions 16
- 생성 설정: max length 512
- 생성 라인 수: 8

### 실행

- 학습 커맨드: `/usr/bin/python3 -m pipeline.train_sft --experiment-id EXP-008 --csv /content/LyriCade2/datasets/kor_verse_datasets.csv --output-dir /content/drive/MyDrive/LyriCade2/outputs/qwen-1_5b-lyrics-sft --metrics-dir /content/drive/MyDrive/LyriCade2/experiments/metrics --plots-dir /content/drive/MyDrive/LyriCade2/experiments/plots --model-name Qwen/Qwen2.5-1.5B-Instruct --epochs 3 --batch-size 1 --gradient-accumulation-steps 2 --max-length 512 --learning-rate 0.0002 --val-ratio 0.2 --seed 42 --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 --rhyme-loss-weight 0.1 --max-rhyme-positions 16`
- 실행 시간: train 약 3182.4332초 / eval 약 117.5637초
- 경고/에러: 치명적 에러 없음

### 결과

- train loss: `6.733920975403639`
- eval loss: `3.4384727478027344`

- 샘플 조건:
  - artist: `dynamicduo`
  - track_genre: `k-rap`
  - tempo: `118.0`
  - energy: `0.4`
  - valence: `0.1`

- 샘플 출력:

```text
Yeah
yeah
I don't know what's going on
Yeah
why are you always on my mind
I'm trying to figure it out
But I can't stop thinking 'bout you
You're the only one that I wanna be with
```

### 평가

- 잘된 점:
- 아쉬운 점:
- 조건 반영도:
- 한국어/영어 혼합 품질:
- 한 줄 결론:

### 다음 실험

- 다음 가설:
- 바꿀 변수:
- 유지할 변수:
