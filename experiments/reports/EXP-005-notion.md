## EXP-005

- 날짜: 2026-05-14
- 실험 ID: `EXP-005`
- 목적: Generate 8-line formatted lyric samples
- 가설: 

### 변경한 것

- 모델: `Qwen/Qwen2.5-1.5B-Instruct`
- 데이터: `kor_verse_datasets.csv` 1134행
- 학습 설정: epoch 5, batch size 1, gradient accumulation 2, max length 512
- 생성 설정: max length 512
- 생성 라인 수: 8

### 실행

- 학습 커맨드: `/usr/bin/python3 -m pipeline.train_sft --experiment-id EXP-005 --csv /content/LyriCade2/datasets/kor_verse_datasets.csv --output-dir /content/drive/MyDrive/LyriCade2/outputs/qwen-1_5b-lyrics-sft --metrics-dir /content/drive/MyDrive/LyriCade2/experiments/metrics --plots-dir /content/drive/MyDrive/LyriCade2/experiments/plots --model-name Qwen/Qwen2.5-1.5B-Instruct --epochs 5 --batch-size 1 --gradient-accumulation-steps 2 --max-length 512 --learning-rate 0.0002 --val-ratio 0.2 --seed 42 --lora-r 16 --lora-alpha 32 --lora-dropout 0.05`
- 실행 시간: train 약 4596.7554초 / eval 약 97.8615초
- 경고/에러: 치명적 에러 없음

### 결과

- train loss: `3.2470315395472857`
- eval loss: `3.4175400733947754`

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
I don't know what's wrong with me
But I feel like I'm on top of the world
Yeah
it's like a dream come true
And I can't wait to see what tomorrow has in store
I've been through so much
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
