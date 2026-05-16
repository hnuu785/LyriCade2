## EXP-007

- 날짜: 2026-05-16
- 실험 ID: `EXP-007`
- 목적: Refine prompt wording for more Korean-centered k-rap output
- 가설: 

### 변경한 것

- 모델: `Qwen/Qwen2.5-1.5B-Instruct`
- 데이터: `kor_verse_datasets.csv` 1134행
- 학습 설정: epoch 3, batch size 1, gradient accumulation 2, max length 512
- 생성 설정: max length 512
- 생성 라인 수: 8

### 실행

- 학습 커맨드: `/usr/bin/python3 -m pipeline.train_sft --experiment-id EXP-007 --csv /content/LyriCade2/datasets/kor_verse_datasets.csv --output-dir /content/drive/MyDrive/LyriCade2/outputs/qwen-1_5b-lyrics-sft --metrics-dir /content/drive/MyDrive/LyriCade2/experiments/metrics --plots-dir /content/drive/MyDrive/LyriCade2/experiments/plots --model-name Qwen/Qwen2.5-1.5B-Instruct --epochs 3 --batch-size 1 --gradient-accumulation-steps 2 --max-length 512 --learning-rate 0.0002 --val-ratio 0.2 --seed 42 --lora-r 16 --lora-alpha 32 --lora-dropout 0.05`
- 실행 시간: train 약 2938.1466초 / eval 약 107.6062초
- 경고/에러: 치명적 에러 없음

### 결과

- train loss: `3.3745891095608167`
- eval loss: `3.4243879318237305`

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
I'm the king of the world
I don't care what they say
Yeah
you can't stop me
it's my time to shine
And I ain't got nothin' to hide
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
