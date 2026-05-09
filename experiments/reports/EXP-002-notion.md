## EXP-002

- 날짜: 2026-05-09
- 실험 ID: `EXP-002`
- 목적: Same hyperparameters with new dataset file
- 가설: 

### 변경한 것

- 모델: `Qwen/Qwen2.5-1.5B-Instruct`
- 데이터: `kor_verse_datasets.csv` 1134행
- 학습 설정: epoch 1, batch size 1, gradient accumulation 2, max length 256
- 생성 설정: max length 256

### 실행

- 학습 커맨드: `/usr/bin/python3 -m pipeline.train_sft --experiment-id EXP-002 --csv /content/LyriCade2/datasets/kor_verse_datasets.csv --output-dir /content/drive/MyDrive/LyriCade2/outputs/qwen-1_5b-lyrics-sft --metrics-dir /content/drive/MyDrive/LyriCade2/experiments/metrics --plots-dir /content/drive/MyDrive/LyriCade2/experiments/plots --model-name Qwen/Qwen2.5-1.5B-Instruct --epochs 1 --batch-size 1 --gradient-accumulation-steps 2 --max-length 256 --learning-rate 0.0002 --val-ratio 0.2 --seed 42 --lora-r 16 --lora-alpha 32 --lora-dropout 0.05`
- 실행 시간: train 약 649.2831초 / eval 약 77.5211초
- 경고/에러: 치명적 에러 없음

### 결과

- train loss: `3.6217984817101567`
- eval loss: `3.562471389770508`

- 샘플 조건:
  - artist: `dynamicduo`
  - track_genre: `k-rap`
  - tempo: `118.0`
  - energy: `0.4`
  - valence: `0.1`

- 샘플 출력:

```text
난 빛을 쏟아내고, 내 손을 잡아주지
난 뭘 해도 끝나지 않아, 나의 꿈을
나의 삶을 바꾸고 싶어, 그저 내 목표를
내게 떠오르는
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
