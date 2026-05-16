## EXP-006

- 날짜: 2026-05-15
- 실험 ID: `EXP-006`
- 목적: Compare Qwen 3B against the 1.5B baseline
- 가설: 

### 변경한 것

- 모델: `Qwen/Qwen2.5-1.5B-Instruct`
- 데이터: `kor_verse_datasets.csv` 1134행
- 학습 설정: epoch 10, batch size 1, gradient accumulation 2, max length 512
- 생성 설정: max length 512
- 생성 라인 수: 8

### 실행

- 학습 커맨드: `/usr/bin/python3 -m pipeline.train_sft --experiment-id EXP-006 --csv /content/LyriCade2/datasets/kor_verse_datasets.csv --output-dir /content/drive/MyDrive/LyriCade2/outputs/qwen-1_5b-lyrics-sft --metrics-dir /content/drive/MyDrive/LyriCade2/experiments/metrics --plots-dir /content/drive/MyDrive/LyriCade2/experiments/plots --model-name Qwen/Qwen2.5-1.5B-Instruct --epochs 10 --batch-size 1 --gradient-accumulation-steps 2 --max-length 512 --learning-rate 0.0002 --val-ratio 0.2 --seed 42 --lora-r 16 --lora-alpha 32 --lora-dropout 0.05`
- 실행 시간: train 약 9617.7997초 / eval 약 104.9854초
- 경고/에러: 치명적 에러 없음

### 결과

- train loss: `2.9057572486642176`
- eval loss: `3.4166667461395264`

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
I'm on my way
나는 빠르게 움직이고 있어
내 속도는 멈추지 않아
I'm goin' on a trip
여전히 내 발걸음은 멀리 뻗어있어
어느 곳에 갈지 몰라도
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
