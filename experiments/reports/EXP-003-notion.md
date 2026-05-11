## EXP-003

- 날짜: 2026-05-09
- 실험 ID: `EXP-003`
- 목적: Increase epochs and context length on the new dataset
- 가설: 

### 변경한 것

- 모델: `Qwen/Qwen2.5-1.5B-Instruct`
- 데이터: `kor_verse_datasets.csv` 1134행
- 학습 설정: epoch 5, batch size 1, gradient accumulation 2, max length 512
- 생성 설정: max length 512

### 실행

- 학습 커맨드: `/usr/bin/python3 -m pipeline.train_sft --experiment-id EXP-003 --csv /content/LyriCade2/datasets/kor_verse_datasets.csv --output-dir /content/drive/MyDrive/LyriCade2/outputs/qwen-1_5b-lyrics-sft --metrics-dir /content/drive/MyDrive/LyriCade2/experiments/metrics --plots-dir /content/drive/MyDrive/LyriCade2/experiments/plots --model-name Qwen/Qwen2.5-1.5B-Instruct --epochs 5 --batch-size 1 --gradient-accumulation-steps 2 --max-length 512 --learning-rate 0.0002 --val-ratio 0.2 --seed 42 --lora-r 16 --lora-alpha 32 --lora-dropout 0.05`
- 실행 시간: train 약 4619.7468초 / eval 약 98.8161초
- 경고/에러: 치명적 에러 없음

### 결과

- train loss: `3.246736639081644`
- eval loss: `3.4140284061431885`

- 샘플 조건:
  - artist: `dynamicduo`
  - track_genre: `k-rap`
  - tempo: `118.0`
  - energy: `0.4`
  - valence: `0.1`

- 샘플 출력:

```text
Yeah, yeah, I don't need nothin'
난 돈을 벌고 싶지 않아도 돼
내가 원하는 게 있다면
그게 뭐든 돈다고 말할 수 있어
Yeah, 난 뭔가를 원해
그래서 그걸 얻어낼 수 있는 방법을 찾아가
I'm on a mission to get what I want
나의 목표를 이루기 위해
우리 둘만이 필요하다고 믿어
너무 많은 시간을 써도 되는 거야
다시 말해 너는 나를 위해 필요한 게 없다는 뜻이지
아무것도 필요하지 않으니까
난 내게 필요한 것들만을 배우고
자신을 키워 나아갈 수 있게 돕기로 했어 (Yeah)
내게 도움을 청하길 바란다면
네가 필요로 하는 건 단 한 가지뿐이야 (One)
그것이 바로 나와 함께하는 놀라운 순간을 보여줄 수 있기를 바래 (Wanna see that moment with me)
나는 네게 어떤 걸 원했는지 몰라
하지만 내가 원했던 건 넌 아니었을 뿐이니까 (No, no)
난 너를 위한 노래를 불러 (Yuh)
넌 나의 곁에 있으면서도 (Uh)
너의 마음을 움직일 수 없을 것 같아 (Uhh)
아
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
