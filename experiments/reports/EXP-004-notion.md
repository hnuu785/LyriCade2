## EXP-004

- 날짜: 2026-05-11
- 실험 ID: `EXP-004`
- 목적: Add fixed style tag to prompt schema
- 가설: 

### 변경한 것

- 모델: `Qwen/Qwen2.5-1.5B-Instruct`
- 데이터: `kor_verse_datasets.csv` 1134행
- 학습 설정: epoch 5, batch size 1, gradient accumulation 2, max length 512
- 생성 설정: max length 512

### 실행

- 학습 커맨드: `/usr/bin/python3 -m pipeline.train_sft --experiment-id EXP-004 --csv /content/LyriCade2/datasets/kor_verse_datasets.csv --output-dir /content/drive/MyDrive/LyriCade2/outputs/qwen-1_5b-lyrics-sft --metrics-dir /content/drive/MyDrive/LyriCade2/experiments/metrics --plots-dir /content/drive/MyDrive/LyriCade2/experiments/plots --model-name Qwen/Qwen2.5-1.5B-Instruct --epochs 5 --batch-size 1 --gradient-accumulation-steps 2 --max-length 512 --learning-rate 0.0002 --val-ratio 0.2 --seed 42 --lora-r 16 --lora-alpha 32 --lora-dropout 0.05`
- 실행 시간: train 약 4728.1834초 / eval 약 103.5795초
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
네가 뭘 해도 돼
내가 너를 향해 날아갈 거야
너의 손을 잡아줘
난 너의 꿈을 꾸게 해줄게
어쩌면 너는 내게 도움이 될지도 몰라
아무리 봐도 너와 나의 관계는 끝나지 않겠지
네가 나를 잊어버리기 전에
나는 널 보고 싶어
그러니 넌 나에게서 멀어지지 말고
다시 내 앞에 와줘, come back to me
I'll be here for you, I'll always be with you
넌 내가 원했던 걸 보여줄 수 있을까?
너가 원하는 건 무엇일까? I don't know
But I know that you're the only one I want
You're my one and only love
그래, 너가 필요할 때면 나와 함께할 수 있으면 좋겠어 (Yeah, yeah)
그런 것만으로도 내 삶이 풍요롭게 될 거라고 믿어, baby
사랑이란 게 얼마나 아름다운지 깨닫는 순간
우리의 사랑은 더 빛나게 뜨거워질 거란 걸 알게 됐다
오늘 밤이야, 난 너에게로 가는 길을 걷고 있어
밤이
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
