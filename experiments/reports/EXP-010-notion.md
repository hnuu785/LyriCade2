## EXP-010

- 날짜: 2026-05-17
- 실험 ID: `EXP-010`
- 목적: Natural Korean prompt for 8-bar nafla boom bap lyric generation
- 가설: 

### 변경한 것

- 모델: `Qwen/Qwen2.5-1.5B-Instruct`
- 데이터: `kor_verse_datasets_slim.csv` 1134행
- 학습 설정: epoch 4, batch size 1, gradient accumulation 4, max length 256
- 라이밍 설정: rhyme loss weight 0.0, max rhyme positions 16
- 생성 설정: max length 256

### 실행

- 학습 커맨드: `/usr/bin/python3 -m pipeline.train_sft --experiment-id EXP-010 --csv /content/LyriCade2/datasets/kor_verse_datasets_slim.csv --output-dir /content/LyriCade2/outputs/qwen-1_5b-lyrics-sft --metrics-dir /content/drive/MyDrive/LyriCade2/experiments/metrics --plots-dir /content/drive/MyDrive/LyriCade2/experiments/plots --model-name Qwen/Qwen2.5-1.5B-Instruct --epochs 4 --batch-size 1 --gradient-accumulation-steps 4 --max-length 256 --learning-rate 0.0002 --val-ratio 0.2 --seed 42 --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 --rhyme-loss-weight 0.0 --max-rhyme-positions 16`
- 실행 시간: train 약 2460.3817초 / eval 약 72.7319초
- 경고/에러: 치명적 에러 없음

### 결과

- train loss: `13.51882094328624`
- eval loss: `3.524868965148926`

- 샘플 조건:
  - artist: `nafla`
  - bpm: `92.0`
  - bpm_class: `slow`
  - density_class: `medium`
  - line_count: `8`
  - avg_chars_per_line: `11.0`

- 샘플 출력:

```text
BPM 92의 slow 비트에 맞춰 nafla 스타일의 랩 가사를 8마디 작성해.
한 줄이 한 마디이며, 줄바꿈해서 8줄 써.
출력은 가사만 작성하고 설명은 쓰지 마.
언어는 한국어와 영어를 자유롭게 사용해.
<가사>: 난 너의 눈을 보고 싶어
너의 손을 꼭 잡아줄게
난 너를 꿈꾸고 있어
내 마음이 널 향해 가고 있네
넌 나를 기다리고 있는 것 같아
그냥 내게 다가오면 돼
I wanna be with you
You're the only one for me
And I wanna hold you tight
Don't let me go
Oh, oh, baby, I'm in love
Baby, don't you see
That you're my one and only
When you look at me like that
It makes my heart skip a beat
Yeah, yeah, we're gonna be together
We're going to have a baby
Just you and me forever
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
