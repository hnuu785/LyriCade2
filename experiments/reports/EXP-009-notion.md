## EXP-009

- 날짜: 2026-05-17
- 실험 ID: `EXP-009`
- 목적: Korean prompt rebuild with nafla boom bap structure cues
- 가설: 

### 변경한 것

- 모델: `Qwen/Qwen2.5-1.5B-Instruct`
- 데이터: `kor_verse_datasets_slim.csv` 1134행
- 학습 설정: epoch 4, batch size 2, gradient accumulation 4, max length 512
- 라이밍 설정: rhyme loss weight 0.1, max rhyme positions 16
- 생성 설정: max length 512

### 실행

- 학습 커맨드: `/usr/bin/python3 -m pipeline.train_sft --experiment-id EXP-009 --csv /content/LyriCade2/datasets/kor_verse_datasets_slim.csv --output-dir /content/drive/MyDrive/LyriCade2/outputs/qwen-1_5b-lyrics-sft --metrics-dir /content/drive/MyDrive/LyriCade2/experiments/metrics --plots-dir /content/drive/MyDrive/LyriCade2/experiments/plots --model-name Qwen/Qwen2.5-1.5B-Instruct --epochs 4 --batch-size 2 --gradient-accumulation-steps 4 --max-length 512 --learning-rate 0.0002 --val-ratio 0.2 --seed 42 --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 --rhyme-loss-weight 0.1 --max-rhyme-positions 16`
- 실행 시간: train 약 3765.1155초 / eval 약 97.5609초
- 경고/에러: 치명적 에러 없음

### 결과

- train loss: `13.587516701012326`
- eval loss: `3.4673638343811035`

- 샘플 조건:
  - artist: `nafla`
  - bpm: `92.0`
  - bpm_class: `slow`
  - density_class: `medium`
  - line_count: `8`
  - avg_chars_per_line: `11.0`

- 샘플 출력:

```text
<아티스트: nafla>
<BPM: 92>
<비트 스타일: slow>
<줄 수: 8줄>
<한 줄은 한 마디>
<한 마디 글자 수: medium>
<가사 스타일: 한국어 중심 랩 가사, 영어는 짧게만 사용>
<가사>: 네가 내게 말했을 때
나는 그걸 믿지 못했어
그때 너의 눈빛이
아직도 나를 떠올리고 있어
너의 손을 잡고 싶은데
우리 사이에 끼는 거리가 너무 멀어서
난 너를 잊을 수가 없었지
내 마음속엔 너만 남아있었고
그래서 넌 나의 곁에 있어야만 했었는데
네가 나에게 와서 말하길
넌 나와 함께할 수 있겠어?
그 말을 듣고 난
눈물을 흘리며 널 품에 안았었지만
이젠 너와 나란히 살기엔 너무 먼 거리를 두고 살게 될 것 같아서
사랑이란 걸 다시 생각해봐야 할 것 같은 느낌이 들어
어쩌면 이젠 사랑이 아니었을지도 모른다는 생각이 든다
하지만 너는 내가 원했던 걸 해주길 바랬던 것처럼 말해줘
왜냐면 나도 그랬기 때문이었기 때문에
오랜 시간을 헤쳐나가면서도
마음속으로는 너가 완벽한 사람이라고 생각했었거든
누군가를 사랑하려면 많은 것들이 필요해
돈, 명예, 권력, 그리고 사랑
근데 그게 왜 필요하냐고 물어보면
대답할 만한 대답이 없을 거야
시간이 지나고 나면 모든 걸 잃어버릴 거라고
말해준 적이 많았던 그녀의 말이 맞았을까?
나도 모르겠지, 그저 사랑하는 사람을 보는 것만으로도 충분하다고 느껴졌던 너였던 걸로 보여도 돼
모든 걸 뺏겨버려도 되는 건
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
