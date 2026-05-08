## EXP-001

- 날짜: 2026-05-08
- 실험 ID: `exp-qwen15b-e1-ga2-len256-v1`
- 목적: Qwen 1.5B LoRA SFT가 현재 데이터셋에서 최소한의 조건부 가사 생성을 수행하는지 확인
- 가설: 1 epoch의 가벼운 LoRA SFT만으로도 base model 대비 가사 스타일 adaptation이 일부 나타난다

### 변경한 것

- 모델: `Qwen/Qwen2.5-1.5B-Instruct`
- 데이터: `kor_verse_train.csv` 228행
- 학습 설정: epoch 1, batch size 1, gradient accumulation 2, max length 256
- 생성 설정: `artist=dynamicduo`, `track_genre=k-rap`, `tempo=118`, `energy=0.4`, `valence=0.1`

### 실행

- 학습 커맨드: `PYTORCH_ENABLE_MPS_FALLBACK=1 python3 -m pipeline.train_sft ...`
- 생성 커맨드: `python3 -m pipeline.generate_sft ...`
- 실행 시간: train 약 129.98초 / eval 약 13.53초
- 경고/에러: 치명적 에러 없음, tokenizer deprecation / MPS pin_memory 경고 / fast tokenizer 성능 경고

### 결과

- train loss: `3.6491`
- eval loss: `3.6036`
- 샘플 조건:
  - artist: `dynamicduo`
  - track_genre: `k-rap`
  - tempo: `118`
  - energy: `0.4`
  - valence: `0.1`

- 샘플 출력:

```text
네가 빨리 나를 찾을 거예요
내가 너를 찾아왔어
너의 손을 잡고 싶어, 내게도
나는 너의 허락을 받아야 해
그러니까 너는 내 마음속에 있어야 돼
네가
```

### 평가

- 잘된 점: 파이프라인이 끝까지 정상 동작했고 adapter가 출력에 변화를 주는 흔적은 있다
- 아쉬운 점: 랩/힙합 톤이 약하고 일반적인 감정 서술로 흐른다
- 조건 반영도: 낮음
- 한국어/영어 혼합 품질: 거의 없음
- 한 줄 결론: 코드와 학습 루프는 성공적으로 검증됐지만 bilingual rap lyric conditioning은 아직 약하다

### 다음 실험

- 다음 가설: prompt에 스타일/형식 지시를 더 넣으면 랩 톤과 bilingual 혼합이 개선된다
- 바꿀 변수: epoch 증가, prompt format 강화, generation sampling 조정
- 유지할 변수: Qwen baseline, LoRA SFT 경로, 과도하게 길지 않은 max length
