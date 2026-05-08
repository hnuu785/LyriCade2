# Experiment Log

이 문서는 실제로 수행한 실험을 순서대로 기록한다.  
공통 배경, 목표, 평가 기준은 [experiment-overview.md](/Users/cho/Developer/PMTMProject/LyricGeneration/docs/experiment-overview.md) 를 따른다.

---

## EXP-001

### 1. 실험 개요

- 날짜: 2026-05-08
- 실험 ID: `exp-qwen15b-e1-ga2-len256-v1`
- 목적: Qwen 1.5B LoRA SFT가 현재 데이터셋에서 최소한의 조건부 가사 생성을 수행하는지 확인
- 한 줄 가설: 1 epoch의 가벼운 LoRA SFT만으로도 base model 대비 가사 스타일 adaptation이 일부 나타난다

### 2. 이번 실험에서 바꾼 것

- 베이스 모델: `Qwen/Qwen2.5-1.5B-Instruct`
- 학습 규모: 1 epoch
- 실행 조건: Apple Silicon MPS에서 동작 가능한 최소 설정
- 길이 제한: `max_length=256`

핵심 변경 요약:

- 7B 대신 1.5B로 축소
- gradient accumulation을 2로 설정
- 스모크 테스트 목적의 최소 학습 설정 사용

### 3. 실행 정보

- 실행 커맨드:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 -m pipeline.train_sft \
  --csv /Users/cho/Developer/PMTMProject/LyricGeneration/datasets/kor_verse_train.csv \
  --output-dir /Users/cho/Developer/PMTMProject/LyricGeneration/outputs/qwen-1_5b-lyrics-sft \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --epochs 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 2 \
  --max-length 256
```

- 실행 환경:
  - Machine: MacBook Pro
  - Chip: Apple M5
  - RAM: 16 GB unified memory
  - Device path: MPS
- 학습 시간: 약 129.98초
- 평가 시간: 약 13.53초
- 중단 여부: 없음

### 4. 데이터

- Input CSV: `/Users/cho/Developer/PMTMProject/LyricGeneration/datasets/kor_verse_train.csv`
- Dataset size: 228
- Train rows: 182
- Validation rows: 46
- 전처리: verse 중심 CSV + feature-conditioned prompt

### 5. 정량 결과

- Final train loss: `3.6491`
- Final eval loss: `3.6036`
- Train samples/sec: `1.4`
- Train steps/sec: `0.7`

핵심 로그:

```text
{'loss': 3.6344, 'grad_norm': 1.5735, 'learning_rate': 0.0001802, 'epoch': 0.11}
{'loss': 3.7711, 'grad_norm': 1.6920, 'learning_rate': 0.0001582, 'epoch': 0.22}
{'loss': 3.4907, 'grad_norm': 1.5070, 'learning_rate': 0.0001363, 'epoch': 0.33}
{'loss': 3.5617, 'grad_norm': 1.7435, 'learning_rate': 0.0001143, 'epoch': 0.44}
{'loss': 3.6850, 'grad_norm': 2.1904, 'learning_rate': 0.0000923, 'epoch': 0.55}
{'loss': 3.6415, 'grad_norm': 1.7812, 'learning_rate': 0.0000703, 'epoch': 0.66}
{'loss': 3.6132, 'grad_norm': 1.3963, 'learning_rate': 0.0000484, 'epoch': 0.77}
{'loss': 3.6125, 'grad_norm': 1.7044, 'learning_rate': 0.0000264, 'epoch': 0.88}
{'loss': 3.8195, 'grad_norm': 1.7299, 'learning_rate': 0.0000044, 'epoch': 0.99}
{'eval_loss': 3.603569746017456, 'eval_runtime': 13.5311, 'eval_samples_per_second': 3.4, 'eval_steps_per_second': 3.4, 'epoch': 1.0}
{'train_runtime': 129.982, 'train_samples_per_second': 1.4, 'train_steps_per_second': 0.7, 'train_loss': 3.6491061566950203, 'epoch': 1.0}
```

정량 해석:

- 학습/평가/저장이 정상적으로 끝났으므로 파이프라인은 성공
- loss가 터지지 않았고 grad norm도 비정상적이지 않음
- 다만 1 epoch, 228개 데이터 기준이라 품질 평가는 생성 샘플 중심으로 봐야 함

### 6. 런타임 메모

경고:

- `Trainer.__init__ tokenizer is deprecated`
- `pin_memory not supported on MPS`
- `Qwen2TokenizerFast ... __call__ is faster`

에러:

- 없음

해석:

- 모두 치명적 에러는 아니었음
- Qwen 3B는 초기 시도에서 지나치게 무거웠고, 1.5B는 현재 환경에서 안정적으로 완료됨

### 7. 생성 결과

- 생성 커맨드:

```bash
python3 -m pipeline.generate_sft \
  --adapter-dir /Users/cho/Developer/PMTMProject/LyricGeneration/outputs/qwen-1_5b-lyrics-sft/2026-05-08/final_adapter \
  --tokenizer-dir /Users/cho/Developer/PMTMProject/LyricGeneration/outputs/qwen-1_5b-lyrics-sft/2026-05-08/final_tokenizer \
  --artist dynamicduo \
  --track-genre k-rap \
  --tempo 118 \
  --energy 0.4 \
  --valence 0.1
```

- 입력 조건:
  - artist: `dynamicduo`
  - genre: `k-rap`
  - tempo: `118`
  - energy: `0.4`
  - valence: `0.1`

- 출력 샘플:

```text
네가 빨리 나를 찾을 거예요
내가 너를 찾아왔어
너의 손을 잡고 싶어, 내게도
나는 너의 허락을 받아야 해
그러니까 너는 내 마음속에 있어야 돼
네가
```

### 8. 정성 평가

- 문장 자연스러움: 6/10
- 가사다움: 4/10
- 랩 느낌: 3/10
- 한국어 품질: 6/10
- 영어 품질: 1/10
- 한국어/영어 혼합 자연스러움: 1/10
- 조건 반영도: 2/10
- 독창성: 4/10

짧은 총평:

- 문장 자체는 생성되지만, 랩/힙합 가사보다는 일반적인 감정 서술 쪽에 가깝다
- bilingual lyric generation 목표에는 아직 한참 부족하다
- base model adaptation은 보이지만 conditioning fidelity는 매우 약하다

### 9. 관찰

잘된 점:

- 파이프라인이 끝까지 정상 동작했다
- adapter가 base model 출력에 최소한의 변화를 주는 흔적은 있다

이상한 점:

- `dynamicduo`, `k-rap`, `tempo 118`, `energy 0.4`, `valence 0.1` 조건이 거의 드러나지 않는다
- 출력이 사랑 노랫말/평서문 쪽으로 흐른다

반복 패턴:

- 직접적 감정 표현
- 짧고 일반적인 연애 문장

의심되는 원인:

- 데이터 규모 부족
- 1 epoch로는 adaptation이 약함
- prompt schema가 태스크 지시를 충분히 주지 못함
- instruct base model의 일반 응답 성향이 강함

### 10. 결론

- 결과 판정: `promising`
- 한 줄 결론: 코드와 파이프라인은 성공적으로 검증됐지만, 현재 설정만으로는 bilingual rap lyric conditioning이 충분히 학습되지 않았다

유지할 것:

- Qwen 계열 baseline
- LoRA SFT 경로
- Apple Silicon MPS 실행 방식

버릴 것:

- 품질 판단 없이 1 epoch 결과만으로 방향을 확정하는 것
- 현재 약한 prompt schema를 그대로 장기 사용

### 11. 다음 실험

- 다음 실험 가설: prompt에 스타일/형식 지시를 더 넣으면 랩 톤과 bilingual 혼합이 개선된다
- 다음에 바꿀 변수:
  - epoch 증가 (`2~3`)
  - prompt format 강화
  - generation sampling 조정
- 그대로 둘 변수:
  - Qwen baseline
  - LoRA SFT 경로
  - max_length를 지나치게 늘리지 않는 실행 전략

다음 후보 실험 ID:

- `exp-qwen15b-e3-promptv2`
