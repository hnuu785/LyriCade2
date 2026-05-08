# Experiment Overview

이 문서는 이 저장소에서 수행하는 LLM 가사 생성 실험의 공통 원칙을 정리한다.  
목적은 매 실험마다 같은 배경 설명을 반복하지 않고, 무엇을 왜 검증하는지 기준을 고정하는 데 있다.

## 1. 실험 목표

현재 프로젝트의 목표는 아래와 같다.

- 한국어와 영어가 섞인 가사를 생성할 수 있는 모델 만들기
- 단순 문장 생성이 아니라 가사다운 리듬감과 장르 톤을 반영하기
- `artist`, `genre`, `tempo`, `energy`, `valence` 같은 조건이 실제 출력에 반영되는지 확인하기
- 로컬 환경에서 반복 가능한 fine-tuning 실험 루프를 만드는 것

핵심은 "한국어 전용 텍스트 생성"이 아니라 "한국어/영어 code-switching lyric generation" 이다.

## 2. 현재 기본 방향

- 기본 학습 경로: `pipeline.train_sft`
- 기본 생성 경로: `pipeline.generate_sft`
- 기본 실험 계열: LoRA SFT

현재 기본 베이스라인 모델 방향은 Qwen 계열이다.

## 3. 모델 선택 원칙

현재 모델 선택 기준은 아래와 같다.

1. 한국어와 영어를 모두 안정적으로 다룰 수 있어야 한다
2. 로컬 LoRA 파인튜닝이 가능해야 한다
3. 반복 실험과 추론 검증이 쉬워야 한다
4. 필요하면 추후 커스텀 objective를 붙일 수 있어야 한다

### 현재 판단

- 1차 베이스라인: `Qwen/Qwen2.5-*`
- 2차 비교군: Gemma 계열
- GPT / Gemini: 현재 저장소의 기본 경로로는 두지 않음

이유는 GPT/Gemini는 호스티드 튜닝에는 강하지만, 현재처럼 로컬 루프 기반의 trial-and-error 실험과는 맞지 않기 때문이다.

## 4. 왜 Qwen인가

현재 프로젝트에서 Qwen을 우선하는 이유는 아래와 같다.

- 한국어/영어 혼합 텍스트 처리에 무난하다
- 로컬 추론/LoRA 실험 자료가 많다
- Hugging Face 기반 실험과 잘 맞는다
- 이후 1.5B / 3B / 7B 식으로 실험 확장이 쉽다

현재는 7B를 최종 기본 후보로 두되, 실제 장비 제약 때문에 1.5B 또는 3B로 먼저 검증할 수 있다.

## 5. 데이터 가정

현재 데이터셋은 verse 중심의 가사 CSV를 사용한다.

입력 컬럼의 핵심은 아래와 같다.

- `title`
- `artist`
- `lyrics`
- `bpm`
- `energy`
- `danceability`
- `loudness`
- `valence`

현재 파이프라인은 이 정보를 feature-conditioned prompt로 직렬화해서 학습한다.

## 6. 실험에서 먼저 검증할 것

실험은 아래 순서로 검증한다.

1. 파이프라인이 정상적으로 돌아가는가
2. 생성 결과가 최소한 가사처럼 보이는가
3. 조건(`artist`, `genre`, `tempo` 등)이 출력에 반영되는가
4. 한국어/영어 혼합이 자연스럽게 나오는가
5. 랩/힙합 톤이 살아나는가

즉, 단순히 loss가 내려가는 것만으로 성공이라고 판단하지 않는다.

## 7. 평가 기준

실험 결과는 아래 기준으로 본다.

### 정량

- train loss
- eval loss
- train runtime
- eval runtime

### 정성

- 문장 자연스러움
- 가사다움
- 랩 느낌
- 한국어 품질
- 영어 품질
- 한국어/영어 혼합 자연스러움
- conditioning fidelity
- 반복/클리셰 정도

## 8. 해석 원칙

아래 해석은 금지한다.

- loss만 보고 "품질이 좋다"고 판단
- 샘플 하나만 보고 "모델이 됐다/안 됐다" 결론
- 여러 변수를 한 번에 바꾸고 원인을 단정

실험은 가능하면 한 번에 1~2개 변수만 바꾼다.

## 9. 현재까지 코드에서 확보한 것

현재 코드 기준으로 이미 확보한 사항:

- padding token loss 마스킹
- train/inference 공통 feature transform 저장 및 재사용
- prompt/completion 분리 라벨링
- LoRA adapter 전용 생성 스크립트 추가
- GPT-2 하드코딩 제거

## 10. 현재까지 남은 주요 과제

- Qwen 기준 prompt schema 개선
- bilingual lyric style 유도 강화
- conditioning fidelity 개선
- rhyme objective 를 안정적인 baseline 위에 다시 얹기
- 모델 크기(1.5B / 3B / 7B)별 품질 대비 비용 비교

## 11. 실험 기록 원칙

실제 실험 결과는 `docs/experiment-log.md` 에 누적한다.

실험 로그에는 아래만 분명히 남긴다.

- 이번에 무엇을 바꿨는지
- 왜 그렇게 바꿨는지
- 결과가 어땠는지
- 다음 실험에서 무엇을 바꿀지
