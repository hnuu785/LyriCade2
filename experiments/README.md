# Experiments

이 디렉터리는 기계가 읽을 수 있는 실험 기록을 저장한다.

## Layout

```text
experiments/
  README.md
  runs/
    EXP-001.json
    EXP-002.json
  reports/
    EXP-001.md
    EXP-002.md
    EXP-001-notion.md
  metrics/
    EXP-001-train.csv
    EXP-001-eval.csv
  plots/
    EXP-001-loss.png
  index.jsonl
```

## Rules

- `runs/EXP-XXX.json`: 한 번의 실험 실행 결과 전체
- `reports/EXP-XXX.md`: 사람이 읽고 LLM에 바로 넣기 쉬운 Markdown 초안
- `reports/EXP-XXX-notion.md`: 노션에 바로 붙여넣기 쉬운 짧은 요약 초안
- `metrics/EXP-XXX-train.csv`: 학습 과정 step별 로그
- `metrics/EXP-XXX-eval.csv`: 평가 시점 로그
- `plots/EXP-XXX-loss.png`: train/eval loss 그래프
- `index.jsonl`: 각 실행의 요약 레코드 한 줄씩 누적
- 사람이 읽는 해설은 `docs/experiment-log.md` 에 남기고, 자동화는 이 디렉터리를 기준으로 읽는다

## Minimal Run Schema

```json
{
  "experiment_id": "EXP-001",
  "date": "2026-05-08",
  "status": "completed",
  "goal": "Qwen 1.5B baseline smoke test",
  "train": {
    "command": ["python3", "-m", "pipeline.train_sft", "..."],
    "config": {
      "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
      "epochs": 1,
      "batch_size": 1,
      "gradient_accumulation_steps": 2,
      "max_length": 256
    },
    "metrics": {
      "train_loss": 3.6491,
      "eval_loss": 3.6036
    }
  },
  "generation": {
    "samples": [
      {
        "conditions": {
          "artist": "dynamicduo",
          "track_genre": "k-rap"
        },
        "text": "..."
      }
    ]
  },
  "review": {
    "reviewer": "",
    "summary": "",
    "result": "",
    "scores": {
      "conditioning_fidelity": null,
      "bilingual_mixing": null
    },
    "next_hypothesis": ""
  }
}
```

## Suggested Workflow

1. `scripts/run_experiment.py` 로 학습/생성 실행
2. `runs/EXP-XXX.json`, `reports/EXP-XXX.md`, `reports/EXP-XXX-notion.md` 생성 확인
3. 사람 또는 외부 LLM이 `reports/EXP-XXX.md` 또는 `reports/EXP-XXX-notion.md` 를 읽고 평가
4. 평가 결과를 JSON의 `review` 필드와 사람이 읽는 로그에 반영
5. 필요하면 `scripts/evaluate_experiment.py` 를 참고용 보조 점수 계산에만 사용
