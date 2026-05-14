import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_storage_root() -> Path:
    drive_root = Path("/content/drive/MyDrive")
    if drive_root.exists():
        return drive_root / PROJECT_ROOT.name
    return PROJECT_ROOT


STORAGE_ROOT = resolve_storage_root()
RUNS_DIR = STORAGE_ROOT / "experiments" / "runs"
REPORTS_DIR = STORAGE_ROOT / "experiments" / "reports"
METRICS_DIR = STORAGE_ROOT / "experiments" / "metrics"
PLOTS_DIR = STORAGE_ROOT / "experiments" / "plots"
INDEX_PATH = STORAGE_ROOT / "experiments" / "index.jsonl"
DEFAULT_OUTPUT_DIR = STORAGE_ROOT / "outputs" / "qwen-1_5b-lyrics-sft"


def ensure_dirs():
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)


def next_experiment_id():
    existing = sorted(RUNS_DIR.glob("EXP-*.json"))
    if not existing:
        return "EXP-001"
    last = existing[-1].stem
    number = int(last.split("-")[1]) + 1
    return f"EXP-{number:03d}"


def parse_args():
    parser = argparse.ArgumentParser(description="Run a train/generate experiment and persist a machine-readable record.")
    parser.add_argument("--experiment-id", help="Optional explicit experiment ID such as EXP-002.")
    parser.add_argument("--goal", required=True, help="Short goal for this run.")
    parser.add_argument("--notes", default="", help="Optional hypothesis or operator note.")
    parser.add_argument("--csv", required=True, help="Training CSV path.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Base output directory for training artifacts.",
    )
    parser.add_argument("--base-model", required=True, help="Base model name/path.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--mps-fallback", action="store_true")
    parser.add_argument("--sample-artist", action="append", dest="sample_artists", default=[])
    parser.add_argument("--sample-track-genre", action="append", dest="sample_genres", default=[])
    parser.add_argument("--sample-tempo", action="append", dest="sample_tempos", type=float, default=[])
    parser.add_argument("--sample-energy", action="append", dest="sample_energies", type=float, default=[])
    parser.add_argument("--sample-valence", action="append", dest="sample_valences", type=float, default=[])
    parser.add_argument("--sample-line-count", type=int, default=8, help="Number of lyric lines/bars to format in generated samples.")
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--skip-generate", action="store_true")
    return parser.parse_args()


def build_train_command(args):
    command = [
        sys.executable,
        "-m",
        "pipeline.train_sft",
        "--experiment-id",
        args.experiment_id_for_train,
        "--csv",
        str(Path(args.csv).resolve()),
        "--output-dir",
        str(Path(args.output_dir).resolve()),
        "--metrics-dir",
        str(METRICS_DIR.resolve()),
        "--plots-dir",
        str(PLOTS_DIR.resolve()),
        "--model-name",
        args.base_model,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--max-length",
        str(args.max_length),
        "--learning-rate",
        str(args.learning_rate),
        "--val-ratio",
        str(args.val_ratio),
        "--seed",
        str(args.seed),
        "--lora-r",
        str(args.lora_r),
        "--lora-alpha",
        str(args.lora_alpha),
        "--lora-dropout",
        str(args.lora_dropout),
    ]
    if args.local_files_only:
        command.append("--local-files-only")
    return command


def shell_join(command):
    return " ".join(shlex.quote(part) for part in command)


def run_command(command, env=None):
    started_at = datetime.now().isoformat()
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        env=env,
    )
    ended_at = datetime.now().isoformat()
    return {
        "command": command,
        "command_str": shell_join(command),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "started_at": started_at,
        "ended_at": ended_at,
    }


def extract_metrics(stdout):
    metrics = {}

    train_loss_match = re.search(r"'train_loss':\s*([0-9.]+)", stdout)
    eval_loss_match = re.search(r"'eval_loss':\s*([0-9.]+)", stdout)
    train_runtime_match = re.search(r"'train_runtime':\s*([0-9.]+)", stdout)
    eval_runtime_match = re.search(r"'eval_runtime':\s*([0-9.]+)", stdout)
    train_sps_match = re.search(r"'train_samples_per_second':\s*([0-9.]+)", stdout)
    train_steps_match = re.search(r"'train_steps_per_second':\s*([0-9.]+)", stdout)

    if train_loss_match:
        metrics["train_loss"] = float(train_loss_match.group(1))
    if eval_loss_match:
        metrics["eval_loss"] = float(eval_loss_match.group(1))
    if train_runtime_match:
        metrics["train_runtime"] = float(train_runtime_match.group(1))
    if eval_runtime_match:
        metrics["eval_runtime"] = float(eval_runtime_match.group(1))
    if train_sps_match:
        metrics["train_samples_per_second"] = float(train_sps_match.group(1))
    if train_steps_match:
        metrics["train_steps_per_second"] = float(train_steps_match.group(1))

    loaded_rows = re.search(r"Loaded (\d+) rows", stdout)
    train_rows = re.search(r"Train rows: (\d+)", stdout)
    validation_rows = re.search(r"Validation rows: (\d+)", stdout)
    save_dir = re.search(r"Saving checkpoints under (.+)", stdout)

    if loaded_rows:
        metrics["dataset_rows"] = int(loaded_rows.group(1))
    if train_rows:
        metrics["train_rows"] = int(train_rows.group(1))
    if validation_rows:
        metrics["validation_rows"] = int(validation_rows.group(1))
    if save_dir:
        metrics["artifact_dir"] = save_dir.group(1).strip()

    return metrics


def default_sample_requests(args):
    artists = args.sample_artists or ["dynamicduo"]
    genres = args.sample_genres or ["k-rap"]
    tempos = args.sample_tempos or [118.0]
    energies = args.sample_energies or [0.4]
    valences = args.sample_valences or [0.1]

    sample_count = max(len(artists), len(genres), len(tempos), len(energies), len(valences))
    requests = []
    for idx in range(sample_count):
        requests.append(
            {
                "artist": artists[idx] if idx < len(artists) else artists[-1],
                "track_genre": genres[idx] if idx < len(genres) else genres[-1],
                "tempo": tempos[idx] if idx < len(tempos) else tempos[-1],
                "energy": energies[idx] if idx < len(energies) else energies[-1],
                "valence": valences[idx] if idx < len(valences) else valences[-1],
            }
        )
    return requests


def build_generate_command(artifact_dir, request, args):
    adapter_dir = Path(artifact_dir) / "final_adapter"
    tokenizer_dir = Path(artifact_dir) / "final_tokenizer"
    command = [
        sys.executable,
        "-m",
        "pipeline.generate_sft",
        "--adapter-dir",
        str(adapter_dir),
        "--tokenizer-dir",
        str(tokenizer_dir),
        "--artist",
        request["artist"],
        "--track-genre",
        request["track_genre"],
        "--tempo",
        str(request["tempo"]),
        "--energy",
        str(request["energy"]),
        "--valence",
        str(request["valence"]),
        "--line-count",
        str(args.sample_line_count),
        "--max-length",
        str(args.max_length),
        "--num-beams",
        str(args.num_beams),
        "--temperature",
        str(args.temperature),
        "--top-k",
        str(args.top_k),
        "--top-p",
        str(args.top_p),
    ]
    if args.local_files_only:
        command.append("--local-files-only")
    return command


def append_index(summary):
    with INDEX_PATH.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(summary, ensure_ascii=False) + "\n")


def build_record_stem(experiment_id, status):
    if status == "failed":
        return f"{experiment_id}-failed"
    return experiment_id


def format_code_block(language, content):
    return f"```{language}\n{content.rstrip()}\n```"


def build_markdown_report(run_record):
    train = run_record["train"]
    config = train["config"]
    metrics = train["metrics"]
    generation_samples = run_record["generation"]["samples"]

    lines = [
        f"# {run_record['experiment_id']}",
        "",
        "## 1. 실험 개요",
        "",
        f"- 날짜: {run_record['date']}",
        f"- 실험 ID: `{run_record['experiment_id']}`",
        f"- 목적: {run_record['goal']}",
        f"- 가설: {run_record.get('hypothesis', '')}",
        f"- 사전 메모: {run_record.get('notes', '')}",
        "",
        "## 2. 실행 설정",
        "",
        f"- 베이스 모델: `{config['base_model']}`",
        f"- 데이터셋: `{config['csv']}`",
        f"- Epochs: {config['epochs']}",
        f"- Batch size: {config['batch_size']}",
        f"- Gradient accumulation: {config['gradient_accumulation_steps']}",
        f"- Max length: {config['max_length']}",
        f"- Learning rate: {config['learning_rate']}",
        f"- Seed: {config['seed']}",
        "",
        "### 학습 커맨드",
        "",
        format_code_block("bash", train["command_str"]),
        "",
        "## 3. 학습 결과 원본",
        "",
        f"- 상태: `{run_record['status']}`",
        f"- Train loss: {metrics.get('train_loss', '')}",
        f"- Eval loss: {metrics.get('eval_loss', '')}",
        f"- Train runtime: {metrics.get('train_runtime', '')}",
        f"- Eval runtime: {metrics.get('eval_runtime', '')}",
        f"- Artifact dir: `{metrics.get('artifact_dir', '')}`",
        "",
        "### 핵심 로그",
        "",
        format_code_block("text", train["stdout"][-4000:] if train["stdout"] else ""),
        "",
        "## 4. 생성 결과",
        "",
    ]

    if generation_samples:
        for index, sample in enumerate(generation_samples, start=1):
            conditions = sample["conditions"]
            lines.extend(
                [
                    f"### Sample {index}",
                    "",
                    f"- artist: `{conditions['artist']}`",
                    f"- track_genre: `{conditions['track_genre']}`",
                    f"- tempo: `{conditions['tempo']}`",
                    f"- energy: `{conditions['energy']}`",
                    f"- valence: `{conditions['valence']}`",
                    "",
                    "#### 생성 커맨드",
                    "",
                    format_code_block("bash", sample["command_str"]),
                    "",
                    "#### 출력",
                    "",
                    format_code_block("text", sample["text"]),
                    "",
                ]
            )
    else:
        lines.extend(["- 생성 샘플 없음", ""])

    lines.extend(
        [
            "## 5. 평가",
            "",
            "- 평가자:",
            "- 총평:",
            "- 결과 판정:",
            "",
            "### 점수",
            "",
            "- 문장 자연스러움:",
            "- 가사다움:",
            "- 랩 느낌:",
            "- 한국어 품질:",
            "- 영어 품질:",
            "- 한국어/영어 혼합 자연스러움:",
            "- 조건 반영도:",
            "- 독창성:",
            "",
            "### 관찰",
            "",
            "- 잘된 점:",
            "- 이상한 점:",
            "- 반복 패턴:",
            "- 의심되는 원인:",
            "",
            "## 6. 다음 가설",
            "",
            "- 다음 실험 가설:",
            "- 다음에 바꿀 변수:",
            "- 그대로 둘 변수:",
            "- 다음 실험 ID:",
            "",
        ]
    )
    return "\n".join(lines)


def build_notion_summary(run_record):
    train = run_record["train"]
    config = train["config"]
    metrics = train["metrics"]
    generation_samples = run_record["generation"]["samples"]

    lines = [
        f"## {run_record['experiment_id']}",
        "",
        f"- 날짜: {run_record['date']}",
        f"- 실험 ID: `{run_record['experiment_id']}`",
        f"- 목적: {run_record['goal']}",
        f"- 가설: {run_record.get('hypothesis', '')}",
        "",
        "### 변경한 것",
        "",
        f"- 모델: `{config['base_model']}`",
        f"- 데이터: `{Path(config['csv']).name}` {metrics.get('dataset_rows', '')}행",
        f"- 학습 설정: epoch {config['epochs']}, batch size {config['batch_size']}, gradient accumulation {config['gradient_accumulation_steps']}, max length {config['max_length']}",
        f"- 생성 설정: max length {config['max_length']}",
        f"- 생성 라인 수: {config.get('sample_line_count', '')}",
        "",
        "### 실행",
        "",
        f"- 학습 커맨드: `{train['command_str']}`",
        f"- 실행 시간: train 약 {metrics.get('train_runtime', '')}초 / eval 약 {metrics.get('eval_runtime', '')}초",
        f"- 경고/에러: {'치명적 에러 없음' if train['returncode'] == 0 else '실패 로그 확인 필요'}",
        "",
        "### 결과",
        "",
        f"- train loss: `{metrics.get('train_loss', '')}`",
        f"- eval loss: `{metrics.get('eval_loss', '')}`",
        "",
    ]

    if generation_samples:
        first_sample = generation_samples[0]
        conditions = first_sample["conditions"]
        lines.extend(
            [
                "- 샘플 조건:",
                f"  - artist: `{conditions['artist']}`",
                f"  - track_genre: `{conditions['track_genre']}`",
                f"  - tempo: `{conditions['tempo']}`",
                f"  - energy: `{conditions['energy']}`",
                f"  - valence: `{conditions['valence']}`",
                "",
                "- 샘플 출력:",
                "",
                format_code_block("text", first_sample["text"]),
                "",
            ]
        )
    else:
        lines.extend(["- 샘플 출력: 없음", ""])

    lines.extend(
        [
            "### 평가",
            "",
            "- 잘된 점:",
            "- 아쉬운 점:",
            "- 조건 반영도:",
            "- 한국어/영어 혼합 품질:",
            "- 한 줄 결론:",
            "",
            "### 다음 실험",
            "",
            "- 다음 가설:",
            "- 바꿀 변수:",
            "- 유지할 변수:",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    ensure_dirs()
    args = parse_args()
    experiment_id = args.experiment_id or next_experiment_id()
    args.experiment_id_for_train = experiment_id

    env = os.environ.copy()
    if args.mps_fallback:
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    train_command = build_train_command(args)
    train_result = run_command(train_command, env=env)
    train_metrics = extract_metrics(train_result["stdout"])

    run_record = {
        "experiment_id": experiment_id,
        "date": str(date.today()),
        "created_at": datetime.now().isoformat(),
        "status": "completed" if train_result["returncode"] == 0 else "failed",
        "goal": args.goal,
        "hypothesis": "",
        "notes": args.notes,
        "train": {
            "command": train_result["command"],
            "command_str": train_result["command_str"],
            "returncode": train_result["returncode"],
            "stdout": train_result["stdout"],
            "stderr": train_result["stderr"],
            "config": {
                "csv": str(Path(args.csv).resolve()),
                "output_dir": str(Path(args.output_dir).resolve()),
                "base_model": args.base_model,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "max_length": args.max_length,
                "learning_rate": args.learning_rate,
                "val_ratio": args.val_ratio,
                "seed": args.seed,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "sample_line_count": args.sample_line_count,
                "local_files_only": args.local_files_only,
                "mps_fallback": args.mps_fallback,
            },
            "metrics": train_metrics,
        },
        "generation": {
            "samples": [],
        },
        "review": {
            "reviewer": "",
            "summary": "",
            "result": "",
            "scores": {
                "fluency": None,
                "lyricness": None,
                "rap_feel": None,
                "korean_quality": None,
                "english_quality": None,
                "bilingual_mixing": None,
                "conditioning_fidelity": None,
                "originality": None,
            },
            "observations": {
                "strengths": [],
                "oddities": [],
                "repeated_patterns": [],
                "suspected_causes": [],
            },
            "next_hypothesis": "",
            "next_changes": [],
            "keep_fixed": [],
            "next_experiment_id": "",
        },
        "evaluation": {},
    }

    if train_result["returncode"] == 0 and not args.skip_generate and "artifact_dir" in train_metrics:
        artifact_dir = train_metrics["artifact_dir"]
        for request in default_sample_requests(args):
            generate_command = build_generate_command(artifact_dir, request, args)
            generate_result = run_command(generate_command, env=env)
            run_record["generation"]["samples"].append(
                {
                    "conditions": request,
                    "command": generate_result["command"],
                    "command_str": generate_result["command_str"],
                    "returncode": generate_result["returncode"],
                    "stdout": generate_result["stdout"],
                    "stderr": generate_result["stderr"],
                    "text": generate_result["stdout"].strip(),
                }
            )

    record_stem = build_record_stem(experiment_id, run_record["status"])

    run_path = RUNS_DIR / f"{record_stem}.json"
    with run_path.open("w", encoding="utf-8") as fp:
        json.dump(run_record, fp, ensure_ascii=False, indent=2)

    report_path = REPORTS_DIR / f"{record_stem}.md"
    with report_path.open("w", encoding="utf-8") as fp:
        fp.write(build_markdown_report(run_record))

    notion_report_path = REPORTS_DIR / f"{record_stem}-notion.md"
    with notion_report_path.open("w", encoding="utf-8") as fp:
        fp.write(build_notion_summary(run_record))

    append_index(
        {
            "experiment_id": experiment_id,
            "date": run_record["date"],
            "status": run_record["status"],
            "goal": args.goal,
            "base_model": args.base_model,
            "train_loss": train_metrics.get("train_loss"),
            "eval_loss": train_metrics.get("eval_loss"),
            "artifact_dir": train_metrics.get("artifact_dir"),
            "run_path": str(run_path),
            "report_path": str(report_path),
            "notion_report_path": str(notion_report_path),
            "train_metrics_csv": str((METRICS_DIR / f"{experiment_id}-train.csv").resolve()),
            "eval_metrics_csv": str((METRICS_DIR / f"{experiment_id}-eval.csv").resolve()),
            "loss_plot_path": str((PLOTS_DIR / f"{experiment_id}-loss.png").resolve()),
        }
    )

    print(f"Saved run record to {run_path}")
    print(f"Saved markdown report to {report_path}")
    print(f"Saved Notion summary to {notion_report_path}")
    print(f"Saved train metrics CSV to {METRICS_DIR / f'{experiment_id}-train.csv'}")
    print(f"Saved eval metrics CSV to {METRICS_DIR / f'{experiment_id}-eval.csv'}")
    print(f"Saved loss plot to {PLOTS_DIR / f'{experiment_id}-loss.png'}")
    if run_record["generation"]["samples"]:
        print(f"Saved {len(run_record['generation']['samples'])} generation sample(s)")


if __name__ == "__main__":
    main()
