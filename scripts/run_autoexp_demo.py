"""Run AutoExp through the shared application service."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoexp.application import AutoExpApplicationService


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the controlled AutoExp experiment loop."
    )
    parser.add_argument("--template", default="text-classification-v1")
    parser.add_argument(
        "--dataset-id",
        default=None,
        help="Registered dataset ID; defaults to the template dataset.",
    )
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--planner", choices=["deterministic", "llm", "auto"], default=None
    )
    parser.add_argument("--executor", choices=["local", "docker"], default=None)
    parser.add_argument("--tracker", choices=["none", "mlflow"], default=None)
    parser.add_argument("--objective", default=None)
    parser.add_argument("--hypothesis", default=None)
    parser.add_argument("--output", type=Path, default=Path("workspaces/autoexp-demo"))
    parser.add_argument("--db", type=Path, default=None, help="SQLite database path.")
    parser.add_argument(
        "--artifact-root", type=Path, default=None, help="Artifact storage directory."
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume an existing run ID."
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    service = AutoExpApplicationService(
        project_root,
        template_id=args.template,
        planner_mode=args.planner,
        database_path=args.db,
        artifact_root=args.artifact_root,
        executor_mode=args.executor,
        tracker_mode=args.tracker,
    )
    objective = (
        args.objective
        or service.manifest.default_objective
        or "Optimize the registered metric."
    )
    hypothesis = (
        args.hypothesis
        or service.manifest.default_hypothesis
        or "Registered parameter changes improve the primary metric."
    )
    if args.resume:
        result = service.resume(args.resume)
    else:
        output_root = (
            args.output if args.output.is_absolute() else project_root / args.output
        )
        result = service.run(
            objective=objective,
            hypothesis=hypothesis,
            max_trials=args.trials,
            output_root=output_root,
            seed=args.seed,
            planner_mode=args.planner,
            dataset_id=args.dataset_id,
        )
    summary = service.summarize(result)
    summary.pop("report_markdown", None)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if result.status != "COMPLETED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
