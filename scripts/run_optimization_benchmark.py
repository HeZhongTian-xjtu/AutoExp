from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoexp.application import AutoExpApplicationService
from autoexp.benchmark import OptimizationBenchmarkCatalog, OptimizationBenchmarkRunner


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare AutoExp optimization policies fairly."
    )
    parser.add_argument("--template", default=None)
    parser.add_argument(
        "--all-tasks", action="store_true", help="Run every Phase 1 registered task."
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List the fixed Phase 1 tasks and exit.",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=["random", "optuna", "llm"],
        default=["random", "optuna", "llm"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument(
        "--allow-code-optimization",
        action="store_true",
        help="Allow LLM Repair; disabled by default for fair parameter comparisons.",
    )
    parser.add_argument("--output", type=Path, default=Path("workspaces/benchmarks"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    catalog = OptimizationBenchmarkCatalog(root)
    tasks = catalog.list()

    if args.list_tasks:
        for task in tasks:
            print(
                f"{task.template_id}	{task.metric_name} ({task.metric_direction})	{task.description}"
            )
        return

    if args.template and args.all_tasks:
        parser.error("--template and --all-tasks cannot be used together")

    selected = (
        tasks
        if args.all_tasks
        else [catalog.get(args.template or tasks[0].template_id)]
    )

    for task in selected:

        def factory(benchmark_root=None, **kwargs):
            scope = Path(benchmark_root or args.output / "state")
            return AutoExpApplicationService(
                root,
                summary_mode="disabled",
                tracker_mode="none",
                database_path=scope / "autoexp.sqlite3",
                artifact_root=scope / "artifacts",
                checkpoint_path=scope / "checkpoints.sqlite3",
                **kwargs,
            )

        report = OptimizationBenchmarkRunner(factory).run(
            task.template_id,
            "Optimize the registered metric.",
            "Result-conditioned policies may reach a better value with the same budget.",
            args.trials,
            args.seeds,
            args.policies,
            args.output / task.template_id if args.all_tasks else args.output,
            allow_code_optimization=args.allow_code_optimization,
        )
        print(report.to_markdown())


if __name__ == "__main__":
    main()
