from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autoexp.benchmark import RepairBenchmarkCatalog, RepairBenchmarkRunner
from autoexp.domain import RepairSpec
from autoexp.execution import build_executor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare or verify a controlled AutoExp repair benchmark."
    )
    parser.add_argument("case_id")
    parser.add_argument(
        "--repair", type=Path, help="JSON RepairSpec produced by an LLM"
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Call the configured external LLM and verify its patch",
    )
    parser.add_argument(
        "--allow-source-egress",
        action="store_true",
        help="Explicitly authorize sending the controlled fixture source and logs to the configured LLM.",
    )
    parser.add_argument("--executor", choices=["local", "docker"], default="local")
    parser.add_argument(
        "--output", type=Path, default=Path("workspaces/repair-benchmarks")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    case = RepairBenchmarkCatalog(root).get(args.case_id)
    runner = RepairBenchmarkRunner(root, build_executor(args.executor))
    if args.llm and not args.allow_source_egress:
        parser.error(
            "--llm requires --allow-source-egress because the fixture source and failure logs leave the machine."
        )
    if args.llm:
        print(json.dumps(runner.run_llm(case, args.output), indent=2))
        return
    workspace = runner.prepare(case, args.output)
    if args.repair:
        repair = RepairSpec.model_validate_json(args.repair.read_text(encoding="utf-8"))
        print(json.dumps(runner.validate_repair(case, workspace, repair), indent=2))
    else:
        print(
            json.dumps(
                {
                    "case": case.case_id,
                    "workspace": str(workspace),
                    "next": "Provide --repair RepairSpec.json to validate a repair.",
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
