from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from autoexp.domain import RepairSpec, TemplateManifest
from autoexp.llm import FunctionSpec
from autoexp.planning.compat import _bounded_openai_query
from autoexp.repair import apply_unified_patch, validate_repaired_template
from autoexp.validation import ValidationPipeline


@dataclass(frozen=True)
class RepairBenchmarkCase:
    case_id: str
    template_id: str
    description: str
    expected_failure: str
    fault: RepairSpec


class RepairBenchmarkCatalog:
    def __init__(self, project_root: Path | str):
        self.root = Path(project_root).resolve() / "repair_benchmarks"

    def list(self) -> list[RepairBenchmarkCase]:
        cases = []
        if not self.root.is_dir():
            return cases
        for path in sorted(self.root.glob("*/case.yaml")):
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            patch_path = path.parent / data["fault_patch"]
            cases.append(
                RepairBenchmarkCase(
                    case_id=data["case_id"],
                    template_id=data["template_id"],
                    description=data["description"],
                    expected_failure=data["expected_failure"],
                    fault=RepairSpec(
                        target_file=data["target_file"],
                        patch=patch_path.read_text(encoding="utf-8"),
                        reason=f"Inject benchmark fault {data['case_id']}",
                    ),
                )
            )
        return cases

    def get(self, case_id: str) -> RepairBenchmarkCase:
        for case in self.list():
            if case.case_id == case_id:
                return case
        raise KeyError(f"repair benchmark not found: {case_id}")


class RepairBenchmarkRunner:
    """Inject a fault into an isolated template and validate a proposed repair."""

    def __init__(self, project_root: Path | str, executor: Any):
        self.project_root, self.executor = Path(project_root).resolve(), executor

    def prepare(self, case: RepairBenchmarkCase, output_root: Path) -> Path:
        from autoexp.application import TemplateCatalog

        descriptor = TemplateCatalog(self.project_root).get(case.template_id)
        workspace = output_root.resolve() / case.case_id
        if workspace.exists():
            shutil.rmtree(workspace)
        shutil.copytree(descriptor.root, workspace)
        runtime_config = workspace / "configs" / "experiment.yaml"
        if not runtime_config.exists():
            runtime_config.write_text(
                yaml.safe_dump(
                    {**descriptor.manifest.default_parameters, "seed": 42},
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        apply_unified_patch(workspace, case.fault, descriptor.manifest)
        evidence = {
            "case_id": case.case_id,
            "description": case.description,
            "expected_failure": case.expected_failure,
            "fault_patch_sha256": hashlib.sha256(case.fault.patch.encode()).hexdigest(),
        }
        (workspace / "benchmark_evidence.json").write_text(
            json.dumps(evidence, indent=2), encoding="utf-8"
        )
        return workspace

    def validate_repair(
        self, case: RepairBenchmarkCase, workspace: Path, repair: RepairSpec
    ) -> dict[str, Any]:
        manifest = TemplateManifest.load(workspace / "manifest.yaml")
        applied = apply_unified_patch(workspace, repair, manifest)
        preflight = validate_repaired_template(
            workspace, manifest, self.project_root / "requirements.txt"
        )
        validation = ValidationPipeline(self.executor, manifest).run(
            workspace,
            uuid4(),
            uuid4(),
            int(manifest.resource_policy.get("timeout_seconds", 120)),
        )
        result = {
            "case_id": case.case_id,
            "accepted": bool(preflight.passed and validation.passed),
            "repair": repair.model_dump(mode="json"),
            "patch_result": applied.result.model_dump(mode="json"),
            "preflight": preflight.model_dump(mode="json"),
            "validation": validation.model_dump(mode="json"),
        }
        (workspace / "repair_result.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        return result

    def run_llm(
        self,
        case: RepairBenchmarkCase,
        output_root: Path,
        query_fn: Any = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Execute the fault, ask an external LLM for RepairSpec, then verify it."""
        workspace = self.prepare(case, output_root)
        manifest = TemplateManifest.load(workspace / "manifest.yaml")
        failed = ValidationPipeline(self.executor, manifest).run(
            workspace,
            uuid4(),
            uuid4(),
            int(manifest.resource_policy.get("timeout_seconds", 120)),
        )
        logs = []
        for gate in failed.gates:
            for path in (gate.stdout_path, gate.stderr_path):
                if path and Path(path).is_file():
                    logs.append(
                        Path(path).read_text(encoding="utf-8", errors="replace")[-3000:]
                    )
        target = workspace / case.fault.target_file
        prompt = json.dumps(
            {
                "benchmark": {
                    "id": case.case_id,
                    "description": case.description,
                    "expected_failure": case.expected_failure,
                },
                "failure_logs": "\n".join(logs)[-6000:],
                "target_file": case.fault.target_file,
                "target_source": target.read_text(encoding="utf-8"),
                "mutable_files": sorted(manifest.patchable_files),
                "immutable_files": sorted(manifest.immutable_files),
                "instruction": "Diagnose the failure and return the smallest unified diff that repairs it. Both diff headers must exactly equal target_file.",
            },
            ensure_ascii=False,
        )
        function = FunctionSpec(
            name="repair_template",
            description="Return one bounded repair patch.",
            json_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "target_file": {"type": "string"},
                    "patch": {"type": "string"},
                    "reason": {"type": "string"},
                    "expected_base_sha256": {"type": ["string", "null"]},
                    "verification": {"type": "string"},
                },
                "required": ["target_file", "patch", "reason"],
            },
        )
        raw = (query_fn or _bounded_openai_query)(
            system_message="You are a controlled ML code repair agent. Never modify immutable files or return shell commands.",
            user_message=prompt,
            model=model
            or __import__("os").getenv("AUTOEXP_PLANNER_MODEL", "deepseek-v4-flash"),
            temperature=0.0,
            max_tokens=2000,
            func_spec=function,
        )
        if isinstance(raw, str):
            raw = json.loads(raw)
        repair = RepairSpec.model_validate(
            raw.get("repair") if isinstance(raw.get("repair"), dict) else raw
        )
        result = self.validate_repair(case, workspace, repair)
        result["failed_before_repair"] = not failed.passed
        return result
