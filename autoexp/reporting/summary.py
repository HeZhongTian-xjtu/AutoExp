from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from autoexp.llm import FunctionSpec
from autoexp.planning.compat import _bounded_openai_query


class SummaryMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["llm", "deterministic"]
    model: str | None = None
    attempts: int = Field(default=1, ge=1)
    elapsed_seconds: float = Field(default=0.0, ge=0.0)
    error: str | None = None


class SummaryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    markdown: str = Field(min_length=1)
    metadata: SummaryMetadata


def _summary_function_spec() -> FunctionSpec:
    return FunctionSpec(
        name="write_experiment_summary",
        description="Write a detailed human-readable report of the complete AutoExp run and every Trial transition.",
        json_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "markdown": {
                    "type": "string",
                    "description": "A detailed Chinese Markdown report following the requested headings and covering every Trial.",
                },
            },
            "required": ["markdown"],
        },
    )


class RunSummaryGenerator:
    """Generate a readable experiment-process report without changing run state."""

    def __init__(
        self,
        mode: str = "deterministic",
        model: str | None = None,
        query_fn: Callable[..., Any] | None = None,
        max_attempts: int = 2,
    ):
        load_dotenv(override=True)
        selected = (mode or "deterministic").strip().lower()
        if selected not in {"auto", "llm", "deterministic"}:
            raise ValueError("summary mode must be auto, llm, or deterministic")
        self.mode = selected
        self.model = model or os.getenv(
            "AUTOEXP_REPORT_MODEL",
            os.getenv("AUTOEXP_PLANNER_MODEL", "deepseek-v4-flash"),
        )
        self.query_fn = query_fn or _bounded_openai_query
        self.max_attempts = max(1, max_attempts)

    def generate(self, run: Any) -> SummaryResult:
        if self.mode != "deterministic" and os.getenv("OPENAI_API_KEY"):
            try:
                return self._generate_llm(run)
            except Exception as exc:
                return SummaryResult(
                    markdown=self._deterministic_summary(run),
                    metadata=SummaryMetadata(
                        source="deterministic",
                        model=self.model,
                        error=str(exc),
                    ),
                )
        return SummaryResult(
            markdown=self._deterministic_summary(run),
            metadata=SummaryMetadata(source="deterministic"),
        )

    def _generate_llm(self, run: Any) -> SummaryResult:
        system_message = (
            "You are the AutoExp experiment reporting assistant. "
            "Write a detailed but readable Chinese Markdown report for a developer or interviewer. "
            "Explain the full experiment trajectory and every executed Trial. "
            "Use only facts in the supplied JSON. Never invent metrics, parameters, causes, decisions, or citations. "
            "When a decision reason is absent, explicitly say that the run record did not provide one."
        )
        started = time.perf_counter()
        last_error: str | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                raw = self.query_fn(
                    system_message=system_message,
                    user_message=self._prompt(run),
                    model=self.model,
                    temperature=0.2,
                    max_tokens=2_400,
                    func_spec=_summary_function_spec(),
                )
                return SummaryResult(
                    markdown=self._extract_markdown(raw),
                    metadata=SummaryMetadata(
                        source="llm",
                        model=self.model,
                        attempts=attempt,
                        elapsed_seconds=time.perf_counter() - started,
                    ),
                )
            except Exception as exc:
                last_error = str(exc)
        raise RuntimeError(last_error or "run summary generation failed")

    def _prompt(self, run: Any) -> str:
        context = self._build_context(run)
        return (
            "Use exactly these Markdown headings:\n"
            "### 实验结论\n"
            "### 整体流程\n"
            "### Trial 演进\n"
            "### Agent 决策复盘\n"
            "### 局限与下一步\n\n"
            "Requirements:\n"
            "1. Trial 演进 must contain one numbered item for every executed Trial, in ordinal order.\n"
            "2. For Trial 1, identify the initial baseline and report its parameters and metric.\n"
            "3. For every later Trial, explain which parameters changed from the previous Trial, how the metric changed, "
            "whether it improved and whether it became the best-so-far.\n"
            "4. Connect each later Trial to selection_decision when available. A decision_after_trial selects a future Trial, "
            "not the Trial that already finished.\n"
            "5. Mention Pytest/Smoke or failure status only when the supplied gate data supports it.\n"
            "6. Explain the metric direction, first-to-best improvement, final stopping reason or budget event, failures, "
            "and one concrete next experiment around the best observed parameters.\n"
            "7. Distinguish observed facts from cautious interpretation. Do not call the best observed Trial a global optimum.\n"
            "8. Target roughly 600-1200 Chinese characters; use compact bullets rather than a large parameter table.\n\n"
            "9. Any next-experiment parameter suggestion must stay inside registered_search_space. Do not claim that "
            "the search space is unknown when registered_search_space is present.\n"
            "10. Describe every parameter direction from the explicit from/to values. Never say all parameters increased "
            "or decreased unless that is numerically true for every changed parameter.\n"
            "11. Do not discuss null optional fields unless their absence caused a failure or materially limits interpretation.\n\n"
            "Run JSON:\n"
            + json.dumps(context, ensure_ascii=False, indent=2, default=str)
        ).replace("\\n", chr(10))

    @staticmethod
    def _build_context(run: Any) -> dict[str, Any]:
        spec = run.spec
        best = run.best
        direction = spec.metric.direction if spec else "minimize"
        decisions = [
            {
                "trial_id": str(record.trial_id) if record.trial_id else None,
                "action": record.decision.action,
                "reason": record.decision.reason,
                "parameters": record.decision.parameters,
                "search_space": record.decision.search_space,
                "source": record.metadata.get("source", "unknown"),
                "remaining_trials": record.observation.remaining_trials,
            }
            for record in run.decisions
        ]
        decisions_by_trial: dict[str, list[dict[str, Any]]] = {}
        for decision in decisions:
            if decision["trial_id"]:
                decisions_by_trial.setdefault(decision["trial_id"], []).append(decision)

        timeline: list[dict[str, Any]] = []
        best_so_far: float | None = None
        best_ordinal: int | None = None
        first_metric: float | None = None
        previous_parameters: dict[str, Any] | None = None
        previous_metric: float | None = None
        previous_trial_id: str | None = None
        for ordinal, outcome in enumerate(run.outcomes, start=1):
            value = outcome.metrics.primary.value if outcome.metrics else None
            if ordinal == 1:
                first_metric = value
            improved_from_previous = RunSummaryGenerator._is_better(
                value, previous_metric, direction
            )
            became_best = RunSummaryGenerator._is_better(value, best_so_far, direction)
            if became_best:
                best_so_far = value
                best_ordinal = ordinal

            selection_decision = None
            if previous_trial_id:
                for candidate in decisions_by_trial.get(previous_trial_id, []):
                    if candidate.get("parameters") == outcome.parameters:
                        selection_decision = candidate
                        break
            gates = [
                {"name": gate.name, "status": gate.status}
                for gate in (outcome.validation.gates if outcome.validation else [])
            ]
            trial_id = str(outcome.trial_id)
            timeline.append(
                {
                    "ordinal": ordinal,
                    "trial_id": trial_id,
                    "status": outcome.status,
                    "parameters": outcome.parameters,
                    "parameter_changes_from_previous": RunSummaryGenerator._parameter_changes(
                        previous_parameters,
                        outcome.parameters,
                    ),
                    "primary_metric": value,
                    "metric_delta_from_previous": (
                        value - previous_metric
                        if value is not None and previous_metric is not None
                        else None
                    ),
                    "improvement_percent_from_previous": RunSummaryGenerator._improvement_percent(
                        previous_metric,
                        value,
                        direction,
                    ),
                    "improved_from_previous": (
                        improved_from_previous if previous_metric is not None else None
                    ),
                    "became_best_so_far": became_best,
                    "best_so_far": best_so_far,
                    "secondary_metrics": (
                        outcome.metrics.secondary if outcome.metrics else {}
                    ),
                    "preflight_passed": bool(
                        outcome.preflight and outcome.preflight.passed
                    ),
                    "validation_gates": gates,
                    "failure_codes": [issue.code for issue in outcome.issues],
                    "selection_decision": selection_decision,
                    "decisions_after_trial": decisions_by_trial.get(trial_id, []),
                }
            )
            previous_parameters = dict(outcome.parameters)
            previous_metric = value
            previous_trial_id = trial_id

        best_metric = best.metrics.primary.value if best and best.metrics else None
        event_sequence = [
            {
                "name": event.name,
                "message": event.message,
                "trial_id": str(event.trial_id) if event.trial_id else None,
            }
            for event in run.events
            if event.name
            in {"plan.generated", "experiment.budget_reached", "experiment.completed"}
            or event.name.startswith(("trial.", "action.", "repair."))
        ]
        return {
            "status": run.status,
            "phase": getattr(run, "phase", run.status),
            "objective": spec.objective if spec else None,
            "hypothesis": spec.hypothesis if spec else None,
            "template_id": spec.template_id if spec else None,
            "dataset_id": spec.dataset_id if spec else None,
            "model_id": spec.model_id if spec else None,
            "metric": spec.metric.model_dump(mode="json") if spec else None,
            "budget": spec.budget.model_dump(mode="json") if spec else None,
            "seed": spec.seed if spec else None,
            "fixed_parameters": spec.fixed_parameters if spec else {},
            "registered_search_space": {
                name: parameter_range.model_dump(mode="json")
                for name, parameter_range in (spec.search_space.items() if spec else [])
            },
            "trial_count": len(run.outcomes),
            "best_trial_ordinal": best_ordinal,
            "best_metric": best_metric,
            "best_parameters": best.parameters if best else None,
            "first_metric": first_metric,
            "first_to_best_improvement_percent": RunSummaryGenerator._improvement_percent(
                first_metric,
                best_metric,
                direction,
            ),
            "trial_timeline": timeline,
            "agent_decisions": decisions,
            "event_sequence": event_sequence,
            "issues": [issue.model_dump(mode="json") for issue in run.issues],
        }

    @staticmethod
    def _parameter_changes(
        previous: dict[str, Any] | None, current: dict[str, Any]
    ) -> list[dict[str, Any]]:
        if previous is None:
            return []
        return [
            {"name": name, "from": previous.get(name), "to": current.get(name)}
            for name in sorted(set(previous) | set(current))
            if previous.get(name) != current.get(name)
        ]

    @staticmethod
    def _is_better(
        current: float | None, reference: float | None, direction: str
    ) -> bool:
        if current is None:
            return False
        if reference is None:
            return True
        return current > reference if direction == "maximize" else current < reference

    @staticmethod
    def _improvement_percent(
        previous: float | None, current: float | None, direction: str
    ) -> float | None:
        if previous is None or current is None or previous == 0:
            return None
        signed_gain = (
            current - previous if direction == "maximize" else previous - current
        )
        return signed_gain / abs(previous) * 100.0

    @staticmethod
    def _extract_markdown(raw: Any) -> str:
        value = raw
        if isinstance(raw, dict):
            value = (
                raw.get("markdown") or raw.get("report_markdown") or raw.get("summary")
            )
        if not isinstance(value, str):
            raise ValueError("summary response did not contain Markdown text")
        text = value.strip()
        fence = chr(96) * 3
        if text.startswith(fence):
            text = text[len(fence) :].lstrip()
        if text.endswith(fence):
            text = text[: -len(fence)].rstrip()
        if len(text) < 40:
            raise ValueError("summary response was too short")
        return text[:8_000]

    @staticmethod
    def _deterministic_summary(run: Any) -> str:
        context = RunSummaryGenerator._build_context(run)
        metric = context.get("metric") or {}
        metric_name = metric.get("name", "primary metric")
        direction = "越高越好" if metric.get("direction") == "maximize" else "越低越好"
        best_metric = context.get("best_metric")
        improvement = context.get("first_to_best_improvement_percent")
        if best_metric is not None:
            conclusion = (
                f"本次实验状态为 {run.status}，共执行 {context['trial_count']} 轮 Trial。"
                f"最佳 {metric_name} 为 {best_metric:.6f}（{direction}），出现在 Trial {context['best_trial_ordinal']}。"
            )
            if improvement is not None:
                conclusion += f"相对首轮基线改善 {improvement:.2f}%。"
        else:
            conclusion = f"本次实验状态为 {run.status}，共执行 {context['trial_count']} 轮 Trial，但没有产生可用主指标。"

        timeline_lines: list[str] = []
        for trial in context["trial_timeline"]:
            metric_text = (
                f"{metric_name}={trial['primary_metric']:.6f}"
                if trial["primary_metric"] is not None
                else "无有效指标"
            )
            if trial["ordinal"] == 1:
                transition = "初始基线；参数=" + json.dumps(
                    trial["parameters"], ensure_ascii=False, sort_keys=True
                )
            else:
                changes = trial["parameter_changes_from_previous"]
                transition = (
                    "；".join(
                        f"{item['name']}: {item['from']}→{item['to']}"
                        for item in changes
                    )
                    or "参数未变化"
                )
                percent = trial["improvement_percent_from_previous"]
                if percent is not None:
                    transition += f"；相对上一轮{'改善' if percent > 0 else '下降'} {abs(percent):.2f}%"
            best_mark = "，刷新当前最佳" if trial["became_best_so_far"] else ""
            selector = trial.get("selection_decision")
            reason = f"；选择依据：{selector['reason']}" if selector else ""
            timeline_lines.append(
                f"{trial['ordinal']}. Trial {trial['ordinal']}：{transition}；{metric_text}{best_mark}；状态 {trial['status']}{reason}。"
            )

        decision_lines = [
            f"- Trial {index} 后：{decision['action']}（{decision['source']}）。{decision['reason']}"
            for index, decision in enumerate(context["agent_decisions"], start=1)
        ] or ["- 没有记录到 Trial 后续决策。"]
        issue_count = len(context["issues"]) + sum(
            len(item["failure_codes"]) for item in context["trial_timeline"]
        )
        limitation = "当前最佳仅是有限预算内的最佳观测结果，不能视为全局最优。"
        if issue_count:
            limitation += f" 运行记录中包含 {issue_count} 项问题或失败代码，应结合 Run details 复核。"
        suggestion = (
            f"下一轮建议围绕 Trial {context['best_trial_ordinal']} 的参数做更窄范围搜索，并使用额外随机种子验证稳定性。"
            if context.get("best_trial_ordinal")
            else "下一轮建议先修复失败原因，再从可复现基线重新开始。"
        )
        objective = context.get("objective") or "未生成有效 ExperimentSpec"
        event_names = (
            " → ".join(item["name"] for item in context["event_sequence"])
            or "无事件记录"
        )
        return "\n".join(
            [
                "### 实验结论",
                conclusion,
                "",
                "### 整体流程",
                f'实验目标：{objective}。固定随机种子为 {context.get("seed")}，执行轨迹为：{event_names}。',
                "",
                "### Trial 演进",
                *timeline_lines,
                "",
                "### Agent 决策复盘",
                *decision_lines,
                "",
                "### 局限与下一步",
                limitation,
                suggestion,
            ]
        )


__all__ = ["RunSummaryGenerator", "SummaryMetadata", "SummaryResult"]
