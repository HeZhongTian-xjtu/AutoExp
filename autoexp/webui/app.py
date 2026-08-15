"""Standalone AutoExp Streamlit entrypoint.

This module owns the public experiment workflow and has no legacy page.
"""

from __future__ import annotations

import json
from html import escape
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from uuid import uuid4

# Streamlit may use the script directory as sys.path[0]. Add the repository
# root explicitly so sibling packages such as ``autoexp`` work from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import requests
import streamlit as st

from autoexp.application import AutoExpApplicationService, TemplateCatalog
from autoexp.evaluation import profile_dataset


class AutoExpWebUI:
    """Own the AutoExp page lifecycle."""

    def __init__(self) -> None:
        self.project_root = PROJECT_ROOT
        self.config_session_state()
        self.setup_page()

    @staticmethod
    def config_session_state() -> None:
        defaults = {
            "autoexp_results": None,
            "autoexp_confirm_delete": False,
            "autoexp_delete_notice": None,
            "autoexp_background_run_id": None,
            "autoexp_prompt_template_id": None,
        }
        for key, value in defaults.items():
            st.session_state.setdefault(key, value)

    @staticmethod
    def setup_page() -> None:
        st.set_page_config(page_title="AutoExp", layout="wide")
        css_file = Path(__file__).parent / "style.css"
        if css_file.exists():
            st.markdown(
                f"<style>{css_file.read_text(encoding='utf-8')}</style>",
                unsafe_allow_html=True,
            )

    def run(self) -> None:
        """Render the single public AutoExp workflow."""
        self.render_sidebar()
        autoexp_tab = st.tabs(["AutoExp Unified"])[0]
        with autoexp_tab:
            self.render_autoexp_panel()

    @staticmethod
    def render_sidebar() -> None:
        with st.sidebar:
            st.header("AutoExp")
            st.caption("Controlled LLM planning and reproducible experiments")
            st.caption(
                f"Model: {os.getenv('AUTOEXP_PLANNER_MODEL', os.getenv('OPENAI_MODEL', 'configured at runtime'))}"
            )


def _render_delete_records(project_root: Path) -> None:
    if st.button(
        "Delete all records",
        key="show_delete_records",
        help="Delete persisted runs, decisions, events and generated artifacts.",
        width="stretch",
    ):
        st.session_state.autoexp_confirm_delete = True
    if not st.session_state.get("autoexp_confirm_delete"):
        return

    st.warning(
        "This deletes all AutoExp experiment records and generated artifacts. Templates and input data are kept."
    )
    confirm_col, cancel_col = st.columns(2)
    with confirm_col:
        if st.button("Confirm deletion", type="primary", key="confirm_delete_records"):
            service = AutoExpApplicationService(project_root, summary_mode="disabled")
            deleted = service.delete_all_runs()
            st.session_state.autoexp_results = None
            st.session_state.autoexp_confirm_delete = False
            st.session_state.autoexp_delete_notice = (
                f"Deleted {deleted} experiment run(s)."
            )
            st.rerun()
    with cancel_col:
        if st.button("Cancel", key="cancel_delete_records"):
            st.session_state.autoexp_confirm_delete = False
            st.rerun()


def _render_autoexp_dataset_selector(service: AutoExpApplicationService) -> str:
    st.subheader("Dataset")
    datasets = service.list_datasets()
    options = [dataset.dataset_id for dataset in datasets]
    selected_key = "autoexp_selected_dataset_id"
    by_id = {dataset.dataset_id: dataset for dataset in datasets}
    selected_dataset_id = ""
    selected = None
    if options:
        if st.session_state.get(selected_key) not in options:
            st.session_state[selected_key] = options[0]
        selected_dataset_id = st.selectbox(
            "Dataset",
            options=options,
            key=selected_key,
            format_func=lambda value: f"{by_id[value].display_name}  | Uploaded",
        )
        selected = by_id[selected_dataset_id]
        train_file = selected.file_for_role("train")
        if train_file:
            st.caption(
                f'{train_file.rows_including_header or "?"} rows  |  '
                f'{len(train_file.columns)} columns  |  target: {selected.target_column or "not specified"}'
            )
    else:
        st.info("No compatible dataset is registered. Upload a train.csv to begin.")

    with st.expander("Add dataset", expanded=not bool(options)):
        uploaded_files = st.file_uploader(
            "Upload dataset files",
            accept_multiple_files=True,
            type=["csv", "txt", "json", "md"],
            key="autoexp_dataset_uploads",
            help="Upload train.csv and optional test.csv or data description. The files are registered for future runs after validation.",
        )
        display_name = st.text_input(
            "Dataset name",
            value="",
            placeholder="Example: House Prices subset",
            key="autoexp_dataset_display_name",
        )
        st.caption(
            "Upload a compatible train.csv. Optional test.csv and data description files can be included."
        )
        if st.button(
            "Register dataset", key="autoexp_register_dataset", width="stretch"
        ):
            if not uploaded_files:
                st.warning("Select at least one dataset file first.")
            else:
                try:
                    record = service.register_dataset(
                        [(item.name, item.getvalue()) for item in uploaded_files],
                        display_name=display_name.strip() or None,
                    )
                    st.session_state[selected_key] = record.dataset_id
                    st.session_state.autoexp_dataset_notice = (
                        f"Registered dataset: {record.display_name}"
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Dataset registration failed: {exc}")

    if st.session_state.get("autoexp_dataset_notice"):
        st.success(st.session_state.pop("autoexp_dataset_notice"))
    if selected is not None:
        with st.expander("Selected dataset files", expanded=True):
            for item in selected.files:
                size_kb = max(1, round(item.size_bytes / 1024))
                st.write(f"{item.role}: {item.name}  |  {size_kb} KB")
        with st.expander("Dataset preview", expanded=False):
            profile = profile_dataset(selected.storage_path or "", include_preview=True)
            csv_files = [
                item for item in profile.get("files", []) if item.get("format") == "csv"
            ]
            if not csv_files:
                st.info("No local CSV preview is available for this dataset.")
            for item in csv_files:
                st.caption(
                    f"{item.get('path', '-')}: {item.get('rows', '?')} rows, "
                    f"{len(item.get('columns') or [])} columns"
                )
                preview = item.get("preview") or []
                if preview:
                    st.dataframe(
                        pd.DataFrame(preview), width="stretch", hide_index=True
                    )
    return selected_dataset_id


def _render_autoexp_panel_v2(self: AutoExpWebUI) -> None:
    header_left, header_right = st.columns([5, 1.8], gap="large")
    with header_left:
        st.header("AutoExp Unified")
        st.caption("Experiment setup and execution")
    with header_right:
        _render_delete_records(self.project_root)
    if st.session_state.get("autoexp_delete_notice"):
        st.success(st.session_state.pop("autoexp_delete_notice"))
    setup_left, setup_right = st.columns([1.55, 1], gap="large")
    with setup_left:
        st.subheader("Run configuration")
        template_catalog = TemplateCatalog(self.project_root)
        template_descriptors = template_catalog.list_templates()
        template_ids = [item.template_id for item in template_descriptors]
        template_names = {
            item.template_id: item.display_name for item in template_descriptors
        }
        if not template_ids:
            st.error("No experiment templates are registered.")
            return
        if st.session_state.get("autoexp_template_id") not in template_ids:
            st.session_state.autoexp_template_id = template_ids[0]
        template_id = st.selectbox(
            "Experiment template",
            options=template_ids,
            format_func=lambda value: template_names.get(value, value),
            key="autoexp_template_id",
        )
        selected_descriptor = template_catalog.get(template_id)
        template_meta = (
            f"{selected_descriptor.manifest.dataset_id} / "
            f"{selected_descriptor.manifest.metric_name} / "
            f"{selected_descriptor.manifest.metric_direction}"
        )
        st.caption(template_meta)

        planner_mode = st.selectbox(
            "Planner mode",
            options=["llm", "deterministic"],
            index=0,
            format_func=lambda value: (
                "LLM Planner" if value == "llm" else "Deterministic Planner"
            ),
            key="autoexp_planner_mode",
        )
        executor_col, tracker_col = st.columns(2, gap="medium")
        with executor_col:
            executor_mode = st.selectbox(
                "Executor",
                options=["local", "docker"],
                index=(
                    0
                    if os.getenv("AUTOEXP_EXECUTOR", "docker").lower() == "local"
                    else 1
                ),
                format_func=lambda value: (
                    "Local development" if value == "local" else "Docker isolated"
                ),
                key="autoexp_executor_mode",
                help="Docker mode requires a running Docker daemon and the autoexp-runner image.",
            )
        with tracker_col:
            tracker_mode = st.selectbox(
                "Experiment tracking",
                options=["none", "mlflow"],
                index=(
                    0 if os.getenv("AUTOEXP_TRACKER", "mlflow").lower() == "none" else 1
                ),
                format_func=lambda value: (
                    "SQLite only" if value == "none" else "SQLite + MLflow"
                ),
                key="autoexp_tracker_mode",
                help="MLflow is optional; SQLite remains the source of truth.",
            )
        max_trials = st.slider(
            "Maximum Trials",
            min_value=1,
            max_value=8,
            value=2,
            key="autoexp_trials",
        )
        generate_summary = st.checkbox(
            "Generate AI Run Summary",
            value=False,
            key="autoexp_generate_summary",
            help="Generate a short human-readable report after the run. This uses one model request.",
        )
        api_url = os.getenv("AUTOEXP_API_URL", "").rstrip("/")
        background_run = st.checkbox(
            "Run as background task",
            value=bool(api_url),
            disabled=not bool(api_url),
            help="Available when AUTOEXP_API_URL points to the AutoExp FastAPI service.",
        )
        st.caption("The LLM Planner generates the ExperimentSpec before execution.")

        st.subheader("Research question")
        default_objective = (
            selected_descriptor.manifest.default_objective
            or f"Optimize the registered {selected_descriptor.manifest.metric_name} metric."
        )
        default_hypothesis = (
            selected_descriptor.manifest.default_hypothesis
            or f"Registered parameter changes improve {selected_descriptor.manifest.metric_name}."
        )
        if st.session_state.get("autoexp_prompt_template_id") != template_id:
            st.session_state.autoexp_objective = default_objective
            st.session_state.autoexp_hypothesis = default_hypothesis
            st.session_state.autoexp_prompt_template_id = template_id
        objective = st.text_area(
            "Experiment objective",
            key="autoexp_objective",
            height=90,
        )
        hypothesis = st.text_area(
            "Hypothesis",
            key="autoexp_hypothesis",
            height=90,
        )
    with setup_right:
        dataset_service = AutoExpApplicationService(
            self.project_root,
            template_id=template_id,
            summary_mode="disabled",
            tracker_mode="none",
        )
        dataset_id = _render_autoexp_dataset_selector(dataset_service)
    if st.button("Run AutoExp", type="primary", key="run_autoexp", width="stretch"):
        if not dataset_id:
            st.warning(
                "Upload and register a compatible dataset before running AutoExp."
            )
            return
        output_root = Path(
            os.getenv(
                "AUTOEXP_WEB_OUTPUT_ROOT",
                str(self.project_root / "workspaces" / "autoexp-web"),
            )
        )
        progress_path = output_root / ".progress" / f"{uuid4()}.json"
        try:
            if background_run:
                queued = _submit_background_run(
                    api_url,
                    {
                        "objective": objective,
                        "hypothesis": hypothesis,
                        "template_id": template_id,
                        "dataset_id": dataset_id or None,
                        "planner_mode": planner_mode,
                        "executor_mode": executor_mode,
                        "tracker_mode": tracker_mode,
                        "max_trials": max_trials,
                        "generate_summary": generate_summary,
                    },
                )
                st.session_state.autoexp_background_run_id = queued["run_id"]
                st.success(f"Background Run queued: {queued['run_id']}")
                st.rerun()
            service = AutoExpApplicationService(
                self.project_root,
                template_id=template_id,
                planner_mode=planner_mode,
                summary_mode="auto" if generate_summary else "disabled",
                executor_mode=executor_mode,
                tracker_mode=tracker_mode,
            )
            progress_slot = st.empty()
            with ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="autoexp-web-run"
            ) as pool:
                future = pool.submit(
                    service.run,
                    objective=objective,
                    hypothesis=hypothesis,
                    max_trials=max_trials,
                    planner_mode=planner_mode,
                    output_root=output_root,
                    dataset_id=dataset_id or None,
                    run_id=uuid4(),
                    progress_path=progress_path,
                )
                while not future.done():
                    _render_run_progress(progress_slot, progress_path, max_trials)
                    time.sleep(0.5)
                run = future.result()
            _render_run_progress(progress_slot, progress_path, max_trials)
            st.session_state.autoexp_results = service.summarize(run)
        except Exception as exc:
            st.session_state.autoexp_results = None
            st.error(f"AutoExp failed: {exc}")

    if api_url and st.session_state.get("autoexp_background_run_id"):
        _render_background_run(api_url, st.session_state.autoexp_background_run_id)

    try:
        history_service = AutoExpApplicationService(
            self.project_root,
            template_id=template_id,
            planner_mode=planner_mode,
            summary_mode="deterministic",
            executor_mode=executor_mode,
            tracker_mode="none",
        )
        history = history_service.list_runs(limit=10)
    except Exception as exc:
        history = []
        st.caption(f"Run history unavailable: {exc}")

    if st.session_state.get("autoexp_results"):
        returned_to_list = _render_autoexp_results_v2(
            st.session_state.autoexp_results,
            history,
            history_service,
        )
        if returned_to_list:
            _render_run_history_table(history, history_service)
    else:
        _render_run_history_table(history, history_service)


def _read_run_progress(path: Path) -> dict[str, Any]:
    try:
        if not path.is_file():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _api_headers() -> dict[str, str]:
    token = os.getenv("AUTOEXP_API_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _submit_background_run(api_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(
        f"{api_url}/api/runs", json=payload, headers=_api_headers(), timeout=15
    )
    response.raise_for_status()
    return response.json()


def _render_background_run(api_url: str, run_id: str) -> None:
    with st.container(border=True):
        left, right = st.columns([4, 1])
        left.markdown(f"**Background Run** `{run_id}`")
        if right.button("Refresh", key=f"refresh_background_{run_id}", width="stretch"):
            st.rerun()
        try:
            response = requests.get(
                f"{api_url}/api/runs/{run_id}", headers=_api_headers(), timeout=10
            )
            response.raise_for_status()
            task = response.json()
            st.status(
                f"Status: {task['status']}",
                state=(
                    "complete"
                    if task["status"] == "COMPLETED"
                    else "error" if task["status"] == "FAILED" else "running"
                ),
            )
            if task.get("result") and task["status"] == "COMPLETED":
                st.session_state.autoexp_results = task["result"]
            if task["status"] not in {"COMPLETED", "FAILED", "CANCELLED"}:
                if st.button("Cancel run", key=f"cancel_background_{run_id}"):
                    requests.post(
                        f"{api_url}/api/runs/{run_id}/cancel",
                        headers=_api_headers(),
                        timeout=10,
                    ).raise_for_status()
                    st.rerun()
        except Exception as exc:
            st.warning(f"Background Run status unavailable: {exc}")


def _render_run_progress(slot: Any, path: Path, fallback_total: int) -> None:
    progress = _read_run_progress(path)
    total = max(1, int(progress.get("total_trials") or fallback_total))
    completed = min(total, max(0, int(progress.get("completed_trials") or 0)))
    current = progress.get("trial_index")
    phase = str(progress.get("message") or "Planning and running AutoExp...")
    fraction = completed / total
    if current and completed < total:
        label = f"Trial {current} / {total}  |  {phase}"
    elif completed >= total:
        label = f"All {total} Trials completed"
    else:
        label = phase
    with slot.container():
        st.progress(fraction, text=label)
        st.caption(f"Completed Trials: {completed} / {total}")


def _render_autoexp_results_v2(
    summary: dict[str, Any],
    history: list[dict[str, Any]],
    history_service: AutoExpApplicationService,
) -> bool:
    result_header_left, result_header_right = st.columns([5, 1.2], gap="large")
    with result_header_left:
        st.subheader("AutoExp Run Result")
    with result_header_right:
        if st.button(
            "back",
            key=f'back_to_run_list_{summary.get("run_id", "run")}',
            width="stretch",
        ):
            st.session_state.autoexp_results = None
            return True
    status = summary.get("status", "UNKNOWN")
    metric = summary.get("best_metric")
    planner = summary.get("planner") or {}
    metric_name = ((summary.get("spec") or {}).get("metric") or {}).get(
        "name", "metric"
    )
    run_id = summary.get("run_id", "-")
    planner_source = planner.get("source", "unknown")

    result_left, result_right = st.columns([1, 1.35], gap="large")
    with result_left:
        st.markdown("#### Run overview")
        st.metric("Status", status)
        st.metric("Trials", summary.get("trial_count", 0))
        st.caption(f"Run ID: {run_id}")
        st.caption(f"Planner: {planner_source}")
        baseline = planner.get("baseline_parameters")
        if baseline:
            st.caption("Weak baseline parameters")
            st.json(baseline)
    with result_right:
        st.markdown("#### Best result")
        st.metric(f"Best {metric_name}", f"{metric:.4f}" if metric is not None else "-")
        metric_rows = [
            {"trial": index, "primary_metric": trial["metric"]}
            for index, trial in enumerate(summary.get("trials", []), start=1)
            if trial.get("metric") is not None
        ]
        if metric_rows:
            st.caption("Primary metric by Trial")
            st.line_chart(pd.DataFrame(metric_rows).set_index("trial"), height=190)
        else:
            st.info("No completed metric is available.")

    with st.container(border=True):
        st.markdown("### Run details")
        show_summary = bool(summary.get("ai_summary"))
        has_repairs = any(
            item.get("action") == "REPAIR" for item in summary.get("decisions", [])
        )
        tab_names = [
            "Trials",
            "ExperimentSpec",
            "Planner metadata",
            "Agent decisions",
            "Execution timeline",
            "Tracking",
            "Artifacts",
            "Experiment report",
            "History",
        ]
        if has_repairs:
            tab_names.insert(tab_names.index("Execution timeline"), "Repairs")
        if show_summary:
            tab_names.insert(0, "AI Run Summary")
        tabs = dict(zip(tab_names, st.tabs(tab_names)))
        if show_summary:
            with tabs["AI Run Summary"]:
                _render_ai_summary(summary)
        with tabs["Trials"]:
            _render_trial_table(summary)
        with tabs["ExperimentSpec"]:
            st.json(summary.get("spec") or {})
        with tabs["Planner metadata"]:
            st.json(planner)
        with tabs["Agent decisions"]:
            _render_decision_table(summary.get("decisions") or [])
        if has_repairs:
            with tabs["Repairs"]:
                _render_repairs(summary)
        with tabs["Execution timeline"]:
            _render_event_table(summary.get("events") or [])
        with tabs["Tracking"]:
            tracking = summary.get("tracking") or {}
            if tracking.get("run_url"):
                st.link_button("Open in MLflow", tracking["run_url"])
            st.json(tracking)
        with tabs["Artifacts"]:
            _render_artifact_table(summary.get("artifacts") or [])
        with tabs["Experiment report"]:
            _render_report(summary)
        with tabs["History"]:
            _render_run_history_table(history, history_service)

    issues = summary.get("issues") or []
    if issues:
        st.error(json.dumps(issues, ensure_ascii=False, indent=2))
    return False


def _render_trial_table(summary: dict[str, Any]) -> None:
    rows = [
        {
            "trial_id": trial["trial_id"],
            "status": trial["status"],
            "preflight": "passed" if trial["preflight_passed"] else "not passed",
            "metric": trial["metric"],
            "pytest": _gate_status(trial.get("validation"), "pytest"),
            "smoke": _gate_status(trial.get("validation"), "smoke"),
            "secondary_metrics": json.dumps(
                trial.get("secondary_metrics") or {}, ensure_ascii=False
            ),
            "dataset_sha256": trial.get("dataset_sha256"),
            "parameters": json.dumps(trial["parameters"], ensure_ascii=False),
        }
        for trial in summary.get("trials", [])
    ]
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.info("No trial records.")


def _gate_status(validation: dict[str, Any] | None, name: str) -> str:
    for gate in (validation or {}).get("gates", []):
        if gate.get("name") == name:
            return gate.get("status", "-")
    return "-"


def _render_ai_summary(summary: dict[str, Any]) -> None:
    report = summary.get("ai_summary") or ""
    metadata = summary.get("ai_summary_metadata") or {}
    if metadata:
        source = metadata.get("source", "unknown")
        model = metadata.get("model")
        label = f"Generated by {source}"
        if model:
            label += f" / {model}"
        st.caption(label)
    if report:
        st.markdown(report)
    else:
        st.info("No AI run summary was generated.")


def _render_event_table(events: list[dict[str, Any]]) -> None:
    if events:
        st.dataframe(
            [
                {
                    "time": event.get("created_at"),
                    "event": event.get("name"),
                    "message": event.get("message"),
                    "trial_id": event.get("trial_id"),
                }
                for event in events
            ],
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No events recorded.")


def _render_decision_table(decisions: list[dict[str, Any]]) -> None:
    if decisions:
        st.dataframe(
            [
                {
                    "time": decision.get("created_at"),
                    "action": decision.get("action"),
                    "strategy": (decision.get("metadata") or {}).get("strategy"),
                    "trial_id": decision.get("trial_id"),
                    "reason": decision.get("reason"),
                    "source": (decision.get("metadata") or {}).get("source"),
                    "best_metric": (decision.get("observation") or {}).get(
                        "best_metric"
                    ),
                    "remaining_trials": (decision.get("observation") or {}).get(
                        "remaining_trials"
                    ),
                    "parameters": (
                        json.dumps(decision.get("parameters"), ensure_ascii=False)
                        if decision.get("parameters")
                        else ""
                    ),
                }
                for decision in decisions
            ],
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No Agent decisions recorded.")


def _render_repairs(summary: dict[str, Any]) -> None:
    repairs = [
        item for item in summary.get("decisions", []) if item.get("action") == "REPAIR"
    ]
    for index, item in enumerate(repairs, start=1):
        metadata = item.get("metadata") or {}
        title = (
            "Code optimization"
            if metadata.get("strategy") == "code_optimization"
            else "Repair"
        )
        with st.expander(
            f"{title} {index}: {metadata.get('repair_status', 'proposed')}",
            expanded=index == 1,
        ):
            st.write(item.get("reason") or "")
            st.caption(
                f"Failure fingerprint: {metadata.get('failure_fingerprint') or '-'}"
            )
            st.caption(f"Repair error: {metadata.get('repair_error') or '-'}")
            repair = ((item.get("observation") or {}).get("failure_context") or {}).get(
                "recent_diffs"
            )
            if repair:
                st.json(repair)


def _render_artifact_table(artifacts: list[dict[str, Any]]) -> None:
    if artifacts:
        st.dataframe(
            [
                {
                    "kind": item.get("kind"),
                    "trial_id": item.get("trial_id"),
                    "path": item.get("relative_path"),
                    "size_bytes": item.get("size_bytes"),
                    "sha256": item.get("sha256"),
                }
                for item in artifacts
            ],
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No artifacts recorded.")


def _render_report(summary: dict[str, Any]) -> None:
    report = summary.get("report_markdown") or ""
    if not report:
        st.info("No report has been generated.")
        return
    report_run_id = summary.get("run_id", "run")
    st.download_button(
        "Download report",
        data=report,
        file_name=f"autoexp-{report_run_id}.md",
        mime="text/markdown",
        key=f"download_report_{report_run_id}",
    )
    st.markdown(report)


def _history_task_or_dataset(item: dict[str, Any]) -> str:
    """Return a readable task or Dataset label for a persisted run."""
    return str(
        item.get("dataset_name")
        or item.get("dataset_id")
        or item.get("template_id")
        or item.get("objective")
        or "-"
    )


def _render_history_task_or_dataset(column: Any, item: dict[str, Any]) -> None:
    label = escape(_history_task_or_dataset(item))
    column.markdown(
        f"<div class='autoexp-history-task' title='{label}'>{label}</div>",
        unsafe_allow_html=True,
    )


def _render_run_history_table(
    history: list[dict[str, Any]],
    history_service: AutoExpApplicationService,
) -> None:
    if not history:
        st.info("No persisted AutoExp runs.")
        return
    st.caption("Select a Run ID to open its complete run details.")
    header = st.columns([2.7, 2.6, 1, 1.2, 2.2, 1])
    for column, label in zip(
        header,
        ["Run ID", "Task / Dataset", "Status", "Planner", "Updated at", "Summary"],
    ):
        column.markdown(f"**{label}**")

    for item in history:
        run_id = item.get("run_id") or "-"
        row = st.columns([2.7, 2.6, 1, 1.2, 2.2, 1])
        with row[0]:
            if st.button(
                run_id,
                key=f"load_autoexp_run_{run_id}",
                help="Load this persisted run and open its details.",
                width="stretch",
            ):
                try:
                    loaded = history_service.load_run(run_id)
                    if loaded is None:
                        st.error(f"Run {run_id} could not be found.")
                    else:
                        st.session_state.autoexp_results = history_service.summarize(
                            loaded
                        )
                        st.rerun()
                except Exception as exc:
                    st.error(f"Unable to load run {run_id}: {exc}")
        _render_history_task_or_dataset(row[1], item)
        row[2].write(item.get("status") or "-")
        row[3].write((item.get("planner") or {}).get("source", "unknown"))
        row[4].write(item.get("updated_at") or "-")
        row[5].write("Yes" if item.get("has_ai_summary") else "No")


AutoExpWebUI.render_autoexp_panel = _render_autoexp_panel_v2


if __name__ == "__main__":
    app = AutoExpWebUI()
    app.run()
