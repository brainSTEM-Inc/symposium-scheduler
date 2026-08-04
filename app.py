from __future__ import annotations

import csv
import json
import io
import os
import threading
import uuid
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from flask import Flask, Response, jsonify, redirect, render_template, request, session, url_for
from werkzeug.exceptions import RequestEntityTooLarge

from scheduler import (
    DEFAULT_COL_CONFIG,
    DEFAULT_STRUCTURE,
    PENALTIES,
    _parse_fixed_empty_slots,
    _score_schedule,
    load_dataframe,
    parse_presenters,
    run,
)


app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "dev-change-me")
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_CONTENT_LENGTH", str(16 * 1024 * 1024)))
app.config["JSON_SORT_KEYS"] = False

ALLOWED_CSV_EXTENSIONS = {".csv", ".txt"}
WORKFLOW_TTL_SECONDS = int(os.getenv("WORKFLOW_TTL_SECONDS", str(60 * 60 * 24)))
WORKFLOW_MAX_ACTIVE = int(os.getenv("WORKFLOW_MAX_ACTIVE", "120"))


def _is_allowed_upload(filename: str) -> bool:
    if not filename:
        return False
    extension = Path(filename).suffix.lower()
    if extension in ALLOWED_CSV_EXTENSIONS:
        return True
    return False


WORKFLOW_STATES: Dict[str, Dict[str, Any]] = {}
WORKFLOW_LOCK = threading.RLock()


def _split_csv_lines(raw: str):
    values = [item.strip() for item in (raw or "").split(",") if item.strip()]
    return values


def _normalize_json(raw: str, fallback: Any) -> Any:
    if not raw:
        return fallback
    try:
        parsed = json.loads(raw)
    except Exception:
        return fallback
    return parsed


def _coerce_workflow_id() -> Optional[str]:
    workflow_id = session.get("workflow_id")
    if isinstance(workflow_id, str):
        return workflow_id
    return None


def _with_state():
    workflow_id = _coerce_workflow_id()
    if workflow_id is None:
        return None, None
    return workflow_id, WORKFLOW_STATES.get(workflow_id)


def _prune_workflow_states():
    now = time.time()
    with WORKFLOW_LOCK:
        expired: list[str] = []
        for workflow_id, state in list(WORKFLOW_STATES.items()):
            created_at = float(state.get("created_at", 0))
            if created_at and (now - created_at) > WORKFLOW_TTL_SECONDS:
                expired.append(workflow_id)
        # Hard cap to avoid unbounded growth even if timestamps are missing.
        if len(WORKFLOW_STATES) > WORKFLOW_MAX_ACTIVE:
            ordered = sorted(
                WORKFLOW_STATES.items(),
                key=lambda item: float(item[1].get("created_at", 0)),
            )
            overflow = len(WORKFLOW_STATES) - WORKFLOW_MAX_ACTIVE
            if overflow > 0:
                expired.extend([workflow_id for workflow_id, _ in ordered[:overflow]])

        for workflow_id in set(expired):
            state = WORKFLOW_STATES.pop(workflow_id, None)
            if not state:
                continue
            source = state.get("source", "")
            source_type = state.get("source_type")
            if source_type == "file":
                try:
                    Path(str(source)).unlink(missing_ok=True)
                except Exception:
                    pass


@app.errorhandler(RequestEntityTooLarge)
def _handle_too_large_error(exc):
    return (
        render_template(
            "upload.html",
            error="The uploaded file is too large. Maximum size is 16 MB.",
            col_config=DEFAULT_COL_CONFIG,
            structure=DEFAULT_STRUCTURE,
            PENALTIES=PENALTIES,
        ),
        413,
    )


def _collect_topics(df_records, col_config):
    topics = []
    for row in df_records:
        raw_topics = str(row.get(col_config["topics"], "")).split(",")
        for topic in raw_topics:
            topic = topic.strip()
            if topic and topic not in topics:
                topics.append(topic)
    return topics


def _parse_schedule_slot(raw_slot: Any) -> Optional[tuple[str, str, int]]:
    if raw_slot is None:
        return None
    text = str(raw_slot)
    parts = text.split("|")
    if len(parts) != 3:
        return None
    period = parts[0].strip()
    day = parts[1].strip()
    try:
        room = int(parts[2])
    except (TypeError, ValueError):
        return None
    return period, day, room


def _build_candidate_slot_rows(schedule: Dict[str, Any], structure: Dict[str, Any]) -> list[dict[str, Any]]:
    day_order = {day: i for i, day in enumerate(structure.get("days", []))}
    period_order = {period: i for i, period in enumerate(structure.get("periods", []))}
    rows: list[dict[str, Any]] = []

    for key, presenters in schedule.items():
        parsed = _parse_schedule_slot(key)
        if parsed is None:
            continue
        period, day, room = parsed
        presenters_list = [str(p) for p in (presenters or [])]
        rows.append(
            {
                "day": day,
                "period": period,
                "room": room,
                "presenters": presenters_list,
            }
        )

    if not rows and schedule:
        for raw_key in schedule:
            parsed = _parse_schedule_slot(raw_key)
            if parsed is None:
                continue
            period, day, room = parsed
            if day not in day_order:
                day_order[day] = len(day_order)
            if period not in period_order:
                period_order[period] = len(period_order)
            rows.append({"day": day, "period": period, "room": room, "presenters": []})

    if not rows and structure.get("days") and structure.get("periods"):
        for day in structure["days"]:
            for period in structure["periods"]:
                for room in range(int(structure["num_rooms"])):
                    rows.append({"day": day, "period": period, "room": room, "presenters": []})

    unknown_days = sorted({row["day"] for row in rows if row["day"] not in day_order})
    next_day = len(day_order)
    for day in unknown_days:
        day_order[day] = next_day
        next_day += 1
    unknown_periods = sorted({row["period"] for row in rows if row["period"] not in period_order})
    next_period = len(period_order)
    for period in unknown_periods:
        period_order[period] = next_period
        next_period += 1

    return sorted(rows, key=lambda row: (day_order.get(row["day"], 0), period_order.get(row["period"], 0), row["room"]))


def _candidate_csv_rows(candidate: list[Any], structure: Dict[str, Any]) -> list[list[str]]:
    if not candidate:
        return []
    _, schedule, _ = candidate
    slots = _build_candidate_slot_rows(schedule or {}, structure)
    out: list[list[str]] = []
    for slot in slots:
        for presenter_name in slot["presenters"]:
            out.append(
                [slot["day"], slot["period"], str(slot["room"] + 1), presenter_name]
            )
    return out


def _build_candidate_slot_grid(
    candidate: list[Any],
    structure: Dict[str, Any],
    presenters_by_name: Dict[str, Any],
) -> dict[str, Any]:
    if not candidate:
        return {
            "days": structure.get("days", []),
            "periods": structure.get("periods", []),
            "rooms": list(range(int(structure.get("num_rooms", 1)))),
            "cells": {},
            "presenter_titles": {},
            "large_room_index": int(structure.get("large_room_index", 0)),
        }

    _, schedule, _ = candidate
    parsed_slots: dict[tuple[str, str, int], list[str]] = {}
    initial_pins: dict[str, dict[str, Any]] = {}
    day_order = list(structure.get("days", []))
    period_order = list(structure.get("periods", []))

    for raw_key, presenters in schedule.items():
        parsed = _parse_schedule_slot(raw_key)
        if parsed is None:
            continue
        period, day, room = parsed
        if day not in day_order:
            day_order.append(day)
        if period not in period_order:
            period_order.append(period)
        slot_presenters = [str(p) for p in (presenters or [])]
        parsed_slots[(day, period, room)] = slot_presenters
        for presenter_name in slot_presenters:
            initial_pins[presenter_name] = {"day": day, "period": period, "room": room}

    if not day_order:
        day_order = sorted({day for day, _, _ in parsed_slots}) if parsed_slots else []
    if not period_order:
        period_order = sorted({period for _, period, _ in parsed_slots}) if parsed_slots else []
    num_rooms = int(structure.get("num_rooms", 0) or 0)

    ordered_cells: dict[str, dict[int, dict[str, list[str]]]] = {}
    for day in day_order:
        day_cells = {}
        for room in range(num_rooms):
            room_cells = {period: parsed_slots.get((day, period, room), []) for period in period_order}
            day_cells[room] = room_cells
        ordered_cells[day] = day_cells

    presenter_titles = {}
    topic_colors: Dict[str, str] = {}
    all_topics = sorted(
        {
            topic.strip().lower()
            for presenter in presenters_by_name.values()
            if presenter is not None
            for topic in getattr(presenter, "topics", [])
            if isinstance(topic, str) and topic.strip()
        }
    )
    for index, topic in enumerate(all_topics):
        hue = int((index * 137.5) % 360)
        topic_colors[topic] = f"hsl({hue}, 78%, 88%)"

    presenter_styles: Dict[str, str] = {}
    presenter_topics: Dict[str, str] = {}
    for name, presenter in presenters_by_name.items():
        if presenter is not None:
            presenter_titles[name] = str(getattr(presenter, "title", "") or "")
            topics = [str(topic).strip() for topic in getattr(presenter, "topics", []) if str(topic).strip()]
            presenter_topics[name] = ", ".join(topics)
            normalized_topics = [topic.lower() for topic in topics]
            style = ""
            if len(normalized_topics) == 1:
                color = topic_colors.get(normalized_topics[0], "#f8f9fa")
                style = f"background: {color};"
            elif len(normalized_topics) >= 2:
                color_a = topic_colors.get(normalized_topics[0], "#e9ecef")
                color_b = topic_colors.get(normalized_topics[1], "#dee2e6")
                style = (
                    f"background: linear-gradient(90deg, {color_a} 0% 50%, {color_b} 50% 100%);"
                )
            else:
                style = "background: #f8f9fa;"
            presenter_styles[name] = style

    return {
        "days": day_order,
        "periods": period_order,
        "rooms": list(range(num_rooms)),
        "cells": ordered_cells,
        "presenter_titles": presenter_titles,
        "presenter_topics": presenter_topics,
        "presenter_styles": presenter_styles,
        "initial_pins": initial_pins,
        "large_room_index": int(structure.get("large_room_index", 0)),
        "presenters_per_room": int(structure.get("presenters_per_room", 0)),
    }


def _build_discarded_ppp_warnings(presenters):
    warnings = []
    for presenter in presenters:
        if presenter.discarded_ppp:
            warnings.append(
                {
                    "name": presenter.name,
                    "count": len(presenter.discarded_ppp),
                    "nominations": ", ".join(presenter.discarded_ppp),
                }
            )
    return warnings


def _records_to_dataframe(records, col_config):
    columns = [col_config[k] for k in ["name", "title", "topics", "availability", "ppp", "bf", "large_room", "present_twice"]]
    normalized = []
    for row in records:
        normalized_row = {col: row.get(col, "") for col in columns}
        normalized.append(normalized_row)
    return pd.DataFrame(normalized)


def _recompute_state_from_records(state):
    records = state.get("records", [])
    df = _records_to_dataframe(records, state["col_config"])
    presenters, detected_days = parse_presenters(df, state["col_config"], state["structure"])
    state["presenters"] = presenters
    state["presenters_by_name"] = {p.name: p for p in presenters}
    if not state["structure"]["days"]:
        state["structure"]["days"] = detected_days
    state["discarded_ppp_warnings"] = _build_discarded_ppp_warnings(presenters)
    state["records"] = df.to_dict(orient="records")


def _safe_build_state(source_type: str, source: str, col_config: Dict[str, str], structure: Dict[str, Any]) -> Dict[str, Any]:
    df = load_dataframe(source)
    presenters, detected_days = parse_presenters(df, col_config, structure)
    if not structure["days"]:
        structure["days"] = detected_days
    records = df.to_dict(orient="records")
    topic_values = _collect_topics(records, col_config)

    return {
        "source_type": source_type,
        "source": source,
        "created_at": time.time(),
        "col_config": col_config,
        "structure": structure,
        "records": records,
        "presenters": presenters,
        "topics": topic_values,
        "days": structure["days"],
        "pins": {},
        "manual_large_room": [],
        "fixed_empty_slots": [],
        "num_restarts": 10,
        "num_results": 5,
        "time_budget_seconds": 180,
        "iterations_per_temp": 12,
        "max_outer_iterations": None,
        "penalties": PENALTIES.copy(),
        "run_in_progress": False,
        "run_progress": [],
        "run_result": None,
        "run_error": None,
        "best_score": None,
        "discarded_ppp_warnings": _build_discarded_ppp_warnings(presenters),
        "presenters_by_name": {p.name: p for p in presenters},
    }


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_optional_int(value: Any, default: Optional[int]) -> Optional[int]:
    if value is None:
        return default
    try:
        text = str(value).strip()
        if not text:
            return default
        return int(text)
    except (TypeError, ValueError):
        return default


def _coerce_penalties(form: Any) -> Dict[str, int]:
    if not isinstance(form, dict):
        return PENALTIES.copy()
    penalties: Dict[str, int] = {}
    for key, default_value in PENALTIES.items():
        penalties[key] = max(0, _coerce_int(form.get(f"penalty_{key}"), default_value))
    return penalties


def _build_empty_grid_slots(structure: Dict[str, Any]) -> list[tuple[str, str, int]]:
    slots: list[tuple[str, str, int]] = []
    periods = [str(period) for period in structure.get("periods", []) if str(period).strip()]
    days = [str(day) for day in structure.get("days", []) if str(day).strip()]
    num_rooms = int(structure.get("num_rooms", 0))
    for day in days:
        for period in periods:
            for room in range(num_rooms):
                slots.append((period, day, room))
    return slots


def _schedule_from_candidate(
    candidate_schedule: Dict[str, list[str]], structure: Dict[str, Any]
) -> Dict[tuple[str, str, int], list[str]]:
    grid = {slot: [] for slot in _build_empty_grid_slots(structure)}
    if not isinstance(candidate_schedule, dict):
        return grid

    for raw_slot, presenter_names in candidate_schedule.items():
        parsed = _parse_schedule_slot(raw_slot)
        if parsed is None:
            continue
        grid[parsed] = [name for name in (presenter_names or []) if isinstance(name, str)]
    return grid


def _candidate_has_overflow(schedule: Dict[str, Any], structure: Dict[str, Any]) -> bool:
    cap = int(structure.get("presenters_per_room", 0))
    if cap <= 0:
        return False
    for presenters in schedule.values():
        if isinstance(presenters, list) and len(presenters) > cap:
            return True
    return False


def _collect_schedule_feasibility_violations(
    schedule: Dict[tuple[str, str, int], list[Any]],
    state: Dict[str, Any],
) -> list[str]:
    structure = state.get("structure", {})
    periods = [str(period) for period in structure.get("periods", [])]
    days = [str(day) for day in structure.get("days", [])]
    num_rooms = int(structure.get("num_rooms", 0))
    capacity = int(structure.get("presenters_per_room", 0))
    large_room_index = int(structure.get("large_room_index", 0))
    fixed_empty = _parse_fixed_empty_slots(state.get("fixed_empty_slots", []), structure, days)

    valid_periods = set(periods)
    valid_days = set(days)
    known_presenters = state.get("presenters_by_name", {})
    present_in_slots: set[str] = set()
    violations: list[str] = []

    for slot, presenters in schedule.items():
        period, day, room = slot
        if period not in valid_periods or day not in valid_days:
            violations.append(f"Invalid slot {period}|{day}|{room} in drag result.")
            continue
        if room < 0 or room >= num_rooms:
            violations.append(f"Invalid room index {room} in slot {period}|{day}.")
            continue
        if slot in fixed_empty:
            violations.append(f"Slot {period}|{day}|{room} is fixed empty.")
            continue
        if capacity > 0 and len(presenters) > capacity:
            violations.append(
                f"Slot {period}|{day}|Room {room + 1} has {len(presenters)} presenters, above capacity {capacity}."
            )

        for presenter in presenters:
            name = getattr(presenter, "name", "")
            if not name:
                continue
            if name in present_in_slots:
                violations.append(f"Presenter {name} appears more than once.")
            else:
                present_in_slots.add(name)

            if (period, day) not in getattr(presenter, "_availability_set", set()):
                violations.append(f"{name} is unavailable in {period}|{day}.")
            if room == large_room_index and str(getattr(presenter, "large_room", "")) == "No":
                violations.append(f"{name} cannot be assigned to the large room.")
            if presenter.pinned_slot is not None and (period, day) != presenter.pinned_slot:
                violations.append(f"{name} is pinned to a different period/day.")
            if presenter.pinned_room is not None and room != presenter.pinned_room:
                violations.append(f"{name} is pinned to a different room.")

    missing_presenters = set(known_presenters.keys()) - present_in_slots
    if missing_presenters:
        missing_preview = ", ".join(sorted(missing_presenters))
        if len(missing_presenters) > 8:
            missing_preview += ", ..."
        violations.append(f"Missing presenters in schedule: {missing_preview}.")

    return sorted(set(violations))


def _launch_generation(workflow_id: str):
    def _runner():
        with WORKFLOW_LOCK:
            state = WORKFLOW_STATES.get(workflow_id)
            if not state:
                return
            state["run_in_progress"] = True
            state["run_progress"] = []
            state["run_error"] = None

        def _progress(run_number, total_runs, score):
            with WORKFLOW_LOCK:
                state_ref = WORKFLOW_STATES.get(workflow_id)
                if not state_ref:
                    return
                if run_number is None:
                    return
                state_ref["run_progress"].append(
                    {"run_number": run_number, "total": total_runs, "best_score": score}
                )
                if score is not None and (
                    state_ref.get("best_score") is None or score < state_ref.get("best_score", float("inf"))
                ):
                    state_ref["best_score"] = score

        try:
            with WORKFLOW_LOCK:
                state = WORKFLOW_STATES.get(workflow_id)
                if not state:
                    return
                source = state["source"]
                col_config = state["col_config"]
                structure = dict(state["structure"])
                pins = state.get("pins", {})
                fixed_empty_slots = state.get("fixed_empty_slots", [])
                num_restarts = state.get("num_restarts", 10)
                num_results = state.get("num_results", 5)
                manual_large_room = state.get("manual_large_room", [])
                penalties = state.get("penalties", PENALTIES)
                time_budget_seconds = state.get("time_budget_seconds")
                iterations_per_temp = state.get("iterations_per_temp", 12)
                max_outer_iterations = state.get("max_outer_iterations")

            result = run(
                source,
                col_config=col_config,
                structure=structure,
                pins=pins,
                penalties=penalties,
                num_restarts=num_restarts,
                num_results=num_results,
                fixed_empty_slots=fixed_empty_slots,
                manual_large_room=manual_large_room,
                iterations_per_temp=iterations_per_temp,
                max_outer_iterations=max_outer_iterations,
                time_budget_seconds=time_budget_seconds,
                progress_callback=_progress,
            )

            with WORKFLOW_LOCK:
                if workflow_id in WORKFLOW_STATES:
                    WORKFLOW_STATES[workflow_id]["run_result"] = result
        except Exception as exc:  # pragma: no cover - defensive UI path
            with WORKFLOW_LOCK:
                if workflow_id in WORKFLOW_STATES:
                    WORKFLOW_STATES[workflow_id]["run_error"] = str(exc)
        finally:
            with WORKFLOW_LOCK:
                if workflow_id in WORKFLOW_STATES:
                    WORKFLOW_STATES[workflow_id]["run_in_progress"] = False

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "upload.html",
        col_config=DEFAULT_COL_CONFIG,
        structure=DEFAULT_STRUCTURE,
        PENALTIES=PENALTIES,
    )


@app.route("/start", methods=["POST"])
def start_workflow():
    csv_file = request.files.get("csv_file")
    csv_url = (request.form.get("csv_url") or "").strip()
    if not csv_file or not csv_file.filename:
        if not csv_url:
            return (
                render_template(
                    "upload.html",
                    error="Please upload a CSV file or provide a Google Sheets public CSV link.",
                    col_config=DEFAULT_COL_CONFIG,
                    structure=DEFAULT_STRUCTURE,
                    PENALTIES=PENALTIES,
                ),
                400,
            )
        source = csv_url
        source_type = "url"
    else:
        if not _is_allowed_upload(csv_file.filename):
            return (
                render_template(
                    "upload.html",
                    error="Only CSV files are supported for upload.",
                    col_config=DEFAULT_COL_CONFIG,
                    structure=DEFAULT_STRUCTURE,
                    PENALTIES=PENALTIES,
                ),
                400,
            )
        workflow_file = Path("/tmp") / f"symposium_upload_{os.getpid()}_{uuid.uuid4().hex}.csv"
        csv_file.save(workflow_file)
        source = str(workflow_file)
        source_type = "file"

    periods = _split_csv_lines(request.form.get("periods")) or DEFAULT_STRUCTURE["periods"]
    days = _split_csv_lines(request.form.get("days"))
    structure = {
        "num_rooms": _coerce_int(request.form.get("num_rooms"), DEFAULT_STRUCTURE["num_rooms"]),
        "large_room_index": _coerce_int(request.form.get("large_room_index"), DEFAULT_STRUCTURE["large_room_index"]),
        "presenters_per_room": _coerce_int(
            request.form.get("presenters_per_room"), DEFAULT_STRUCTURE["presenters_per_room"]
        ),
        "periods": periods,
        "days": days,
        "large_room_default": request.form.get("large_room_default") or DEFAULT_STRUCTURE["large_room_default"],
    }

    col_config = {k: request.form.get(f"col_{k}") or v for k, v in DEFAULT_COL_CONFIG.items()}

    try:
        state = _safe_build_state(source_type, source, col_config, structure)
    except Exception as exc:
        return (
            render_template(
                "upload.html",
                error=str(exc),
                col_config=col_config,
                structure=structure,
                PENALTIES=PENALTIES,
            ),
            400,
        )

    _prune_workflow_states()
    workflow_id = uuid.uuid4().hex
    with WORKFLOW_LOCK:
        state["num_restarts"] = _coerce_int(request.form.get("num_restarts"), state.get("num_restarts", 10))
        state["num_results"] = _coerce_int(request.form.get("num_results"), state.get("num_results", 5))
        state["time_budget_seconds"] = _coerce_optional_int(request.form.get("time_budget_seconds"), 180)
        if state["time_budget_seconds"] is not None and state["time_budget_seconds"] <= 0:
            state["time_budget_seconds"] = None
        state["iterations_per_temp"] = _coerce_int(
            request.form.get("iterations_per_temp"), state.get("iterations_per_temp", 12)
        )
        max_outer_iterations = _coerce_optional_int(request.form.get("max_outer_iterations"), state.get("max_outer_iterations"))
        if max_outer_iterations is not None and max_outer_iterations <= 0:
            max_outer_iterations = None
        state["max_outer_iterations"] = max_outer_iterations
        state["penalties"] = _coerce_penalties(request.form)
        WORKFLOW_STATES[workflow_id] = state
    session["workflow_id"] = workflow_id
    return redirect(url_for("responses_view"))


@app.route("/responses", methods=["GET"])
def responses_view():
    workflow_id, state = _with_state()
    if not state:
        return redirect(url_for("index"))
    return render_template(
        "responses.html",
        col_config=state["col_config"],
        presenters=state["records"],
        structure=state["structure"],
        num_presenters=len(state["records"]),
        present_twice_presenters=[
            str(row.get(state["col_config"]["name"], "")).strip()
            for row in state["records"]
            if str(row.get(state["col_config"]["present_twice"], "")).strip().lower() in {"yes", "y", "true", "1"}
        ],
        discarded_ppp_warnings=state.get("discarded_ppp_warnings", []),
    )


@app.route("/responses/edit/<int:index>", methods=["GET", "POST"])
def responses_edit(index: int):
    workflow_id, state = _with_state()
    if not state:
        return redirect(url_for("index"))

    rows = state.get("records") or []
    if index < 0 or index >= len(rows):
        return redirect(url_for("responses_view"))

    row = rows[index]
    if request.method == "POST":
        for key in ["name", "title", "topics", "availability", "ppp", "bf", "large_room", "present_twice"]:
            col = state["col_config"][key]
            row[col] = (request.form.get(col, "") or "").strip()
        _recompute_state_from_records(state)
        return redirect(url_for("responses_view"))

    return render_template("response_edit.html", row=row, col_config=state["col_config"], index=index)


@app.route("/setup", methods=["GET", "POST"])
def setup():
    workflow_id, state = _with_state()
    if not state:
        return redirect(url_for("index"))

    if request.method == "POST":
        manual_large = request.form.getlist("manual_large_room")
        state["manual_large_room"] = manual_large

        _launch_generation(workflow_id)
        return redirect(url_for("generating_view"))

    presenter_rows = []
    for row in state["records"]:
        name = str(row.get(state["col_config"]["name"], "")).strip()
        if not name:
            continue
        large_room = str(row.get(state["col_config"]["large_room"], "")).strip().lower()
        if large_room not in {"yes", "maybe"}:
            continue
        presenter_rows.append(
            {
                "name": name,
                "large_room": str(row.get(state["col_config"]["large_room"], "")).strip() or "Maybe",
                "topics": str(row.get(state["col_config"]["topics"], "")).strip(),
            }
        )

    return render_template(
        "setup.html",
        structure=state["structure"],
        presenter_rows=presenter_rows,
        num_restarts=state["num_restarts"],
        num_results=state["num_results"],
    )


@app.route("/generating", methods=["GET"])
def generating_view():
    _, state = _with_state()
    if not state:
        return redirect(url_for("index"))

    return render_template(
        "generating.html",
        num_restarts=state.get("num_restarts", 10),
        num_results=state.get("num_results", 5),
        time_budget_seconds=state.get("time_budget_seconds"),
    )


@app.route("/api/run_status", methods=["GET"])
def api_run_status():
    workflow_id, state = _with_state()
    if not state:
        return jsonify({"error": "No workflow loaded"}), 400

    return jsonify(
        {
            "run_in_progress": bool(state.get("run_in_progress")),
            "run_error": state.get("run_error"),
            "run_progress": state.get("run_progress", [])[-50:],
            "best_score": state.get("best_score"),
            "done": bool(state.get("run_result") is not None and not state.get("run_in_progress")),
        }
    )


@app.route("/review", methods=["GET"])
def review_view():
    workflow_id, state = _with_state()
    if not state:
        return redirect(url_for("index"))
    if not state.get("run_result"):
        return redirect(url_for("generating_view"))

    result = state["run_result"]
    candidates = result.get("results", [])
    if not candidates:
        return render_template("review.html", result=result, workflow_id=workflow_id)

    selected_candidate_index = _coerce_int(request.args.get("candidate"), 0)
    if selected_candidate_index < 0 or selected_candidate_index >= len(candidates):
        selected_candidate_index = 0

    selected_candidate = candidates[selected_candidate_index]
    selected_schedule = _build_candidate_slot_grid(
        selected_candidate,
        state["structure"],
        state.get("presenters_by_name", {}),
    )
    return render_template(
        "review.html",
        result=result,
        workflow_id=workflow_id,
        candidates=candidates,
        selected_candidate_index=selected_candidate_index,
        selected_candidate=selected_candidate,
        selected_schedule=selected_schedule,
        has_overflow=_candidate_has_overflow(selected_candidate[1], state["structure"]),
        export_error=request.args.get("export_error"),
        constraint_error=request.args.get("constraint_error"),
    )


@app.route("/review/apply_constraints", methods=["POST"])
def review_apply_constraints():
    workflow_id, state = _with_state()
    if not state:
        return redirect(url_for("index"))

    payload = request.get_json(silent=True)
    if isinstance(payload, dict):
        raw_pins = payload.get("pins")
        selected_candidate_index = _coerce_int(payload.get("candidate_index"), 0)
    else:
        raw_pins = request.form.get("pins_json", "{}")
        selected_candidate_index = _coerce_int(request.form.get("candidate_index"), 0)
    parsed = _normalize_json(raw_pins, {}) if not isinstance(raw_pins, dict) else raw_pins
    if not isinstance(parsed, dict):
        parsed = {}
    valid_names = set((state.get("presenters_by_name") or {}).keys())
    normalized: dict[str, dict[str, Any]] = {}
    for raw_name, raw_pin in parsed.items():
        name = str(raw_name).strip()
        if name not in valid_names:
            continue
        if not isinstance(raw_pin, dict):
            continue
        period = raw_pin.get("period")
        day = raw_pin.get("day")
        room = raw_pin.get("room")
        if period is None or day is None or room is None:
            continue
        try:
            room_i = int(room)
        except (TypeError, ValueError):
            continue
        normalized[name] = {"period": str(period), "day": str(day), "room": room_i}

    result = state.get("run_result")
    if not result:
        return redirect(url_for("review_view"))
    candidates = result.get("results", [])
    if not candidates:
        return redirect(url_for("review_view"))
    if selected_candidate_index < 0 or selected_candidate_index >= len(candidates):
        selected_candidate_index = 0

    candidate = candidates[selected_candidate_index]
    if not isinstance(candidate, (list, tuple)) or len(candidate) < 3:
        return redirect(url_for("review_view"))
    _, candidate_schedule, _ = candidate
    if not isinstance(candidate_schedule, dict):
        return redirect(url_for("review_view"))

    base_grid = _schedule_from_candidate(candidate_schedule, state["structure"])
    grid: Dict[tuple[str, str, int], list[Any]] = {}
    presenter_by_name = state.get("presenters_by_name", {})
    for slot, names in base_grid.items():
        occupants: list[Any] = []
        for name in names:
            presenter = presenter_by_name.get(name)
            if presenter is None:
                continue
            occupants.append(presenter)
        grid[slot] = occupants

    for name, target in normalized.items():
        presenter = presenter_by_name.get(name)
        if presenter is None:
            continue
        for occupants in grid.values():
            occupants[:] = [p for p in occupants if p.name != name]
        slot_key = (target["period"], target["day"], target["room"])
        grid.setdefault(slot_key, [])
        grid[slot_key].append(presenter)

    violations = _collect_schedule_feasibility_violations(grid, state)
    if violations:
        error = "Constraint violations prevent recompute: " + "; ".join(violations[:8])
        if len(violations) > 8:
            error += "; ..."
        return redirect(url_for("review_view", candidate=selected_candidate_index, constraint_error=error))
    state["pins"] = normalized

    serialized = {}
    for slot, presenters in grid.items():
        if not presenters:
            continue
        serialized[f"{slot[0]}|{slot[1]}|{slot[2]}"] = [p.name for p in presenters]

    score, breakdown = _score_schedule(
        {slot: list(presenters) for slot, presenters in grid.items()},
        state["structure"],
        state.get("penalties", PENALTIES),
        presenters=state.get("presenters"),
        assignment={p.name: slot for slot, presenters in grid.items() for p in presenters},
    )
    candidates[selected_candidate_index] = [int(score), serialized, breakdown]
    return redirect(url_for("review_view", candidate=selected_candidate_index))


@app.route("/review/export/<int:index>", methods=["GET"])
def review_export(index: int):
    workflow_id, state = _with_state()
    if not state:
        return redirect(url_for("index"))
    if not state.get("run_result"):
        return redirect(url_for("generating_view"))

    result = state["run_result"]
    candidates = result.get("results", [])
    if index < 0 or index >= len(candidates):
        return redirect(url_for("review_view"))

    format_type = (request.args.get("format") or "csv").lower()
    candidate = candidates[index]
    score = candidate[0]
    schedule = candidate[1]
    breakdown = candidate[2]

    if _candidate_has_overflow(schedule, state["structure"]):
        return redirect(url_for("review_view", candidate=index, export_error="Resolve overflow slots before exporting."))

    if format_type == "json":
        payload = {
            "workflow_id": workflow_id,
            "candidate_index": index,
            "score": score,
            "breakdown": breakdown,
            "schedule": schedule,
            "structure": state["structure"],
            "hard_conflicts": result.get("hard_conflicts", []),
            "warnings": result.get("warnings", []),
        }
        return jsonify(payload)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Day", "Period", "Room", "Presenter"])
    rows = _candidate_csv_rows(candidate, state["structure"])
    if not rows:
        for slot in _build_candidate_slot_rows(schedule, state["structure"]):
            writer.writerow([slot["day"], slot["period"], slot["room"] + 1, ""])
    else:
        for row in rows:
            writer.writerow(row)

    output.seek(0)
    response = Response(output.getvalue(), mimetype="text/csv")
    response.headers["Content-Disposition"] = f'attachment; filename="schedule_candidate_{index + 1}.csv"'
    return response


@app.route("/api/run", methods=["POST"])
def api_run():
    payload = request.get_json(silent=True) or {}
    source = payload.get("url_or_path")
    if not source:
        return jsonify({"error": "url_or_path is required"}), 400

    result = run(
        source,
        col_config=payload.get("col_config", DEFAULT_COL_CONFIG),
        structure=payload.get("structure", DEFAULT_STRUCTURE),
        pins=payload.get("pins", {}),
        penalties=payload.get("penalties", PENALTIES),
        num_restarts=int(payload.get("num_restarts", 20)),
        num_results=int(payload.get("num_results", 10)),
        iterations_per_temp=int(payload.get("iterations_per_temp", 12)),
        max_outer_iterations=_coerce_optional_int(payload.get("max_outer_iterations"), None),
        time_budget_seconds=_coerce_optional_int(payload.get("time_budget_seconds"), None),
        fixed_empty_slots=payload.get("fixed_empty_slots", []),
        manual_large_room=payload.get("manual_large_room", []),
    )
    return jsonify(
        {
            "hard_conflicts": result["hard_conflicts"],
            "warnings": result["warnings"],
            "unavoidable_minimum": result["unavoidable_minimum"],
            "results": result["results"],
            "num_presenters": len(result["presenters"]),
        }
    )


    if __name__ == "__main__":
        debug_mode = os.getenv("FLASK_DEBUG", "0").lower() in {"1", "true", "yes"}
        app.run(
            host="0.0.0.0",
            port=int(os.getenv("PORT", "5000")),
            debug=debug_mode,
    )
