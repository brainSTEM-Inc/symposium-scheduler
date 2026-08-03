from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import pandas as pd

from scheduler import (
    DEFAULT_COL_CONFIG,
    DEFAULT_STRUCTURE,
    PENALTIES,
    _extract_name_candidates,
    _parse_availability,
    _parse_fixed_empty_slots,
    run,
)


def test_parse_availability():
    periods = ["PD 2", "PD 3", "PD 4"]
    raw = "PD 2, Tuesday, December 15th, PD 3, Thursday, December 17th"
    assert _parse_availability(raw, periods) == [
        ("PD 2", "Tuesday, December 15th"),
        ("PD 3", "Thursday, December 17th"),
    ]


def test_extract_name_candidates():
    valid = {"saran, akash", "xu, katherine", "zhou, anna"}
    assert _extract_name_candidates("Saran, Akash, Xu, Katherine", valid) == [
        "Saran, Akash",
        "Xu, Katherine",
    ]
    assert _extract_name_candidates("Unknown, Name", valid) == []


def test_parse_fixed_empty_slots_tolerates_dict_or_tuple():
    structure = {"periods": ["PD 2", "PD 3"], "num_rooms": 3}
    days = ["Tuesday, December 15th", "Thursday, December 17th"]
    parsed = _parse_fixed_empty_slots(
        [
            {"period": "PD 2", "day": "Tuesday, December 15th", "room": 1},
            ["PD 3", "Tuesday, December 15th", "2"],
        ],
        structure,
        days,
    )
    assert ("PD 2", "Tuesday, December 15th", 1) in parsed
    assert ("PD 3", "Tuesday, December 15th", 2) in parsed
    assert ("PD 2", "Thursday, December 17th", 1) not in parsed


def test_run_basic_smoke():
    # Build a tiny synthetic dataset.
    rows = [
        [
            "A, B",
            "Project 1",
            "Biology",
            "PD 2, Tuesday, December 15th",
            "C, D",
            "E, F",
            "Maybe",
            "No",
        ],
        [
            "C, D",
            "Project 2",
            "Biology",
            "PD 2, Tuesday, December 15th",
            "A, B",
            "E, F",
            "Yes",
            "No",
        ],
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            DEFAULT_COL_CONFIG["name"],
            DEFAULT_COL_CONFIG["title"],
            DEFAULT_COL_CONFIG["topics"],
            DEFAULT_COL_CONFIG["availability"],
            DEFAULT_COL_CONFIG["ppp"],
            DEFAULT_COL_CONFIG["bf"],
            DEFAULT_COL_CONFIG["large_room"],
            DEFAULT_COL_CONFIG["present_twice"],
        ],
    )

    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "presenters.csv"
        df.to_csv(csv_path, index=False)
        result = run(
            str(csv_path),
            col_config=DEFAULT_COL_CONFIG,
            structure=DEFAULT_STRUCTURE.copy(),
            penalties=PENALTIES,
            num_restarts=2,
            num_results=1,
        )

    assert "presenters" in result
        assert len(result["presenters"]) == 2
        assert isinstance(result["hard_conflicts"], list)


def test_fixed_empty_slot_blocks_assignment():
    rows = [
        [
            "A, B",
            "Project 1",
            "Biology",
            "PD 2, Tuesday, December 15th",
            "",
            "",
            "Maybe",
            "No",
        ]
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            DEFAULT_COL_CONFIG["name"],
            DEFAULT_COL_CONFIG["title"],
            DEFAULT_COL_CONFIG["topics"],
            DEFAULT_COL_CONFIG["availability"],
            DEFAULT_COL_CONFIG["ppp"],
            DEFAULT_COL_CONFIG["bf"],
            DEFAULT_COL_CONFIG["large_room"],
            DEFAULT_COL_CONFIG["present_twice"],
        ],
    )

    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "presenters.csv"
        df.to_csv(csv_path, index=False)
        result = run(
            str(csv_path),
            col_config=DEFAULT_COL_CONFIG,
            structure={
                **DEFAULT_STRUCTURE,
                "num_rooms": 1,
                "presenters_per_room": 1,
                "periods": ["PD 2"],
                "days": ["Tuesday, December 15th"],
            },
            penalties=PENALTIES,
            num_restarts=1,
            num_results=1,
            fixed_empty_slots=[["PD 2", "Tuesday, December 15th", 0]],
        )

    assert result["hard_conflicts"], "Expected hard conflict when all slots are fixed-empty"
    assert "no available timeslots" in result["hard_conflicts"][0]["message"].lower()
