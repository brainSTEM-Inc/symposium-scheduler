from __future__ import annotations

import math
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

SlotKey = Tuple[str, str, int]

PENALTIES: Dict[str, int] = {
    "topic_mismatch_room": 100,
    "mutual_ppp_not_paired": 40,
    "same_topic_across_rooms": 30,
    "mutual_bf_same_timeslot": 30,
    "one_way_ppp_not_paired": 15,
    "one_way_bf_same_timeslot": 10,
    "empty_slot_not_clustered": 10,
    "large_room_missed": 10,
}

DEFAULT_COL_CONFIG: Dict[str, str] = {
    "name": "Name",
    "title": "Project Title",
    "topics": "Project Topic (please select up to two)",
    "availability": "Availability",
    "ppp": "Preferred co-presenter(s)",
    "bf": "Best friend(s)",
    "large_room": "Interested in presenting in the SMCS Hub?",
    "present_twice": "Interested in presenting twice?",
}

DEFAULT_STRUCTURE: Dict[str, Any] = {
    "num_rooms": 3,
    "large_room_index": 0,
    "presenters_per_room": 2,
    "periods": ["PD 2", "PD 3", "PD 4", "PD 5", "PD 6"],
    "days": [],
    "large_room_default": "Maybe",
}


@dataclass
class Presenter:
    name: str
    title: str
    topics: List[str]
    availability: List[Tuple[str, str]]
    ppp_names: List[str]
    bf_names: List[str]
    large_room: str
    present_twice: bool
    ppp: List["Presenter"] = field(default_factory=list)
    bf: List["Presenter"] = field(default_factory=list)
    mutual_ppp: List["Presenter"] = field(default_factory=list)
    mutual_bf: List["Presenter"] = field(default_factory=list)
    discarded_ppp: List[str] = field(default_factory=list)
    pinned_slot: Optional[Tuple[str, str]] = None
    pinned_room: Optional[int] = None
    _topic_keys: set[str] = field(default_factory=set, init=False, repr=False)
    _availability_set: set[Tuple[str, str]] = field(default_factory=set, init=False, repr=False)
    _ppp_set: set["Presenter"] = field(default_factory=set, init=False, repr=False)
    _bf_set: set["Presenter"] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        self._topic_keys = {str(topic).strip().lower() for topic in self.topics if str(topic).strip()}
        self._availability_set = set(self.availability)

    def shares_topic(self, other: "Presenter") -> bool:
        return bool(self._topic_keys & other._topic_keys)

    def topic_affinity(self, other: "Presenter") -> int:
        return len(self._topic_keys & other._topic_keys)

    def __hash__(self) -> int:
        return hash(self.name.lower())


def _normalize_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _normalize_bool(value: Any, default: bool = False) -> bool:
    text = _normalize_text(value).lower()
    if text in {"yes", "y", "true", "1"}:
        return True
    if text in {"no", "n", "false", "0"}:
        return False
    return default


def _parse_large_room(value: Any, default: str) -> str:
    text = _normalize_text(value).strip()
    if not text:
        return default
    normalized = text.strip().lower()
    if normalized in {"yes", "y", "true"}:
        return "Yes"
    if normalized in {"no", "n", "false"}:
        return "No"
    if normalized in {"maybe", "m"}:
        return "Maybe"
    return default


def _parse_topics(raw: Any) -> List[str]:
    text = _normalize_text(raw)
    if not text:
        return []
    values = []
    for part in text.split(","):
        v = part.strip()
        if not v:
            continue
        values.append(v)
        if len(values) == 2:
            break
    # Deduplicate while preserving order.
    deduped: List[str] = []
    seen = set()
    for topic in values:
        key = topic.lower()
        if key in seen:
            continue
        deduped.append(topic)
        seen.add(key)
    return deduped


def _parse_availability(raw: Any, periods: Sequence[str]) -> List[Tuple[str, str]]:
    period_set = set(p.strip() for p in periods)
    text = _normalize_text(raw)
    if not text:
        return []
    tokens = [token.strip() for token in text.split(",")]
    availability: List[Tuple[str, str]] = []
    current_period = None
    current_day_parts: List[str] = []

    def flush_pair() -> None:
        nonlocal current_period, current_day_parts
        if current_period is None:
            return
        if current_day_parts:
            day = ", ".join(current_day_parts).strip()
            if current_period and day:
                availability.append((current_period, day))
            current_day_parts = []

    for token in tokens:
        if token in period_set:
            flush_pair()
            current_period = token
            continue
        if current_period is None:
            continue
        current_day_parts.append(token)

    flush_pair()

    # Remove duplicate availability entries while preserving order.
    deduped: List[Tuple[str, str]] = []
    seen_pairs = set()
    for pair in availability:
        if pair in seen_pairs:
            continue
        deduped.append(pair)
        seen_pairs.add(pair)
    return deduped


def _split_name_tokens(raw: Any) -> List[str]:
    text = _normalize_text(raw)
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def _extract_name_candidates(raw: Any, valid_names: Iterable[str]) -> List[str]:
    """
    Parse "Last, First" values while preserving the comma-based name format.
    """
    parts = _split_name_tokens(raw)
    if not parts:
        return []

    normalized_valid = {name.lower(): name for name in valid_names}
    used = set()
    out: List[str] = []
    i = 0
    while i < len(parts):
        if i + 1 < len(parts):
            candidate = f"{parts[i]}, {parts[i+1]}"
            key = candidate.lower()
            if key in used:
                i += 2
                continue
            if key in normalized_valid and candidate not in out:
                out.append(candidate)
                used.add(key)
                i += 2
                continue
        plain = parts[i].strip()
        key_plain = plain.lower()
        if key_plain in normalized_valid and plain not in out:
            out.append(plain)
            used.add(key_plain)
        i += 1
    return out


def load_dataframe(url_or_path: str) -> pd.DataFrame:
    if _normalize_text(url_or_path).lower().startswith("http"):
        return pd.read_csv(url_or_path)
    return pd.read_csv(url_or_path)


def _normalize_days(days: Iterable[str]) -> List[str]:
    deduped: List[str] = []
    seen = set()
    for day in days:
        key = day.strip()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped


def parse_presenters(df: pd.DataFrame, col_config: Dict[str, str], structure: Dict[str, Any]):
    required = ("name", "title", "topics", "availability", "ppp", "bf", "large_room", "present_twice")
    missing = [key for key in required if col_config.get(key) not in df.columns]
    if missing:
        raise KeyError(
            "Missing required columns: "
            + ", ".join(col_config[key] for key in missing)
        )

    periods = structure["periods"]
    large_default = structure.get("large_room_default", "Maybe")
    presenters: List[Presenter] = []
    detected_days: List[str] = []
    by_name: Dict[str, Presenter] = {}
    for _, row in df.iterrows():
        name = _normalize_text(row[col_config["name"]])
        if not name or name.lower() == "nan":
            continue
        title = _normalize_text(row[col_config["title"]])
        topics = _parse_topics(row[col_config["topics"]])
        availability = _parse_availability(row[col_config["availability"]], periods)
        for _, day in availability:
            detected_days.extend(_normalize_days([day]))
        presenter = Presenter(
            name=name,
            title=title,
            topics=topics,
            availability=availability,
            ppp_names=[],
            bf_names=[],
            large_room=_parse_large_room(row[col_config["large_room"]], large_default),
            present_twice=_normalize_bool(row[col_config["present_twice"]], default=False),
        )
        presenters.append(presenter)
        by_name[name.strip().lower()] = presenter

    _hydrate_nomination_fields(presenters, df, col_config, by_name)

    for presenter in presenters:
        # Replace PPP/BF names with Presenter references.
        target = by_name.get(presenter.name.lower())
        if target is None:
            continue
        for name in presenter.ppp_names:
            resolved = by_name.get(name.lower())
            if resolved is None or resolved is target:
                continue
            if resolved in presenter.ppp:
                continue
            if presenter.shares_topic(resolved):
                presenter.ppp.append(resolved)
            else:
                presenter.discarded_ppp.append(name)
        for name in presenter.bf_names:
            resolved = by_name.get(name.lower())
            if resolved is None or resolved is target:
                continue
            if resolved in presenter.bf:
                continue
            presenter.bf.append(resolved)

    # Build mutual relation lists.
    for presenter in presenters:
        ppp_mutual: List[Presenter] = []
        bf_mutual: List[Presenter] = []
        for other in presenter.ppp:
            if presenter in other.ppp:
                ppp_mutual.append(other)
        for other in presenter.bf:
            if presenter in other.bf:
                bf_mutual.append(other)
        presenter.mutual_ppp = ppp_mutual
        presenter.mutual_bf = bf_mutual
        presenter._ppp_set = set(presenter.ppp)
        presenter._bf_set = set(presenter.bf)

    return presenters, _normalize_days(detected_days)


def _apply_manual_large_room_overrides(presenters: List[Presenter], manual_large_room: Optional[Iterable[str]]) -> None:
    if not manual_large_room:
        return
    by_name = {p.name.lower(): p for p in presenters}
    for raw_name in manual_large_room:
        name = _normalize_text(raw_name).lower()
        presenter = by_name.get(name)
        if presenter is None:
            continue
        if presenter.large_room == "No":
            continue
        presenter.large_room = "Yes"


def _hydrate_nomination_fields(
    presenters: List[Presenter], df: pd.DataFrame, col_config: Dict[str, str], by_name: Dict[str, Presenter]
) -> None:
    normalized_names = list(by_name.keys())
    for presenter in presenters:
        row_mask = df[col_config["name"]].astype(str).str.strip().str.lower() == presenter.name.lower()
        ppp_raw = df.loc[row_mask, col_config["ppp"]].iloc[0] if row_mask.any() else ""
        bf_raw = df.loc[row_mask, col_config["bf"]].iloc[0] if row_mask.any() else ""
        presenter.ppp_names = _extract_name_candidates(ppp_raw, normalized_names)
        presenter.bf_names = _extract_name_candidates(bf_raw, normalized_names)


def _load_slots(structure: Dict[str, Any], days: Sequence[str]) -> List[SlotKey]:
    periods = structure["periods"]
    num_rooms = int(structure["num_rooms"])
    slots: List[SlotKey] = []
    for day in days:
        for period in periods:
            for room in range(num_rooms):
                slots.append((period, day, room))
    return slots


def _parse_pins(pins: Any) -> Dict[str, Dict[str, Any]]:
    if not pins:
        return {}
    if isinstance(pins, dict):
        return {str(k): v for k, v in pins.items() if isinstance(v, dict)}
    return {}


def _parse_fixed_empty_slots(raw: Any, structure: Dict[str, Any], days: Sequence[str]) -> set[SlotKey]:
    slots: set[SlotKey] = set()
    if not raw:
        return slots
    if not isinstance(raw, Iterable):
        return slots
    for value in raw:
        if isinstance(value, dict):
            period = value.get("period")
            day = value.get("day")
            room = value.get("room", value.get("room_index"))
        elif isinstance(value, Sequence) and not isinstance(value, str):
            if len(value) < 3:
                continue
            period, day, room = value[0], value[1], value[2]
        else:
            continue
        try:
            room_i = int(room)
        except (TypeError, ValueError):
            continue
        if str(period).strip() in structure["periods"] and str(day).strip() in days and 0 <= room_i < int(structure["num_rooms"]):
            slots.add((str(period).strip(), str(day).strip(), room_i))
    return slots


def _parse_room_unavailable_slots(
    raw: Any,
    structure: Dict[str, Any],
    days: Sequence[str],
) -> set[SlotKey]:
    if raw is None:
        return set()
    if not isinstance(raw, Iterable) or isinstance(raw, str):
        return set()
    periods = {str(period).strip() for period in structure.get("periods", [])}
    valid_days = {str(day).strip() for day in days}
    num_rooms = int(structure.get("num_rooms", 0) or 0)
    unavailable: set[SlotKey] = set()
    for item in raw:
        if isinstance(item, str):
            parts = [part.strip() for part in item.split("|")]
            if len(parts) != 3:
                continue
            period, day, room_raw = parts
        elif isinstance(item, (list, tuple)) and len(item) == 3:
            period = str(item[0]).strip()
            day = str(item[1]).strip()
            room_raw = item[2]
        else:
            continue
        if period not in periods or day not in valid_days:
            continue
        try:
            room = int(room_raw)
        except (TypeError, ValueError):
            continue
        if not (0 <= room < num_rooms):
            continue
        unavailable.add((period, day, room))
    return unavailable


def _coerce_pin_to_slot(
    pin: Dict[str, Any],
    periods: Sequence[str],
    days: Sequence[str],
    num_rooms: int,
) -> Dict[str, Any]:
    if not isinstance(pin, dict):
        return {}
    period = pin.get("period")
    day = pin.get("day")
    room = pin.get("room", pin.get("room_index"))
    if period is None or day is None:
        return {}
    if str(period).strip() not in periods or str(day).strip() not in days:
        return {}
    try:
        room_i = None if room is None else int(room)
    except (TypeError, ValueError):
        room_i = None
    if room_i is not None and not (0 <= room_i < num_rooms):
        room_i = None
    return {"period": str(period).strip(), "day": str(day).strip(), "room": room_i}


def _build_presenter_pin_lookup(
    presenters: List[Presenter], pins: Dict[str, Dict[str, Any]], structure: Dict[str, Any], days: Sequence[str]
) -> Tuple[Dict[Presenter, Dict[str, Any]], List[dict]]:
    by_name = {p.name.lower(): p for p in presenters}
    pinned: Dict[Presenter, Dict[str, Any]] = {}
    unknown: List[dict] = []
    periods = structure["periods"]
    num_rooms = int(structure["num_rooms"])
    for raw_name, raw_pin in pins.items():
        presenter = by_name.get(_normalize_text(raw_name).lower())
        if presenter is None:
            unknown.append({"name": raw_name})
            continue
        resolved = _coerce_pin_to_slot(raw_pin, periods, days, num_rooms)
        if not resolved:
            unknown.append({"name": raw_name, "pin": raw_pin})
            continue
        pinned[presenter] = resolved
        presenter.pinned_slot = (resolved["period"], resolved["day"])
        presenter.pinned_room = resolved["room"]
    return pinned, unknown


def _initial_schedule(
    slots: Sequence[SlotKey],
    presenters: List[Presenter],
    fixed_empty: set[SlotKey],
) -> Dict[SlotKey, List[Presenter]]:
    schedule = {slot: [] for slot in slots}
    for slot in fixed_empty:
        schedule.pop(slot, None)
        schedule[slot] = []
    return schedule


def _schedule_signature(schedule: Dict[SlotKey, List[Presenter]]) -> Dict[str, SlotKey]:
    result: Dict[str, SlotKey] = {}
    for slot, room_presenters in schedule.items():
        for presenter in room_presenters:
            result[presenter.name] = slot
    return result


def _schedule_diff(a: Dict[str, SlotKey], b: Dict[str, SlotKey], presenters: Sequence[Presenter]) -> int:
    return sum(1 for p in presenters if a.get(p.name) != b.get(p.name))


def _slot_available_for_presenter(
    presenter: Presenter,
    slot: SlotKey,
    schedule: Dict[SlotKey, List[Presenter]],
    structure: Dict[str, Any],
    fixed_empty: set[SlotKey],
    room_unavailable: set[SlotKey],
) -> bool:
    if slot in fixed_empty:
        return False
    if slot in room_unavailable:
        return False
    if (slot[0], slot[1]) not in presenter._availability_set:
        return False
    if presenter.pinned_slot is not None and (slot[0], slot[1]) != presenter.pinned_slot:
        return False
    if presenter.pinned_room is not None and slot[2] != presenter.pinned_room:
        return False
    presenters_in_slot = schedule.get(slot)
    if presenters_in_slot is None:
        return False
    if len(presenters_in_slot) >= int(structure["presenters_per_room"]):
        return False
    if presenter.large_room == "No" and slot[2] == int(structure["large_room_index"]):
        return False
    return True


def _domain_for_presenter(
    presenter: Presenter,
    slots: Sequence[SlotKey],
    structure: Dict[str, Any],
    fixed_empty: set[SlotKey],
    room_unavailable: set[SlotKey],
) -> List[SlotKey]:
    allowed: List[SlotKey] = []
    if presenter.pinned_room is not None and presenter.pinned_slot is not None:
        slot = (presenter.pinned_slot[0], presenter.pinned_slot[1], presenter.pinned_room)
        if slot in fixed_empty:
            return []
        if slot in room_unavailable:
            return []
        if slot in slots:
            return [slot]
        return []

    period_day_pairs = presenter._availability_set
    for period, day, room in slots:
        if period_day_pairs and (period, day) not in period_day_pairs:
            continue
        if room < 0 or room >= int(structure["num_rooms"]):
            continue
        if (period, day, room) in fixed_empty:
            continue
        if (period, day, room) in room_unavailable:
            continue
        if presenter.pinned_slot is not None and (period, day) != presenter.pinned_slot:
            continue
        allowed.append((period, day, room))
    return allowed

def _score_schedule(
    schedule: Dict[SlotKey, List[Presenter]],
    structure: Dict[str, Any],
    penalties: Dict[str, int],
    presenters: Optional[Sequence[Presenter]] = None,
    assignment: Optional[Dict[str, SlotKey]] = None,
) -> Tuple[int, Dict[str, int]]:
    breakdown: Dict[str, int] = {key: 0 for key in penalties}
    timeslot_rooms: Dict[Tuple[str, str], List[SlotKey]] = {}
    if presenters is None:
        all_presenters: List[Presenter] = []
        for room_contents in schedule.values():
            all_presenters.extend(room_contents)
    else:
        all_presenters = list(presenters)

    for slot, room_contents in schedule.items():
        timeslot = (slot[0], slot[1])
        timeslot_rooms.setdefault(timeslot, []).append(slot)

        # Penalty: topic mismatch inside room.
        if len(room_contents) > 1:
            for i, a in enumerate(room_contents):
                for b in room_contents[i + 1 :]:
                    if not a.shares_topic(b):
                        breakdown["topic_mismatch_room"] += penalties["topic_mismatch_room"]

        # Penalty: large room preferred but missed.
        if slot[2] != int(structure["large_room_index"]):
            for presenter in room_contents:
                if presenter.large_room == "Yes":
                    breakdown["large_room_missed"] += penalties["large_room_missed"]

    # Penalty: same topic across rooms in the same period/day.
    for timeslot, slots_in_timeslot in timeslot_rooms.items():
        topic_rooms: Dict[str, int] = {}
        presenters_in_timeslot = 0
        for slot in slots_in_timeslot:
            contents = schedule.get(slot, [])
            presenters_in_timeslot += len(contents)
            topics_in_room = set()
            for presenter in contents:
                topics_in_room.update(presenter._topic_keys)
            for topic in topics_in_room:
                topic_rooms[topic] = topic_rooms.get(topic, 0) + 1
        for count in topic_rooms.values():
            if count > 1:
                # One room is okay; each extra room sharing this topic incurs penalty.
                breakdown["same_topic_across_rooms"] += (count - 1) * penalties["same_topic_across_rooms"]

        # Empties should be clustered into as few rooms as possible.
        if presenters_in_timeslot > 0:
            empty_rooms = sum(1 for slot in slots_in_timeslot if len(schedule.get(slot, [])) == 0)
            if empty_rooms > 1:
                breakdown["empty_slot_not_clustered"] += (
                    (empty_rooms - 1) * penalties["empty_slot_not_clustered"]
                )

    # Penalty: PPP/BF across presenters.
    # Build presenter to position lookup if not provided.
    if assignment is None:
        assignment = _schedule_signature(schedule)
    presenter_count = len(all_presenters)
    for i, presenter_a in enumerate(all_presenters):
        slot_a = assignment.get(presenter_a.name)
        for presenter_b in all_presenters[i + 1 : presenter_count]:
            slot_b = assignment.get(presenter_b.name)
            if slot_a is None or slot_b is None:
                continue
            if slot_a == slot_b:
                # Same slot and room satisfies both PPP and BF constraints.
                continue

            same_timeslot = slot_a[0] == slot_b[0] and slot_a[1] == slot_b[1]
            if presenter_b in presenter_a._ppp_set and presenter_a in presenter_b._ppp_set:
                breakdown["mutual_ppp_not_paired"] += penalties["mutual_ppp_not_paired"]
            elif presenter_b in presenter_a._ppp_set or presenter_a in presenter_b._ppp_set:
                breakdown["one_way_ppp_not_paired"] += penalties["one_way_ppp_not_paired"]

            if same_timeslot:
                if presenter_b in presenter_a._bf_set and presenter_a in presenter_b._bf_set:
                    breakdown["mutual_bf_same_timeslot"] += penalties["mutual_bf_same_timeslot"]
                elif presenter_b in presenter_a._bf_set or presenter_a in presenter_b._bf_set:
                    breakdown["one_way_bf_same_timeslot"] += penalties["one_way_bf_same_timeslot"]

    total = sum(breakdown.values())
    return total, breakdown


def _find_best_slot_for_presenter(
    presenter: Presenter,
    domains: Sequence[SlotKey],
    schedule: Dict[SlotKey, List[Presenter]],
    structure: Dict[str, Any],
    fixed_empty: set[SlotKey],
    room_unavailable: set[SlotKey],
    penalties: Dict[str, int],
    rng: random.Random,
) -> Optional[SlotKey]:
    candidates: List[Tuple[float, SlotKey]] = []
    large_room_idx = int(structure["large_room_index"])

    for slot in domains:
        if not _slot_available_for_presenter(
            presenter, slot, schedule, structure, fixed_empty, room_unavailable
        ):
            continue
        roommates = schedule.get(slot, [])
        score = 0.0
        for existing in roommates:
            score -= presenter.topic_affinity(existing) * 2.0
            if existing in presenter._ppp_set:
                score -= 20.0
        if slot[2] == large_room_idx:
            if presenter.large_room == "No":
                score += 10000.0
            elif presenter.large_room == "Yes":
                score -= float(penalties["large_room_missed"])
        score += rng.random()
        candidates.append((score, slot))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _seed_schedule(
    presenters: List[Presenter],
    slots: List[SlotKey],
    structure: Dict[str, Any],
    penalties: Dict[str, int],
    domains: Dict[Presenter, List[SlotKey]],
    fixed_empty: set[SlotKey],
    room_unavailable: set[SlotKey],
    rng: random.Random,
) -> Tuple[Optional[Dict[SlotKey, List[Presenter]]], List[str]]:
    schedule = _initial_schedule(slots, presenters, fixed_empty)
    hard_conflicts: List[str] = []
    # Fixed room pins are immutable and placed first.
    for presenter in presenters:
        if presenter.pinned_slot and presenter.pinned_room is not None:
            target = (presenter.pinned_slot[0], presenter.pinned_slot[1], presenter.pinned_room)
            if not _slot_available_for_presenter(
                presenter, target, schedule, structure, fixed_empty, room_unavailable
            ):
                hard_conflicts.append(
                    f"{presenter.name} is pinned to room {presenter.pinned_room} in {presenter.pinned_slot[0]} | {presenter.pinned_slot[1]} but cannot be placed."
                )
                return None, hard_conflicts
            schedule[target].append(presenter)

    # Slot-only pins are fixed to slot, but room may be chosen to reduce penalty.
    pinned_slot_only = [p for p in presenters if p.pinned_slot is not None and p.pinned_room is None]
    for presenter in pinned_slot_only:
        domain = [slot for slot in domains.get(presenter, []) if (slot[0], slot[1]) == presenter.pinned_slot]
        choice = _find_best_slot_for_presenter(
            presenter, domain, schedule, structure, fixed_empty, room_unavailable, penalties, rng
        )
        if choice is None:
            hard_conflicts.append(
                f"{presenter.name} is pinned to {presenter.pinned_slot[0]} | {presenter.pinned_slot[1]} but no feasible room is available."
            )
            return None, hard_conflicts
        schedule[choice].append(presenter)

    # Remaining presenters.
    def priority_key(p: Presenter) -> Tuple[int, int, int]:
        return (
            len(domains.get(p, [])),
            -len(p.mutual_ppp),
            -len(p.ppp),
        )

    unpinned = [p for p in presenters if p.pinned_slot is None]
    unpinned.sort(key=priority_key)
    for presenter in unpinned:
        best_slot = _find_best_slot_for_presenter(
            presenter, domains.get(presenter, []), schedule, structure, fixed_empty, room_unavailable, penalties, rng
        )
        if best_slot is None:
            hard_conflicts.append(f"{presenter.name} has no available legal slot in the greedy seeding phase.")
            return None, hard_conflicts
        schedule[best_slot].append(presenter)

    return schedule, []


def _anneal(
    initial_schedule: Dict[SlotKey, List[Presenter]],
    structure: Dict[str, Any],
    penalties: Dict[str, int],
    fixed_empty: set[SlotKey],
    room_unavailable: set[SlotKey],
    iterations_per_temp: int = 10,
    max_outer_iterations: Optional[int] = None,
    time_budget_seconds: Optional[float] = None,
    progress_callback=None,
    rng_seed: Optional[int] = None,
) -> Tuple[int, Dict[SlotKey, List[Presenter]], Dict[str, int]]:
    rng = random.Random(rng_seed)
    current_schedule = {slot: list(occupants) for slot, occupants in initial_schedule.items()}
    all_presenters = [p for room in current_schedule.values() for p in room]
    assignment: Dict[str, SlotKey] = {}
    for slot, occupants in current_schedule.items():
        for presenter in occupants:
            assignment[presenter.name] = slot

    current_score, current_breakdown = _score_schedule(
        current_schedule, structure, penalties, presenters=all_presenters, assignment=assignment
    )
    best_score, best_breakdown, best_schedule = current_score, dict(current_breakdown), current_schedule

    initial_temp = 100.0
    cooling_rate = 0.995
    min_temp = 0.1

    temp = initial_temp
    movable = [p for p in all_presenters if p.pinned_slot is None and p.pinned_room is None]
    if not movable:
        return best_score, best_schedule, best_breakdown

    outer_limit = max_outer_iterations
    if outer_limit is None:
        # Default schedule count-dependent annealing budget tuned for better solution
        # quality on medium-size instances (around 50-70 presenters).
        outer_limit = max(240, min(540, len(all_presenters) * 8))

    outer_count = 0
    if time_budget_seconds is not None and time_budget_seconds <= 0:
        time_budget_seconds = None
    start_time = time.perf_counter()

    while temp > min_temp and outer_count < outer_limit:
        if time_budget_seconds is not None and (time.perf_counter() - start_time) >= time_budget_seconds:
            break
        for _ in range(iterations_per_temp):
            a = rng.choice(movable)
            b = rng.choice(movable)
            if a is b:
                continue

            slot_a = assignment[a.name]
            slot_b = assignment[b.name]
            if slot_a == slot_b or slot_a in fixed_empty or slot_b in fixed_empty:
                continue

            current_schedule[slot_a].remove(a)
            current_schedule[slot_b].remove(b)
            assignment[a.name] = slot_b
            assignment[b.name] = slot_a

            def can_place(p: Presenter, target_slot: SlotKey) -> bool:
                if not _slot_available_for_presenter(
                    p, target_slot, current_schedule, structure, fixed_empty, room_unavailable
                ):
                    return False
                return True

            if can_place(a, slot_b) and can_place(b, slot_a):
                current_schedule[slot_a].append(b)
                current_schedule[slot_b].append(a)
                candidate_score, candidate_breakdown = _score_schedule(
                    current_schedule,
                    structure,
                    penalties,
                    presenters=all_presenters,
                    assignment=assignment,
                )
                delta = candidate_score - current_score
                accept = False
                if delta <= 0:
                    accept = True
                else:
                    acceptance_prob = math.exp(-delta / temp)
                    if rng.random() < acceptance_prob:
                        accept = True

                if accept:
                    current_score = candidate_score
                    current_breakdown = candidate_breakdown
                    if current_score < best_score:
                        best_score = current_score
                        best_breakdown = current_breakdown
                        best_schedule = {slot: list(occupants) for slot, occupants in current_schedule.items()}
                else:
                    current_schedule[slot_a].remove(b)
                    current_schedule[slot_b].remove(a)
                    current_schedule[slot_a].append(a)
                    current_schedule[slot_b].append(b)
                    assignment[a.name] = slot_a
                    assignment[b.name] = slot_b
            else:
                current_schedule[slot_a].append(a)
                current_schedule[slot_b].append(b)
                assignment[a.name] = slot_a
                assignment[b.name] = slot_b

        outer_count += 1
        temp *= cooling_rate

    return best_score, best_schedule, best_breakdown


def _collect_preflight_warnings(
    presenters: List[Presenter],
    structure: Dict[str, Any],
    days: Sequence[str],
    pinned: Dict[Presenter, Dict[str, Any]],
    hard_conflicts: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    warnings: List[Dict[str, str]] = []
    seen_warning_keys: set[Tuple[str, str]] = set()

    if pinned:
        seen_warning_keys.add(("PINS_ADVISORY", "global"))
        warnings.append(
            {
                "code": "PINS_ADVISORY",
                "message": f"{len(pinned)} presenter(s) are manually pinned. Pins may cause cascading conflicts.",
            }
        )

    # Mutual PPP impossible because no overlapping availability.
    seen = set()
    for p in presenters:
        for q in p.mutual_ppp:
            key = tuple(sorted((p.name.lower(), q.name.lower())))
            if key in seen:
                continue
            seen.add(key)
            if not (p._availability_set & q._availability_set):
                warn_key = ("INFEASIBLE_PPP", f"no_overlap:{key[0]}:{key[1]}")
                if warn_key not in seen_warning_keys:
                    warnings.append(
                        {
                            "code": "INFEASIBLE_PPP",
                            "message": f"{p.name} and {q.name} are mutual PPPs but share no available timeslots — pairing is impossible.",
                        }
                    )
                    seen_warning_keys.add(warn_key)
            if p.pinned_slot is not None and q.pinned_slot is not None and p.pinned_slot != q.pinned_slot:
                warn_key = ("INFEASIBLE_PPP", f"pins:{key[0]}:{key[1]}")
                if warn_key not in seen_warning_keys:
                    warnings.append(
                        {
                            "code": "INFEASIBLE_PPP",
                            "message": f"{p.name} and {q.name} are mutual PPPs but are pinned to different slots — pairing is impossible.",
                        }
                    )
                    seen_warning_keys.add(warn_key)

    # Topic overload warning by simple lower bound.
    total_timeslots = max(1, len(days) * len(structure["periods"]))
    counts: Dict[str, int] = {}
    for p in presenters:
        for topic in p.topics:
            key = topic.lower()
            counts[key] = counts.get(key, 0) + 1
    for topic, count in counts.items():
        if count > total_timeslots:
            warnings.append(
                {
                    "code": "TOPIC_OVERLOAD",
                    "message": f"There are {count} presentations in '{topic}', more than can be scheduled without topic overlap in some timeslots.",
                }
            )

    return warnings


def _build_domains(
    presenters: List[Presenter],
    slots: List[SlotKey],
    structure: Dict[str, Any],
    fixed_empty: set[SlotKey],
    room_unavailable: set[SlotKey],
    hard_conflicts: List[Dict[str, str]],
) -> Tuple[Dict[Presenter, List[SlotKey]], List[str], List[Dict[str, str]]]:
    domains: Dict[Presenter, List[SlotKey]] = {}
    unavoidable_hards: List[str] = []
    slot_pin_counts: Dict[Tuple[str, str], int] = {}
    slot_room_pin_counts: Dict[Tuple[str, str, int], int] = {}

    for p in presenters:
        domain = _domain_for_presenter(p, slots, structure, fixed_empty, room_unavailable)
        if p.pinned_slot is not None and not domain:
            unavoidable_hards.append(
                f"{p.name} is pinned to {p.pinned_slot[0]} | {p.pinned_slot[1]} but is not available at that time."
            )
            continue
        if not domain:
            hard_conflicts.append(
                {
                    "code": "HARD_CONFLICT_AVAILABILITY",
                    "message": f"{p.name} has no available timeslots. Please check their availability data.",
                }
            )
            continue
        domains[p] = domain

    for p in presenters:
        if p.pinned_slot is None:
            continue
        slot_key = (p.pinned_slot[0], p.pinned_slot[1], p.pinned_room if p.pinned_room is not None else -1)
        if p.pinned_room is None:
            slot_pin_counts[(p.pinned_slot[0], p.pinned_slot[1])] = slot_pin_counts.get((p.pinned_slot[0], p.pinned_slot[1]), 0) + 1
        else:
            slot_room_pin_counts[slot_key] = slot_room_pin_counts.get(slot_key, 0) + 1

    total_room_cap = int(structure["num_rooms"]) * int(structure["presenters_per_room"])
    for slot_period_day, count in slot_pin_counts.items():
        if count > total_room_cap:
            hard_conflicts.append(
                {
                    "code": "HARD_CONFLICT_OVERPIN",
                    "message": f"Slot {slot_period_day[0]} | {slot_period_day[1]} has {count} presenters pinned but only {total_room_cap} spaces available.",
                }
            )

    for slot_room, count in slot_room_pin_counts.items():
        room_cap = int(structure["presenters_per_room"])
        if count > room_cap:
            hard_conflicts.append(
                {
                    "code": "HARD_CONFLICT_OVERPIN",
                    "message": f"Slot {slot_room[0]} | {slot_room[1]} | Room {slot_room[2]} has {count} presenters pinned but only {room_cap} spaces available.",
                }
            )

    return domains, unavoidable_hards, hard_conflicts


def _serialize_schedule(schedule: Dict[SlotKey, List[Presenter]]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for slot, presenters in schedule.items():
        key = f"{slot[0]}|{slot[1]}|{slot[2]}"
        out[key] = [p.name for p in presenters]
    return out


def _generate_candidates(
    presenters: List[Presenter],
    structure: Dict[str, Any],
    penalties: Dict[str, int],
    num_restarts: int,
    num_results: int,
    fixed_empty: set[SlotKey],
    domains: Dict[Presenter, List[SlotKey]],
    room_unavailable: set[SlotKey],
    iterations_per_temp: int = 12,
    max_outer_iterations: Optional[int] = None,
    time_budget_seconds: Optional[float] = None,
    progress_callback=None,
) -> List[Tuple[int, Dict[SlotKey, List[Presenter]], Dict[str, int]]]:
    slots = [slot for slot in _load_slots(structure, _collect_days(structure["days"]))]
    all_candidates: List[Tuple[int, Dict[SlotKey, List[Presenter]], Dict[str, int]]] = []

    for run_i in range(num_restarts):
        run_seed = random.randrange(1_000_000_000)
        rng = random.Random(run_seed)
        seeded, hard = _seed_schedule(
            presenters=presenters,
            slots=slots,
            structure=structure,
            penalties=penalties,
            domains=domains,
            fixed_empty=fixed_empty,
            room_unavailable=room_unavailable,
            rng=rng,
        )
        if hard:
            if progress_callback is not None:
                progress_callback(run_i + 1, num_restarts, None, len(all_candidates))
            continue
        score, schedule, breakdown = _anneal(
            seeded,
            structure,
            penalties,
            fixed_empty=fixed_empty,
            iterations_per_temp=iterations_per_temp,
            max_outer_iterations=max_outer_iterations,
            time_budget_seconds=time_budget_seconds,
            room_unavailable=room_unavailable,
            progress_callback=progress_callback,
            rng_seed=run_seed + 13,
        )
        all_candidates.append((score, schedule, breakdown))
        if progress_callback is not None:
            progress_callback(run_i + 1, num_restarts, score, len(all_candidates))

    all_candidates.sort(key=lambda item: item[0])
    distinct: List[Tuple[int, Dict[SlotKey, List[Presenter]], Dict[str, int]]] = []
    signatures: List[Dict[str, SlotKey]] = []
    for score, schedule, breakdown in all_candidates:
        signature = _schedule_signature(schedule)
        if all(_schedule_diff(signature, kept, presenters) >= 3 for kept in signatures):
            distinct.append((score, schedule, breakdown))
            signatures.append(signature)
        if len(distinct) >= num_results:
            break
    return distinct


def _collect_days(raw_days: Sequence[str]) -> List[str]:
    if raw_days:
        return _normalize_days([d for d in raw_days if _normalize_text(d)])
    return []


def run(
    url_or_path: str,
    col_config: Dict[str, str] = None,
    structure: Dict[str, Any] = None,
    pins: Dict[str, Any] = None,
    penalties: Dict[str, int] = None,
    num_restarts: int = 20,
    num_results: int = 10,
    progress_callback=None,
    fixed_empty_slots: Optional[Iterable[Tuple[str, str, int]]] = None,
    manual_large_room: Optional[Iterable[str]] = None,
    room_unavailable_slots: Optional[Iterable[str]] = None,
    iterations_per_temp: int = 12,
    max_outer_iterations: Optional[int] = None,
    time_budget_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    if col_config is None:
        col_config = DEFAULT_COL_CONFIG.copy()
    if structure is None:
        structure = DEFAULT_STRUCTURE.copy()
    if penalties is None:
        penalties = PENALTIES.copy()

    structure = dict(structure)
    structure["periods"] = [str(p).strip() for p in structure.get("periods", DEFAULT_STRUCTURE["periods"]) if str(p).strip()]
    structure["num_rooms"] = int(structure.get("num_rooms", DEFAULT_STRUCTURE["num_rooms"]))
    structure["presenters_per_room"] = int(
        structure.get("presenters_per_room", DEFAULT_STRUCTURE["presenters_per_room"])
    )
    structure["large_room_index"] = int(structure.get("large_room_index", DEFAULT_STRUCTURE["large_room_index"]))
    structure["days"] = _normalize_days(
        [d for d in structure.get("days", DEFAULT_STRUCTURE.get("days", [])) if _normalize_text(d)]
    )
    structure["large_room_default"] = str(structure.get("large_room_default", "Maybe"))

    df = load_dataframe(url_or_path)

    presenters, detected_days = parse_presenters(df, col_config, structure)
    _apply_manual_large_room_overrides(presenters, manual_large_room)
    if not structure["days"]:
        structure["days"] = detected_days

    if not structure["days"]:
        return {
            "presenters": presenters,
            "days": [],
            "df": df,
            "hard_conflicts": [
                {
                    "code": "HARD_CONFLICT_AVAILABILITY",
                    "message": "No valid day labels were detected from availability data.",
                }
            ],
            "warnings": [],
            "unavoidable_minimum": 0,
            "results": [],
        }

    fixed_empty = _parse_fixed_empty_slots(fixed_empty_slots, structure, structure["days"])
    room_unavailable = _parse_room_unavailable_slots(room_unavailable_slots, structure, structure["days"])
    pin_map = _parse_pins(pins or {})
    pinned, _ = _build_presenter_pin_lookup(presenters, pin_map, structure, structure["days"])

    slots = _load_slots(structure, structure["days"])
    hard_conflicts: List[Dict[str, str]] = []
    warnings: List[Dict[str, str]] = []

    domains, _unavoidable_hards, hard_conflicts = _build_domains(
        presenters,
        slots,
        structure,
        fixed_empty,
        room_unavailable,
        hard_conflicts,
    )
    hard_conflicts.extend(
        {
            "code": "HARD_CONFLICT_AVAILABILITY",
            "message": message,
        }
        for message in _unavoidable_hards
    )

    warnings.extend(_collect_preflight_warnings(presenters, structure, structure["days"], pinned, hard_conflicts))

    if hard_conflicts:
        return {
            "presenters": presenters,
            "days": structure["days"],
            "df": df,
            "hard_conflicts": hard_conflicts,
            "warnings": warnings,
            "unavoidable_minimum": 0,
            "results": [],
        }

    candidates = _generate_candidates(
        presenters=presenters,
        structure=structure,
        penalties=penalties,
        num_restarts=num_restarts,
        num_results=num_results,
        fixed_empty=fixed_empty,
        domains=domains,
        room_unavailable=room_unavailable,
        iterations_per_temp=iterations_per_temp,
        max_outer_iterations=max_outer_iterations,
        time_budget_seconds=time_budget_seconds,
        progress_callback=progress_callback,
    )

    if not candidates:
        hard_conflicts.append(
            {
                "code": "HARD_CONFLICT_AVAILABILITY",
                "message": "No feasible schedule could be generated from current inputs.",
            }
        )
        return {
            "presenters": presenters,
            "days": structure["days"],
            "df": df,
            "hard_conflicts": hard_conflicts,
            "warnings": warnings,
            "unavoidable_minimum": 0,
            "results": [],
        }

    best_score = candidates[0][0]
    scores = [score for score, _, _ in candidates]
    if len(scores) >= 2:
        median = statistics.median(scores)
        if median > 3 * best_score:
            warnings.append(
                {
                    "code": "CONVERGENCE_FAILURE",
                    "message": f"Most runs produced significantly worse schedules than the best found (best={best_score}, median={median}). Results may not be optimal.",
                }
            )

    threshold = max(1, 300 * max(1, len(presenters)) / 20)
    if best_score > threshold:
        warnings.append(
            {
                "code": "CONVERGENCE_FAILURE",
                "message": f"The scheduler could not find a satisfactory schedule. This may indicate conflicting constraints. (best={best_score}, threshold={threshold:.0f})",
            }
        )

    serialized_results: List[Tuple[int, Dict[str, List[str]], Dict[str, int]]] = []
    for score, schedule, breakdown in candidates:
        serialized_results.append((score, _serialize_schedule(schedule), breakdown))

    return {
        "presenters": presenters,
        "days": structure["days"],
        "df": df,
        "hard_conflicts": hard_conflicts,
        "warnings": warnings,
        "unavoidable_minimum": 0 if best_score is None else best_score,
        "results": serialized_results,
    }


def synthetic_presenters_dataset(
    num_presenters: int = 56,
    seed: int = 42,
    periods: Optional[Sequence[str]] = None,
    days: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    periods = list(periods or DEFAULT_STRUCTURE["periods"])
    days = list(days or ["Tuesday, December 15th", "Thursday, December 17th"])
    topics_pool = [
        "Biology",
        "Math",
        "Engineering",
        "Social Science",
        "Computer Science",
        "Earth Science",
    ]
    rng = random.Random(seed)
    rows = []
    for i in range(num_presenters):
        fname = ["Alex", "Jordan", "Taylor", "Sam", "Riley", "Morgan", "Noah", "Mia", "Kai", "Lena"][i % 10]
        lname = ["Lee", "Patel", "Nguyen", "Ochoa", "Martens", "Santos", "Chen", "Wong", "Diaz", "Ali"][i % 10]
        name = f"{lname}, {fname}"
        title = f"Project {i+1}"
        selected_topics = [topics_pool[(i + j) % len(topics_pool)] for j in range(rng.choice([1, 2]))]
        chosen_pairs: List[str] = []
        for p_i, period in enumerate(periods):
            for d in days:
                if rng.random() < 0.6:
                    chosen_pairs.append(f"{period}, {d}")
        avail = ", ".join(chosen_pairs) if chosen_pairs else ", ".join([f"{periods[0]}, {days[0]}"])
        rows.append(
            {
                "Name": name,
                "Project Title": title,
                "Project Topic (please select up to two)": ", ".join(selected_topics),
                "Availability [PD 2]": "",
                "Availability [PD 3]": "",
                "Availability [PD 4]": "",
                "Availability [PD 5]": "",
                "Availability [PD 6]": "",
                "Preferred co-presenter(s)": "",
                "Best friend(s)": "",
                "Interested in presenting in the SMCS Hub?": rng.choice(["Yes", "No", "Maybe"]),
                "Interested in presenting twice?": "No",
                "Availability": avail,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    synthetic = synthetic_presenters_dataset(num_presenters=54, seed=7)
    synthetic.to_csv("/tmp/synthetic.csv", index=False)
    result = run(
        "/tmp/synthetic.csv",
        col_config=DEFAULT_COL_CONFIG,
        structure=DEFAULT_STRUCTURE,
        pins={},
        penalties=PENALTIES,
        num_restarts=5,
        num_results=3,
    )
    print("Hard conflicts:", len(result["hard_conflicts"]))
    print("Warnings:", len(result["warnings"]))
    print("Results:", len(result["results"]))
    for idx, (score, schedule, breakdown) in enumerate(result["results"], start=1):
        print(idx, score, breakdown["topic_mismatch_room"] if isinstance(breakdown, dict) else 0)
