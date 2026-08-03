from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
import time
from pathlib import Path
from typing import Dict, List, Sequence

from scheduler import DEFAULT_COL_CONFIG, DEFAULT_STRUCTURE, PENALTIES, run


FIRST_NAMES = [
    "Akash",
    "Katherine",
    "Krishan",
    "Maranchi",
    "Pranavi",
    "Srinivasan",
    "Sabova",
    "Nicole",
    "Cailyn",
    "Farrell",
    "Wang",
    "Sherry",
    "Jordan",
    "Noah",
    "Mia",
    "Leah",
]

LAST_NAMES = [
    "Saran",
    "Xu",
    "Park",
    "Zhou",
    "Santos",
    "Chen",
    "Wong",
    "Park",
    "Ali",
    "Jetta",
    "Jessika",
    "Mitin",
    "Veronica",
    "Farrell",
    "Verma",
    "Lee",
]

TOPICS = [
    "Biology",
    "Math",
    "Engineering",
    "Social Science",
    "Computer Science",
    "Earth Science",
    "Physics",
    "Chemistry",
]

DAYS = ["Tuesday, December 15th", "Thursday, December 17th"]
PERIODS = ["PD 2", "PD 3", "PD 4", "PD 5", "PD 6"]


def _format_seconds(seconds: float) -> str:
    return f"{seconds:.2f}s"


def _normalize_name(first: str, last: str, idx: int) -> str:
    return f"{last}, {first}{idx}"


def _build_availability(rng: random.Random, periods: Sequence[str], days: Sequence[str]) -> str:
    pairs = []
    for period in periods:
        for day in days:
            if rng.random() < 0.62:
                pairs.append((period, day))
    if not pairs:
        pairs.append((periods[0], days[0]))
    return ", ".join([f"{period}, {day}" for period, day in pairs])


def generate_benchmark_csv(path: Path, presenters: int, seed: int = 2026) -> None:
    rng = random.Random(seed)
    rows: List[Dict[str, str]] = []
    names = []

    for i in range(presenters):
        first = FIRST_NAMES[i % len(FIRST_NAMES)]
        last = LAST_NAMES[i % len(LAST_NAMES)]
        names.append(_normalize_name(first, last, i))

    for idx, name in enumerate(names):
        selected_topics = rng.sample(TOPICS, k=rng.choice([1, 2]))
        availability = _build_availability(rng, PERIODS, DAYS)
        # PPP/BF are optional and weakly coupled to realism.
        ppp_candidates = [n for n in names if n != name]
        bf_candidates = [n for n in names if n != name]
        ppp = ", ".join(rng.sample(ppp_candidates, k=min(3, max(1, len(ppp_candidates) // 8))))
        bf = ", ".join(rng.sample(bf_candidates, k=min(3, max(1, len(bf_candidates) // 10))))
        rows.append(
            {
                DEFAULT_COL_CONFIG["name"]: name,
                DEFAULT_COL_CONFIG["title"]: f"Project {idx + 1}",
                DEFAULT_COL_CONFIG["topics"]: ", ".join(selected_topics),
                DEFAULT_COL_CONFIG["availability"]: availability,
                DEFAULT_COL_CONFIG["ppp"]: ppp,
                DEFAULT_COL_CONFIG["bf"]: bf,
                DEFAULT_COL_CONFIG["large_room"]: rng.choice(["Yes", "No", "Maybe"]),
                DEFAULT_COL_CONFIG["present_twice"]: "No",
            }
        )

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            DEFAULT_COL_CONFIG["name"],
            DEFAULT_COL_CONFIG["title"],
            DEFAULT_COL_CONFIG["topics"],
            DEFAULT_COL_CONFIG["availability"],
            DEFAULT_COL_CONFIG["ppp"],
            DEFAULT_COL_CONFIG["bf"],
            DEFAULT_COL_CONFIG["large_room"],
            DEFAULT_COL_CONFIG["present_twice"],
        ])
        writer.writeheader()
        writer.writerows(rows)


def _dataset_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def ensure_dataset(path: Path, presenters: int, seed: int, force: bool = False) -> None:
    if force or not path.exists() or _dataset_row_count(path) != presenters:
        if path.exists():
            print(f"Regenerating dataset at {path} ({presenters} presenters).")
        generate_benchmark_csv(path, presenters=presenters, seed=seed)


def benchmark(
    csv_path: Path,
    presenter_count: int,
    restarts: int,
    results: int,
    iterations_per_temp: int,
    max_outer_iterations: int | None,
    time_budget_seconds: float | None,
    repeat: int,
) -> None:
    print(f"Dataset: {csv_path}")
    print(f"Presenters: {presenter_count}, restarts={restarts}, results={results}")
    if max_outer_iterations is not None and max_outer_iterations < 1:
        max_outer_iterations = None
    if time_budget_seconds is not None and time_budget_seconds < 0:
        time_budget_seconds = None
    budget_display = "none" if time_budget_seconds is None else f"{time_budget_seconds:.2f}s (per-run)"
    print(
        f"Iterations/temp: {iterations_per_temp}, outer cap: {max_outer_iterations or 'auto'}, "
        f"time budget: {budget_display}"
    )
    elapsed: List[float] = []
    best_scores: List[int] = []

    structure = dict(DEFAULT_STRUCTURE)
    structure["periods"] = PERIODS
    structure["days"] = DAYS

    for i in range(1, repeat + 1):
        start = time.perf_counter()
        result = run(
            str(csv_path),
            col_config=DEFAULT_COL_CONFIG,
            structure=structure,
            penalties=PENALTIES,
            num_restarts=restarts,
            num_results=results,
            iterations_per_temp=iterations_per_temp,
            max_outer_iterations=max_outer_iterations,
            time_budget_seconds=time_budget_seconds,
        )
        elapsed_ms = time.perf_counter() - start
        elapsed.append(elapsed_ms)

        if result["results"]:
            score = result["results"][0][0]
            best_scores.append(score)
        else:
            score = math.inf

        print(
            f"Run {i}/{repeat}: elapsed={_format_seconds(elapsed_ms)}, "
            f"best={score}, hard={len(result['hard_conflicts'])}, warnings={len(result['warnings'])}, "
            f"candidates={len(result['results'])}"
        )

    if elapsed:
        print("\nBenchmark summary")
        print(f"  mean runtime: {statistics.mean(elapsed):.2f}s")
        print(f"  median runtime: {statistics.median(elapsed):.2f}s")
        print(f"  min/max runtime: {min(elapsed):.2f}s / {max(elapsed):.2f}s")
    finite: List[int] = []
    if best_scores:
        finite = [s for s in best_scores if s != math.inf]
        if finite:
            print(f"  best-score mean: {statistics.mean(finite):.2f}")
            print(f"  best-score median: {statistics.median(finite):.2f}")
            print(f"  score stdev: {statistics.pstdev(finite):.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run schedule generation timing benchmarks")
    parser.add_argument("--presenters", type=int, default=60)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--results", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--iterations-per-temp", type=int, default=8)
    parser.add_argument("--max-outer-iterations", type=int, default=None)
    parser.add_argument("--time-budget-seconds", type=float, default=None, help="Wall-clock budget for each benchmark run")
    parser.add_argument("--dataset", default=str(Path("bench_data_60.csv")))
    parser.add_argument("--warmup", type=int, default=0, help="Optional warm-up runs before timing starts")
    parser.add_argument("--force-generate", action="store_true", help="Overwrite dataset file before benchmarking")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    ensure_dataset(dataset_path, presenters=args.presenters, seed=args.seed, force=args.force_generate)

    if args.warmup > 0:
        warmup_runs = min(args.warmup, 3)
        print(f"Running {warmup_runs} warm-up runs...")
        benchmark(
            dataset_path,
            presenter_count=args.presenters,
            restarts=args.restarts,
            results=args.results,
            iterations_per_temp=args.iterations_per_temp,
            max_outer_iterations=args.max_outer_iterations,
            time_budget_seconds=args.time_budget_seconds,
            repeat=warmup_runs,
        )
        print("Warm-up complete.")

    benchmark(
        dataset_path,
        presenter_count=args.presenters,
        restarts=args.restarts,
        results=args.results,
        iterations_per_temp=args.iterations_per_temp,
        max_outer_iterations=args.max_outer_iterations,
        time_budget_seconds=args.time_budget_seconds,
        repeat=args.repeat,
    )


if __name__ == "__main__":
    main()
