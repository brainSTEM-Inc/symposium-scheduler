# Symposium Scheduler (Redesign)

This repository is a clean re-implementation scaffold focused on the scheduling algorithm.

## Structure

- `scheduler.py` — standalone scheduling engine (`run()` entrypoint).
- `app.py` — minimal Flask app for CSV URL/file upload and result review.
- `templates/` — Bootstrap pages for upload and review.

## Quick run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
FLASK_DEBUG=1 python3 -m flask run --app app:app --reload
```

Then open `http://127.0.0.1:5000`.

## Local production run

```bash
python3 -m pip install -r requirements.txt
gunicorn --bind 0.0.0.0:5000 app:app
```

## Render deployment

- `Procfile` and `render.yaml` are included for one-click deploy.
- Set environment variables in Render:
  - `FLASK_SECRET_KEY` (required, set via secret)
  - `FLASK_DEBUG=0`
  - `MAX_CONTENT_LENGTH=16777216` (16 MB upload cap)

## Production smoke-check (post-deploy)

Use this minimal route-level checklist after deployment:

1. `GET /` responds with the upload form and accepts a CSV upload.
2. Upload [sample_58_presenters.csv](sample_58_presenters.csv), then confirm Screen 2 renders all presenters.
3. Edit one row on Screen 2 and continue to Screen 3.
4. From Screen 3, run scheduling with default settings.
5. In `/generating`, verify progress updates via `/api/run_status` and eventual completion.
6. Open `/review`, confirm at least one candidate appears.
7. Validate candidate selection and export:
   - Candidate selection via `/review?candidate=0`
   - CSV export: `/review/export/0?format=csv`
   - JSON export: `/review/export/0?format=json`

If any route fails, capture request/response in the browser network panel and check logs for exception traces.

## Test with CLI

```python
import pathlib, sys
sys.path.insert(0, "symposium-scheduler")
from scheduler import run, DEFAULT_STRUCTURE, DEFAULT_COL_CONFIG, PENALTIES

result = run(
    "sample.csv",
    col_config=DEFAULT_COL_CONFIG,
    structure=DEFAULT_STRUCTURE,
    penalties=PENALTIES,
    num_restarts=4,
    num_results=3,
)
```

## Test CSV

- [`sample_58_presenters.csv`](sample_58_presenters.csv)
  - ~58 synthetic presenters
  - reasonable `Best friend(s)` coverage
  - sparse but present `Preferred co-presenter(s)` entries

## Notes

- Current implementation is intentionally compact and functional-first.
- Data model and algorithm are independent of Flask and can be called directly via `run()`.
