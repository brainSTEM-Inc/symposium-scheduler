# Symposium Scheduler (Redesign)

This repository contains the Flask-based symposium scheduler and its standalone scheduling engine.

## Structure

- `scheduler.py` — standalone scheduling engine (`run()` entrypoint).
- `app.py` — minimal Flask app for CSV URL/file upload and result review.
- `templates/` — five workflow screens, from data upload through schedule review.
- `static/css/app.css` — shared colors, focus indicators, and responsive accessibility styles.
- `static/vendor/` — the Bootstrap 5.3.3 CSS and JavaScript bundle used locally, so screens do not depend on a public asset CDN.

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

- `render.yaml` defines the Render web service. In Render, create a Blueprint from this repository and select the `redesign-2025` branch.
- Set `FLASK_SECRET_KEY` as a secret environment variable before using the service. Render sets `PORT`; the service runs one Gunicorn worker because workflow state and progress are held in process memory.
- The free instance can sleep while idle. A schedule generation continues in the active process and the Screen 4 page polls that process for progress.
- The public `onrender.com` service URL is configured by Render after deployment. Keep the service URL available to the school network administrator to confirm it is reachable from school devices.

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

## Verification

Run the core tests with `python -m pytest -q`. For a larger timing and quality check, `performance_benchmark.py` creates deterministic synthetic CSV datasets and can run repeated instances at 50–60 presenters.

## Test CSV

- [`sample_58_presenters.csv`](sample_58_presenters.csv)
  - ~58 synthetic presenters
  - dense `Best friend(s)` coverage (about 8 per presenter on average)
  - sparse but present `Preferred co-presenter(s)` entries

## Notes

- Current implementation is intentionally compact and functional-first.
- Data model and algorithm are independent of Flask and can be called directly via `run()`.
