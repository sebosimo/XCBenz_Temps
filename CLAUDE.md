# XCBenz Therm — Claude Code Instructions

## What this project is

A Streamlit weather app for paragliding pilots. GitHub Actions fetches ICON-CH1 NWP forecast data every 30 min → commits `.nc` files to the `data` branch → Streamlit Community Cloud serves the app at xcbenz-therm.streamlit.app, which reads data from GitHub at runtime.

---

## Deployment model (critical to understand)

The app reads all weather data **directly from GitHub at runtime** via `raw.githubusercontent.com`. There is no dependency on Streamlit container redeployment for data updates.

- **`main` branch** — app code only. CI commits only `data_version.txt` here (as a Streamlit webhook trigger). Never grows large.
- **`data` branch** — single orphan commit, force-pushed by CI every 30 min. Contains `cache_data/`, `cache_wind/`, and `manifest.json`. No history accumulates.
- `app.py` reads `manifest.json` and `.nc` files from the `data` branch raw URL (`_GH_RAW` points to `/data`)
- `data_version.txt` (tracked on `main`) is updated every CI run to trigger Streamlit's redeploy webhook

**Do not delete or gitignore `data_version.txt`.**
**Never force-push to `main`. Force-pushing `data` is intentional and expected.**

---

## Key files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI — emagram + time-height plot; reads all data from GitHub at runtime |
| `fetch_data.py` | Downloads ICON-CH1 data; `generate_manifest()` writes `manifest.json`; `cleanup_old_runs()` deletes folders older than `RETENTION_DAYS` |
| `.github/workflows/daily_plot.yml` | CI: fetch → commit `data_version.txt` to `main` → orphan-push data to `data` branch |
| `data_version.txt` | Timestamp of last data push; triggers Streamlit redeployment (tracked on `main`) |
| `manifest.json` | Lists all available runs/locations/horizons; lives on `data` branch only |
| `cache_data/{YYYYMMDD_HHMM}/{location}/H{HH}.nc` | Data layout; lives on `data` branch only |

---

## Development workflow

**The local machine does NOT have `cache_data/` — it is too large to keep locally. Do not try to run the app locally with real data.**

### Committing and pushing

CI commits to `main` every 30 min. Always rebase before pushing:

```bash
git pull --rebase origin main && git push
```

Wrapping this in a retry loop (as in `daily_plot.yml`) is good practice for CI.

### How the data branch works

CI creates an orphan commit on a temporary branch `data-temp`, then force-pushes it to `origin/data`. This replaces the entire `data` branch with a single new commit — no history, always a fixed size.

```bash
git checkout --orphan data-temp
git rm -rf --cached .
git add -f cache_data/
git add -f cache_wind/
git add manifest.json
git commit -m "Data snapshot: ..."
git push origin HEAD:data --force
git checkout main
git branch -D data-temp
```

---

## Key architectural decisions made

| Decision | Reason |
|----------|--------|
| `data` branch — single orphan commit, force-pushed | Prevents `.nc` blob history from accumulating on `main`; repo stays small forever |
| `_GH_RAW` points to `data` branch | App reads manifest and `.nc` files from the always-current data snapshot |
| Plots cached as PNG bytes (`io.BytesIO`) | Avoids matplotlib double-free crash when `@st.cache_data` returns a closed figure |
| `@st.cache_data(ttl=1800)` on `render_time_height_plot` | CI runs every 30 min; TTL matches so plots expire and re-render |
| `@st.cache_data(ttl=3600)` on `render_custom_emagram` | Emagram files don't change once written; 1-hour TTL is safe |
| `_last_run` in `st.session_state` | Resets `forecast_index` to 0 when user switches model run |
| `data_version.txt` committed to `main` with each CI run | Triggers Streamlit auto-redeployment webhook |
| `RETENTION_DAYS=2` in CI env | Keeps only last 2 days of runs in the data snapshot |

---

## GitHub

- Repo: `sebosimo/Data_Fetch-ICON-CH1`
- Branch: `main` (code), `data` (weather data snapshots)
- Never force-push to `main`
- Force-pushing `data` is normal and expected
