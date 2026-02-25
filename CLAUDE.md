# XCBenz Therm — Claude Code Instructions

## What this project is

A Streamlit weather app for paragliding pilots. GitHub Actions fetches ICON-CH1 NWP forecast data every 30 min → commits `.nc` files to `cache_data/` → Streamlit Community Cloud deploys from the repo and serves the app at xcbenz-therm.streamlit.app.

---

## Deployment model (critical to understand)

Streamlit Community Cloud serves the app from a **static filesystem snapshot** made at deployment time. The container only gets new `cache_data/` files when it **redeploys**.

- `cache_data/` and `cache_wind/` are **gitignored** but **force-added** in CI with `git add -f`
- **`data_version.txt`** (tracked, in repo root) is written by CI on every data push — this tracked-file change is what triggers Streamlit's auto-redeploy webhook reliably
- If the Streamlit UI shows "Relaunch to update", it means the container is running an old deployment and must be manually relaunched

**Do not delete or gitignore `data_version.txt`.**

---

## Key files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI — emagram + time-height plot |
| `fetch_data.py` | Downloads ICON-CH1 data; `cleanup_old_runs()` deletes folders older than `RETENTION_DAYS` |
| `.github/workflows/daily_plot.yml` | CI: fetch → commit → push on a 30-min schedule |
| `data_version.txt` | Timestamp of last data push; triggers Streamlit redeployment |
| `cache_data/{YYYYMMDD_HHMM}/{location}/H{HH}.nc` | Data layout |

---

## Development workflow

**The local machine does NOT have `cache_data/` — it is too large to keep locally. Do not try to run the app locally with real data.**

### Committing and pushing

CI commits to `main` every 30 min. Always rebase before pushing:

```bash
git pull --rebase origin main && git push
```

Wrapping this in a retry loop (as in `daily_plot.yml`) is good practice for CI.

### Staging cache_data deletions correctly

`cleanup_old_runs()` uses `shutil.rmtree()` on the runner. To ensure deletions are staged:

```bash
git ls-files --deleted cache_data/ | xargs -r git rm -r --cached --
git add -f cache_data/
```

This must be done before `git add -f` or deletions may not be committed.

---

## Key architectural decisions made

| Decision | Reason |
|----------|--------|
| `@st.cache_data(ttl=1800)` on `render_time_height_plot` | CI runs every 30 min; TTL matches so plots expire and re-render |
| `@st.cache_data(ttl=3600)` on `render_custom_emagram` | Emagram files don't change once written; 1-hour TTL is safe |
| `_last_run` in `st.session_state` | Resets `forecast_index` to 0 when user switches model run |
| `data_version.txt` committed with each data push | Guarantees Streamlit auto-redeployment on new data |
| `RETENTION_DAYS=2` in CI env | Keeps only last 2 days of runs to limit repo size |

---

## GitHub

- Repo: `sebosimo/Data_Fetch-ICON-CH1`
- Branch: `main`
- Never force-push to main
