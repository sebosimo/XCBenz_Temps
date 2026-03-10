# XCBenz Therm

A Streamlit weather app for paragliding pilots, visualising ICON-CH1 NWP forecast data
as emagram soundings and time-height lapse rate plots.

**Live app:** [xcbenz-therm.streamlit.app](https://xcbenz-therm.streamlit.app)

---

## How it works

```
GitHub Actions (every 30 min)
  └─ fetch_data.py        downloads latest ICON-CH1 forecast from MeteoSwiss OGD
  └─ data branch          single orphan commit, force-pushed each run
      ├─ cache_data/      .nc files for all runs within the retention window
      └─ manifest.json    index of all available runs / locations / horizons
  └─ main branch          code only; data_version.txt bumped to trigger Streamlit webhook

Streamlit Community Cloud
  └─ app.py               reads manifest.json + .nc files from GitHub raw URLs at runtime
                          (@st.cache_data handles in-process caching — no redeployment needed)
```

The `data` branch is always a **single orphan commit** — it is force-pushed every 30 min
with a fresh snapshot of all runs within the retention window. No blob history accumulates,
so the `main` branch stays small and clones quickly.

---

## Repository structure

| Branch | Contents | Updated by |
|--------|----------|------------|
| `main` | App code, CI workflow, config files | Manual commits + CI (`data_version.txt`) |
| `data` | `cache_data/` `.nc` files + `manifest.json` | CI force-push every 30 min (orphan commit) |

---

## Key files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI — emagram sounding + time-height lapse rate plot |
| `fetch_data.py` | Downloads ICON-CH1 data; generates `manifest.json`; keeps the 2 most recent runs + today/yesterday's 03:00 anchor |
| `.github/workflows/daily_plot.yml` | CI: restore data → fetch new data → push `data` branch + bump `main` |
| `locations.json` | Flying site coordinates used by the data fetcher |
| `data_version.txt` | Timestamp bumped on every CI run; triggers Streamlit auto-redeploy webhook |

---

## Data layout

```
cache_data/               (on the data branch)
  {YYYYMMDD_HHMM}/        model run timestamp (UTC)
    {Location}/
      H00.nc              forecast horizon +0 h
      H01.nc              forecast horizon +1 h
      ...
```

The app constructs raw GitHub URLs from the manifest at runtime:
`https://raw.githubusercontent.com/{repo}/data/cache_data/{run}/{location}/H{HH}.nc`

---

## CI schedule

GitHub Actions runs at `:12` and `:42` past every hour. Before downloading, CI restores
the existing `data` branch snapshot. If the latest ICON-CH1 run is already fully present
on disk, the download is skipped entirely. After fetching, `cleanup_old_runs()` keeps
only the 2 most recent runs plus the 03:00 run from today and yesterday (the longest
forecast, H00–H45). The updated snapshot is force-pushed to the `data` branch as a
single orphan commit.

---

## Local development

```bash
pip install -r requirements.txt
python fetch_data.py       # downloads data into cache_data/
streamlit run app.py       # starts app (reads live from GitHub by default)
```

`cache_data/` is gitignored locally. The deployed app never reads from local disk.
