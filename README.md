# XCBenz Therm

A Streamlit weather app for paragliding pilots, visualising ICON-CH1 NWP forecast data
as emagram soundings and time-height lapse rate plots.

**Live app:** [xcbenz-therm.streamlit.app](https://xcbenz-therm.streamlit.app)

---

## How it works

```
GitHub Actions (every 30 min)
  └─ fetch_data.py          downloads ICON-CH1 forecast data
  └─ cache_data/{run}/...   .nc files committed to repo (force-added)
  └─ manifest.json          index of all available runs/locations/horizons
  └─ data_version.txt       timestamp, triggers Streamlit webhook

Streamlit Community Cloud
  └─ app.py                 reads manifest.json + .nc files directly from
                            GitHub at runtime — no redeployment needed
```

Because the app fetches data from GitHub raw URLs at runtime (cached in-process
with `@st.cache_data`), it always shows the latest model runs without depending on
Streamlit container redeployment.

---

## Key files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI — emagram + time-height lapse rate plot |
| `fetch_data.py` | Downloads ICON-CH1 data; generates `manifest.json`; cleans up runs older than `RETENTION_DAYS` |
| `.github/workflows/daily_plot.yml` | CI: fetch → commit → push on a 30-min schedule |
| `manifest.json` | Auto-generated index of available runs, locations, and horizon files; read by the app at runtime |
| `data_version.txt` | Timestamp of last data push; kept as a Streamlit auto-redeploy webhook trigger |
| `locations.json` | Flying site coordinates used by the data fetcher |

---

## Data layout

```
cache_data/
  {YYYYMMDD_HHMM}/          model run timestamp (UTC)
    {Location}/
      H00.nc                horizon +0 h
      H01.nc                horizon +1 h
      ...
```

`cache_data/` is gitignored but force-added by CI (`git add -f`) so it lives in the
repo and is accessible via `raw.githubusercontent.com`.

---

## CI schedule

GitHub Actions runs at `:12` and `:42` past every hour. It only commits when new
ICON-CH1 data is available (checked via the MeteoSwiss OGD STAC API). Old runs are
deleted after `RETENTION_DAYS=2` to keep repo size manageable.

---

## Local development

```bash
pip install -r requirements.txt
python fetch_data.py       # downloads data into cache_data/
streamlit run app.py       # starts app locally
```

The app reads live from GitHub by default, so `cache_data/` is only needed if you
want to test the fetcher locally. The deployed app on Streamlit Community Cloud never
reads from local disk.
