# CH2 Integration — Testing & Deployment Roadmap

## What's being added

| Component | Description |
|-----------|-------------|
| `fetch_data_ch2.py` | Downloads ICON-CH2-EPS data (5-day horizon, H000–H120) |
| `generate_combined_manifest.py` | Merges CH1 + CH2 run inventories into a single `manifest.json` |
| `app.py` (updated) | CH2 emagram extension, 5-day time-height plot, combined slider |
| `daily_plot.yml` (updated) | Runs CH2 fetcher + manifest generator; restores + commits `cache_data_ch2/` |

---

## Phase 1 — Local: Verify CH2 download works

**Goal:** Confirm `fetch_data_ch2.py` successfully connects to the MeteoSwiss API
and writes `.nc` files.

- [ ] Run `python fetch_data_ch2.py` manually
- [ ] Confirm log shows a run being found (not "No CH2 runs found")
- [ ] Check `cache_data_ch2/{YYYYMMDD_HHMM}/{location}/H000.nc` exists
- [ ] Let it run to completion (or at least ~10 horizons) and restart to verify it resumes from the right horizon
- [ ] If "No CH2 runs found" persists, check API availability for the current hour

---

## Phase 2 — Local: Verify combined manifest

**Goal:** Confirm `generate_combined_manifest.py` produces a valid merged manifest.

- [ ] Run `python generate_combined_manifest.py` after a partial/full CH2 download
- [ ] Open `manifest.json` and check that it contains both:
  - `"runs"` key — CH1 runs
  - `"runs_ch2"` key — CH2 runs with location/step inventory
- [ ] Confirm the CH2 run tag format matches `YYYYMMDD_HHMM`

---

## Phase 3 — Local: Verify app logic (no server needed)

**Goal:** Confirm the new app functions work correctly against local data.

- [ ] Run `streamlit run app.py` locally
- [ ] Select a CH1 run → check that the emagram slider shows CH2-extended steps (labeled `CH2:H...`)
- [ ] Select a CH2-prefixed step in the emagram → confirm it loads without error
- [ ] Open the "ICON-CH2 (5-day)" tab → confirm the time-height plot renders
- [ ] Verify `find_matching_ch2_run()` picks the right CH2 run for the selected CH1 run
- [ ] Verify `get_ch2_extension_steps()` only shows steps strictly beyond the CH1 cutoff

---

## Phase 4 — Push code to GitHub (`main`)

**Goal:** Get updated code live without breaking the current app.

- [ ] Ensure `fetch_data_ch2.py` and `generate_combined_manifest.py` are committed
- [ ] Ensure `app.py` and `.github/workflows/daily_plot.yml` changes are committed
- [ ] Do **not** commit `manifest.json`, `cache_data_ch2/`, or `debug_log_ch2.txt`
- [ ] Run the deploy workflow:

```bash
git pull --rebase origin main
git add fetch_data_ch2.py generate_combined_manifest.py app.py .github/workflows/daily_plot.yml
git commit -m "Add ICON-CH2 integration: fetcher, manifest, app UI"
git push
```

> The current app on Streamlit will continue working — CH1 data is unchanged.
> The app gracefully handles a missing `runs_ch2` key (CH2 section is hidden until data arrives).

---

## Phase 5 — Trigger first CI run with CH2

**Goal:** Confirm GitHub Actions fetches CH2 data and pushes it to the `data` branch.

- [ ] Go to **Actions → daily_plot → Run workflow** (manual trigger)
- [ ] Watch the 3 new steps:
  - `Fetch CH2 Weather Data`
  - `Generate Combined Manifest`
  - Commit block: `git add -f cache_data_ch2/`
- [ ] After CI finishes, check the `data` branch contains `cache_data_ch2/`
- [ ] Download `manifest.json` from the `data` branch raw URL and verify `runs_ch2` is present

---

## Phase 6 — Verify live app on Streamlit

**Goal:** Confirm the deployed app shows CH2 data correctly.

- [ ] Open xcbenz-therm.streamlit.app
- [ ] Select a recent CH1 run → verify the emagram slider now has CH2-extended steps
- [ ] Load the CH2 time-height plot tab → confirm it renders
- [ ] Check that `manifest["generated_at"]` timestamp is recent (footer)
- [ ] Test on mobile (paragliding use case)

---

## Phase 7 — Monitor first few automated runs

**Goal:** Confirm CI runs reliably every 30 min with CH2 included.

- [ ] Check Actions logs after 2–3 automated runs
- [ ] Confirm `cache_data_ch2/` is being restored from the `data` branch (resume logic works)
- [ ] Confirm the CH2 run is being skipped if already complete (`is_run_complete_locally`)
- [ ] Confirm `cleanup_old_runs()` equivalent in CH2 is keeping the `data` branch lean

---

## Rollback plan

If anything breaks after Phase 4:

1. The app still works for CH1 — CH2 section is additive and hidden if `runs_ch2` missing
2. To fully revert: `git revert HEAD` on `main` and push
3. The `data` branch can be left as-is — old manifests without `runs_ch2` work fine

---

## Open issues / known gaps

| Issue | Status |
|-------|--------|
| "No CH2 runs found" during probing | Needs investigation — may be API availability or time-of-day window |
| `cleanup_old_runs()` not yet implemented for CH2 | CH2 cache may grow unbounded — add before full CI deployment |
| Wind maps for CH2 | Not implemented, not needed for initial release |
