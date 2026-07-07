# tools/ — building & publishing the TC-diagnostics archive

`build_archive.py` precomputes the 7 diagnostics (GPIv, vPI, PI,
ventilation_index, VWS, Chi, eta_c) from ERA5 monthly means (NCAR RDA
d633001, OPeNDAP) for 1980–2022 on a 40°S–40°N band at 0.5° (or 1° with --stride 4; release v1 was built at 1° locally), and produces
exactly the files `../tcdiag_data.py` expects on the GitHub Release:

| file | dims | contents |
|---|---|---|
| `tcdiag__{var}__{decade}.nc` (e.g. `tcdiag__vPI__1980s.nc`) | `(time, latitude, longitude)` | monthly means, one file per variable per decade, float32+zlib |
| `tcdiag_clim__{var}.nc` | `(month: 12, latitude, longitude)` | `{var}` mean + `{var}_STD` (interannual std, ddof=1) per calendar month |
| `tcdiag_clim_seasonal__{var}.nc` | `(season: 3, latitude, longitude)` | same for seasons `JJA`, `SON`, `JJASON` |

35 archive files (7 vars × 5 decades) + 14 climatology files.

The 7 ERA5 inputs per month are fetched with a **process** pool (never
threads — netCDF-C is not thread-safe for concurrent OPeNDAP opens) and
subset server-side before download: latitude band, stride, and only the
pressure levels each input needs (T/Q 50–1000 hPa, U/V 200+850 hPa, VO 850 hPa).

## Runbook (Purdue Negishi)

### 0. One-time setup (login node)

```bash
module load anaconda
conda activate research_env
pip install tcpyVPI tcpyPI
```

### 1. Monthly archive — SLURM array, one task per year

```bash
cd ~/TCdiag_streamlit          # repo root (submit dir matters)
mkdir -p logs
sbatch tools/run_archive.slurm # array 1980-2022, 45 tasks
```

Each task runs
`python tools/build_archive.py monthly --years {YEAR}-{YEAR} --out /scratch/negishi/jocegue/tcdiag_archive`.

* **Runtime:** ~30–60 min per year-task (RDA transfer dominates; the PI/GPIv
  compute is seconds per month). Fine for the standby queue's 4 h cap.
* **Resumable / crash-safe:** finished months are skipped on resubmit; all
  writes are atomic (`.tmp` + rename); a month that fails (THREDDS hiccup)
  is logged to `{out}/failed_months.txt` and the run continues — just
  `sbatch` again to fill gaps. If a task died mid-consolidation, delete the
  stale `{out}/.consolidate_*.lock` and rerun with `--consolidate-only`.
* The last task to finish a decade folds the per-month files into the
  per-decade files (lock-protected, concurrent-safe).

### 2. Climatology (ONCE, after all array tasks finish)

Local reads only (no downloads) — a small single job or even the login node:

```bash
OUT=/scratch/negishi/jocegue/tcdiag_archive
CLIM=/scratch/negishi/jocegue/tcdiag_clim
# fold any leftover per-month files into the decade files:
python tools/build_archive.py monthly --years 1980-2022 --out "$OUT" --consolidate-only
# per-calendar-month + seasonal (JJA/SON/JJASON) mean/std, ddof=1:
python tools/build_archive.py climatology --years 1980-2022 --archive "$OUT" --out "$CLIM"
```

(A ready-made single-job script is in the comment block at the bottom of
`run_archive.slurm`.) Takes a few minutes.

### 3. Verify before publishing

```bash
python tools/build_archive.py verify --archive "$OUT" --clim "$CLIM"
```

Checks month coverage (540 months/variable), all-NaN fields, plausible value
ranges, climatology dims/variables. Exits non-zero on any problem — do not
upload until it passes.

Expected sizes: **~875 MB** archive (35 files) + **~60 MB** climatology
(14 files).

### 4. Publish to the GitHub Release

The app reads tag **`tcdiag-v1`** of `Langosmon/TCdiag_streamlit`
(`RELEASE_TAG` in `tcdiag_data.py`). From any machine with `gh`
authenticated and the files visible (run on the cluster if `gh` is
available there, otherwise `scp` the 49 files to your laptop first):

```bash
gh release create tcdiag-v1 \
    --repo Langosmon/TCdiag_streamlit \
    --title "TC-diagnostics archive v1 (ERA5 1980-2022, 0.5deg, 40S-40N)" \
    --notes "Monthly archive + 1980-2022 monthly/seasonal climatology built by tools/build_archive.py. See tools/README.md." \
    /scratch/negishi/jocegue/tcdiag_archive/tcdiag__*.nc \
    /scratch/negishi/jocegue/tcdiag_clim/tcdiag_clim*.nc
```

(The archive glob only matches the per-decade files; leftover per-month
files live under `monthly/` and are consolidated away by step 2.)

To add/replace assets on an existing release:

```bash
gh release upload tcdiag-v1 --repo Langosmon/TCdiag_streamlit --clobber \
    /scratch/negishi/jocegue/tcdiag_archive/tcdiag__*.nc \
    /scratch/negishi/jocegue/tcdiag_clim/tcdiag_clim*.nc
```

**No app code change is needed:** the deployed app probes the release once
per session and switches from live computation to the archive automatically
as soon as the assets exist (a running session may need a restart/refresh to
re-probe).

## Notes

* Keep `--domain`/`--stride` identical across all runs feeding one archive
  directory (grids must match for the decade concat). Defaults: `40,-40`,
  stride 2 (0.5° (or 1° with --stride 4; release v1 was built at 1° locally)).
* `--workers N` controls the fetch process pool (default 7, one per ERA5
  input; `1` = sequential). Never convert this to threads.
* Running elsewhere: only `--out`/paths are cluster-specific; the script
  itself just needs a fast pipe to `thredds.rda.ucar.edu`.
