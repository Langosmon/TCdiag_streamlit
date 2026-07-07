# TCdiag · vPI / PI / Ventilation / GPIv maps

Interactive tropical-cyclone diagnostics from ERA5, built on the lab's
[tcpyVPI](https://github.com/drchavas/tcpyVPI) package (Chavas, Camargo &
Tippett 2025, *J. Climate*). Deployed on Streamlit Cloud and embedded at
[langosmon.github.io](https://langosmon.github.io) (Science Maps tab).

## What it shows

Seven diagnostics on a 40°S–40°N band: **vPI** (ventilated potential
intensity), **PI**, **Ventilation Index**, **GPIv** (genesis potential),
**VWS** (200–850 hPa shear), **χ** (entropy deficit), **η_c** (capped
850 hPa absolute vorticity).

Three modes:

| Mode | Source | Speed |
|---|---|---|
| Climatology (monthly + JJA/SON/JJASON, mean and σ, 1980–2024) | precomputed archive on a GitHub Release | instant |
| Monthly (any month 1980–2024, with anomalies) | archive, with live-compute fallback | instant / ~1 min |
| Daily 6-hourly (00/06/12/18 UTC, any date since 1940, with anomalies) | computed live from RDA THREDDS via tcpyVPI | ~1 min first view, then cached |

## Architecture

- `app.py` — the Streamlit app.
- `tcdiag_data.py` — release-hosted archive access + live computation.
- `tcdiag_fetch.py` — per-variable ERA5 fetch (server-side subsetting;
  sequential on purpose: netCDF-C is not thread-safe for parallel OPeNDAP).
- `_common.py` — UI shell shared with the
  [ERA5_streamlit](https://github.com/Langosmon/ERA5_streamlit) /
  [ERA5_hourly_streamlit](https://github.com/Langosmon/ERA5_hourly_streamlit)
  apps (keep the three copies in sync).
- `tools/` — the cluster pipeline that builds the 1980–2024 archive and
  climatology, plus the runbook for publishing them as the `tcdiag-v1`
  GitHub Release. Until that release exists the app runs in live-compute
  mode with a friendly notice.

Live computation keeps transfers small by requesting only the levels each
term needs (full T/Q profiles for PI; U/V at 200/850 hPa; VO at 850 hPa)
and subsetting to the tropical band server-side before download.

## Deploy

Streamlit Cloud → New app → this repo, `app.py`, Python ≥3.11. The first
build takes a few minutes (tcpyPI compiles its numba kernels once).
