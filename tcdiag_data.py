"""Data access for the TC-diagnostics app.

Three tiers, fastest first:
  1. Climatology files (1980–2022 monthly + seasonal means/stds; RDA's monthly dataset ends at 2022) — published
     on a GitHub Release, fetched once, cached in /tmp.
  2. Monthly archive (one file per variable per decade) — same Release.
  3. Live computation from ERA5 via tcpyVPI — used for daily 6-hourly maps
     (which are never precomputed) and as a fallback for monthly maps until
     the archive is published. RDA transfer dominates the cost (~1–2 min);
     the PI computation itself is seconds on the subset grid.

All fields are on a 40°S–40°N tropical band. Archive/climatology resolution
follows the published release (v1 is 1°); live computation defaults to 1°
(stride 4) for speed with a 0.5° option. Anomalies regrid automatically.

File-name contract with tools/build_archive.py (keep in sync):
  tcdiag__{var}__{decade}.nc            (e.g. tcdiag__vPI__1980s.nc)            (time, latitude, longitude)
  tcdiag_clim__{var}.nc                  (month:12, ...)  {VAR}, {VAR}_STD
  tcdiag_clim_seasonal__{var}.nc         (season:3, ...)  {VAR}, {VAR}_STD
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import streamlit as st
import xarray as xr

# ─────────── catalogue ────────────────────────────────────────────────────────
# name → (file_var, units, cmap_abs, cmap_anom, quantile_default)
# quantile_default=True → default colour-bar spans the 1–99% quantiles
# (GPIv and VI are heavy-tailed; min/max would wash the map out).
DIAGNOSTICS: dict[str, tuple] = {
    "Ventilated PI (vPI)":      ("vPI",  "m s⁻¹", "thermal", "RdBu_r", False),
    "Potential Intensity (PI)": ("PI",   "m s⁻¹", "thermal", "RdBu_r", False),
    "Ventilation Index":        ("ventilation_index", "–", "plasma", "RdBu_r", True),
    "GPIv (genesis index)":     ("GPIv", "–",     "viridis", "PuOr",  True),
    "Vertical Wind Shear":      ("VWS",  "m s⁻¹", "viridis", "RdBu_r", False),
    "Entropy Deficit (χ)":      ("Chi",  "–",     "viridis", "BrBG",  True),
    "Capped Vorticity (η_c)":   ("eta_c", "s⁻¹",  "plasma",  "RdBu_r", False),
}

SEASONS = ["JJA", "SON", "JJASON"]
YEARS = list(range(1980, 2023))   # RDA d633001 monthly means end at 2022

RELEASE_BASE = "https://github.com/Langosmon/TCdiag_streamlit/releases/download"
RELEASE_TAG = "tcdiag-v1"

# Live-compute domain: tropical band, ERA5 latitude runs 90 → -90.
LAT_MAX, LAT_MIN = 40, -40


# ─────────── Release fetch (atomic, cached on disk + in memory) ──────────────
def _release_local(name: str) -> Path:
    cache = Path("/tmp/tcdiag")
    cache.mkdir(parents=True, exist_ok=True)
    return cache / name


@st.cache_resource(show_spinner="Fetching archive…", max_entries=16)
def _release_dataset(name: str) -> xr.Dataset:
    """Download (once) + open (once) a Release asset, fully in memory."""
    local = _release_local(name)
    if not local.exists():
        url = f"{RELEASE_BASE}/{RELEASE_TAG}/{name}"
        part = local.with_suffix(".part")
        try:
            r = requests.get(url, stream=True, timeout=60)
            if r.status_code != 200:
                raise FileNotFoundError(
                    f"'{name}' is not in release '{RELEASE_TAG}' yet "
                    f"(HTTP {r.status_code}). Run tools/build_archive.py and "
                    f"upload — see tools/README.md."
                )
            with open(part, "wb") as f:
                for chunk in r.iter_content(1 << 16):
                    f.write(chunk)
            part.replace(local)
        except requests.exceptions.RequestException as e:
            part.unlink(missing_ok=True)
            raise FileNotFoundError(f"Could not reach the archive host: {e}") from e
        except BaseException:
            part.unlink(missing_ok=True)
            raise
    try:
        with xr.open_dataset(local) as ds:
            return ds.load()
    except Exception:
        local.unlink(missing_ok=True)   # corrupt → refetch next run
        raise


def load_climatology(var: str, seasonal: bool = False) -> xr.Dataset:
    """Climatology Dataset with {VAR} and {VAR}_STD, dim month|season."""
    name = (f"tcdiag_clim_seasonal__{var}.nc" if seasonal
            else f"tcdiag_clim__{var}.nc")
    return _release_dataset(name)


def load_archive_month(var: str, year: int, month: int) -> xr.DataArray:
    """One monthly-mean field from the per-decade archive files."""
    decade = f"{(year // 10) * 10}s"
    ds = _release_dataset(f"tcdiag__{var}__{decade}.nc")
    da = ds[var] if var in ds else ds[list(ds.data_vars)[0]]
    sel = da.sel(time=f"{year}-{month:02d}")
    if "time" in sel.dims:
        if sel.sizes["time"] == 0:
            raise KeyError(f"{year}-{month:02d} not in archive")
        sel = sel.isel(time=0)
    return sel


def archive_available() -> bool:
    """Is the climatology published? Probes GitHub once per session."""
    if "_have_archive" not in st.session_state:
        try:
            load_climatology("vPI")
            st.session_state["_have_archive"] = True
        except Exception:
            st.session_state["_have_archive"] = False
    return st.session_state["_have_archive"]


# ─────────── live computation via tcpyVPI ────────────────────────────────────
# Levels each input actually needs — U/V/VO only feed level-selected terms
# (200/850 hPa shear, 850 hPa vorticity), so skipping the other 30+ levels
# cuts the THREDDS transfer several-fold. T/Q keep the full PI profile
# (tcpyPI integrates up to ptop=50 hPa; 600 hPa also feeds χ).
_LEVELS_NEEDED = {
    "SSTK": None, "SP": None,
    "T": (50, 1000), "Q": (50, 1000),
    "U": [200, 850], "V": [200, 850], "VO": [850],
}


@st.cache_resource(show_spinner=False, max_entries=8)
def compute_live(year: int, month: int, day: Optional[int], hour: Optional[int],
                 stride: int) -> xr.Dataset:
    """Compute all 7 diagnostics for one time from ERA5 via tcpyVPI.

    day=None → monthly mean (d633001); else 6-hourly snapshot (d633000).
    Inputs are fetched SEQUENTIALLY (netCDF-C is not thread-safe for
    concurrent OPeNDAP opens, and process pools re-import the app in spawn
    workers); per-variable level subsetting keeps each transfer small.
    Cached per (time, stride): switching diagnostics on the same map is free.
    """
    from tcpyVPI.era5_loader import (
        _get_monthly_mean_url, _get_hourly_surface_url, _get_hourly_pressure_url,
    )
    from tcpyVPI import compute_gpiv_from_dataset
    from tcdiag_fetch import fetch_field

    sfc = ["SSTK", "SP"]
    pl = ["T", "Q", "U", "V", "VO"]

    jobs = []
    for v in sfc + pl:
        if day is None:
            url = _get_monthly_mean_url(v, year, is_surface=(v in sfc))
            time_spec = ("index", month - 1)
        else:
            url = (_get_hourly_surface_url(v, year, month) if v in sfc
                   else _get_hourly_pressure_url(v, year, month, day))
            time_spec = ("nearest", f"{year}-{month:02d}-{day:02d}T{hour or 0:02d}:00")
        jobs.append((v, url, time_spec, _LEVELS_NEEDED[v], LAT_MAX, LAT_MIN, stride))

    results = [fetch_field(job) for job in jobs]

    arrays = {}
    for r in results:
        coords = {"latitude": r["latitude"], "longitude": r["longitude"]}
        if "level" in r:
            coords["level"] = r["level"]
        arrays[r["name"]] = xr.DataArray(r["values"], dims=r["dims"],
                                         coords=coords, name=r["name"])

    # join="outer" is safe here: T/Q carry the superset of levels, so U/V/VO
    # just get NaN-padded at levels their formulas never select.
    ds = xr.merge(arrays.values(), join="outer")
    res = compute_gpiv_from_dataset(ds, verbose=False)
    # Scrub ±inf → NaN (VI = VWS·χ/PI blows up where PI→0 at the band edges;
    # same scrub the archive builder applies, so live and archived fields
    # agree — and an inf reaching the colour-bar sliders crashes Streamlit).
    for v in res.data_vars:
        res[v] = res[v].where(np.isfinite(res[v]))
    return res
