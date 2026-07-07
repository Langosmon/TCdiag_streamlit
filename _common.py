"""Shared helpers for the ERA5 Streamlit apps.

This module is duplicated between ERA5_streamlit and ERA5_hourly_streamlit so
each deploys independently to Streamlit Cloud. Keep them in sync.

Provides:
  - SURFACE, PRESSURE, COMMON_PLEVELS variable catalogues
  - coastlines_trace(): fast, no-cartopy coastline overlay
  - find_var(): tolerant ERA5 variable name lookup
  - load_field_cached(): cached remote field loader (slices time/level
    server-side BEFORE downloading)
  - load_climatology() / load_climatology_std(): climatology from GitHub
    Releases, one download + one open per (domain, var, lvl)
  - apply_unit_conversions(): K→°C, Pa→hPa, PV→PVU (anomaly-aware)
  - build_figure(): site-themed Plotly figure
  - add_significance_stipple(): dots where |anom| ≥ z·σ
  - configure_page() / render_footer(): site-branded UI shell

Unit-conversion contract: all loaders return fields in ERA5 NATIVE units
(K, Pa, SI potential vorticity). Convert for display with
apply_unit_conversions() AFTER any anomaly subtraction, so the field and the
climatology are always differenced in the same units.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import streamlit as st
import xarray as xr
import plotly.express as px
import plotly.graph_objects as go


# ─────────── catalogues ───────────────────────────────────────────────────────
# (domain, code, vname, units, cmap_abs, cmap_anom)
SURFACE: dict[str, tuple] = {
    "Sea-surface temperature": ("sfc", "034", "sstk", "°C",     "thermal", "RdBu_r"),
    "CAPE":                    ("sfc", "059", "cape", "J kg⁻¹", "viridis", "PuOr"),
    # NOTE: no "Surface geopotential" — 128_129_z does not exist in
    # e5.moda.an.sfc / e5.oper.an.sfc on RDA (it's an invariant field).
    "Surface pressure":        ("sfc", "134", "sp",   "hPa",    "icefire", "RdBu_r"),
    "Mean sea-level press.":   ("sfc", "151", "msl",  "hPa",    "icefire", "RdBu_r"),
    "10-m zonal wind":         ("sfc", "165", "10u",  "m s⁻¹",  "curl",    "RdBu_r"),
    "10-m meridional wind":    ("sfc", "166", "10v",  "m s⁻¹",  "curl_r",  "RdBu_r"),
    "2-m temperature":         ("sfc", "167", "2t",   "°C",     "thermal", "RdBu_r"),
}

PRESSURE: dict[str, tuple] = {
    "Potential vorticity":   ("pl", "060", "pv", "PVU",     "plasma",  "RdBu_r"),
    "Geopotential":          ("pl", "129", "z",  "m² s⁻²",  "magma",   "RdBu_r"),
    "Temperature":           ("pl", "130", "t",  "°C",      "thermal", "RdBu_r"),
    "Zonal wind":            ("pl", "131", "u",  "m s⁻¹",   "curl",    "RdBu_r"),
    "Meridional wind":       ("pl", "132", "v",  "m s⁻¹",   "curl_r",  "RdBu_r"),
    "Specific humidity":     ("pl", "133", "q",  "kg kg⁻¹", "viridis", "BrBG"),
    # ω is pressure-coordinate vertical velocity: positive = DESCENT.
    "Vertical velocity (ω)": ("pl", "135", "w",  "Pa s⁻¹ · + = descent", "icefire", "RdBu"),
    "Relative vorticity":    ("pl", "138", "vo", "s⁻¹",     "plasma",  "RdBu_r"),
    "Divergence":            ("pl", "155", "d",  "s⁻¹",     "plasma",  "RdBu_r"),
    "Relative humidity":     ("pl", "157", "r",  "%",       "viridis", "BrBG"),
    "Ozone":                 ("pl", "203", "o3", "kg kg⁻¹", "viridis", "RdBu_r"),
}

COMMON_PLEVELS: list[int] = [1000, 975, 925, 850, 700, 500, 300, 250, 100, 50, 10]


# ─────────── REMOTE CLIMATOLOGY ──────────────────────────────────────────────
# Climatology .nc files live on a GitHub Release attached to the ERA5_streamlit
# repo (so both apps can pull from the same place). One release tag holds all
# 96 files. URLs are of the form:
#     {CLIM_BASE}/{tag}/{sfc|pl}__{var}[_<lvl>].nc
# (double-underscore separator because GitHub Releases flatten paths).
# Files store ERA5 NATIVE units — see the module docstring.
CLIM_BASE = "https://github.com/Langosmon/ERA5_streamlit/releases/download"
CLIM_TAG  = "climatology-v1"   # bump if you reupload regenerated files


def _clim_remote_url(domain: str, var: str, lvl: Optional[int]) -> str:
    """URL of a single climatology file in the GitHub Release."""
    if domain == "sfc":
        name = f"sfc__{var}.nc"
    else:
        name = f"pl__{var}_{lvl}.nc"
    return f"{CLIM_BASE}/{CLIM_TAG}/{name}"


def _clim_local_path(domain: str, var: str, lvl: Optional[int]) -> Path:
    """Local cache path for a downloaded climatology file."""
    cache = Path("/tmp/era5-climatology")
    cache.mkdir(parents=True, exist_ok=True)
    return cache / (f"sfc__{var}.nc" if domain == "sfc" else f"pl__{var}_{lvl}.nc")


@st.cache_resource(show_spinner="Fetching climatology…")
def _clim_dataset(domain: str, var: str, lvl: Optional[int]) -> xr.Dataset:
    """Download (once) and open (once) a climatology file, fully loaded into
    memory. Every helper below reads from this single cached Dataset, so
    reruns never re-open the netCDF."""
    local = _clim_local_path(domain, var, lvl)

    if not local.exists():
        url = _clim_remote_url(domain, var, lvl)
        # Download to a temp name and rename atomically so an interrupted
        # transfer can never leave a truncated file that poisons the cache.
        part = local.with_suffix(".part")
        try:
            r = requests.get(url, stream=True, timeout=30)
            if r.status_code != 200:
                raise FileNotFoundError(
                    f"Climatology not yet uploaded to release '{CLIM_TAG}'.\n"
                    f"URL: {url}\nStatus: {r.status_code}"
                )
            with open(part, "wb") as f:
                for chunk in r.iter_content(1 << 16):
                    f.write(chunk)
            part.replace(local)
        except requests.exceptions.RequestException as e:
            part.unlink(missing_ok=True)
            raise FileNotFoundError(f"Could not reach climatology host: {e}") from e
        except BaseException:
            part.unlink(missing_ok=True)
            raise

    try:
        with xr.open_dataset(local) as ds:
            return ds.load()
    except Exception:
        # Corrupt file (e.g. legacy partial download) — drop it so the next
        # run re-fetches instead of failing forever.
        local.unlink(missing_ok=True)
        raise


def load_climatology(domain: str, var: str, lvl: Optional[int]) -> xr.DataArray:
    """Climatology mean in NATIVE units, already in memory — slicing by
    month is free."""
    ds = _clim_dataset(domain, var, lvl)
    da = ds[find_var(ds, var)]
    # Guard the native-units contract: a climatology regenerated in display
    # units would silently reintroduce the K-vs-°C offset bug.
    u = str(da.attrs.get("units", "")).strip().lower()
    if u in {"degc", "celsius", "°c", "hpa", "pvu"}:
        raise ValueError(
            f"Climatology for '{var}' is in converted units ({u!r}); the apps "
            "expect ERA5 native units (K, Pa, SI). Rebuild with "
            "tools/build_climatology.py without unit conversion."
        )
    return da


def _find_std_name(ds: xr.Dataset) -> Optional[str]:
    for k in ds.variables:
        if "std" in k.lower() or "stddev" in k.lower() or "sigma" in k.lower():
            return k
    return None


def climatology_has_std(domain: str, var: str, lvl: Optional[int]) -> bool:
    """True if the climatology file includes a std-dev variable (needed for
    statistical-significance tests)."""
    try:
        return _find_std_name(_clim_dataset(domain, var, lvl)) is not None
    except Exception:
        return False


def load_climatology_std(domain: str, var: str, lvl: Optional[int]) -> xr.DataArray:
    """The std-dev DataArray from the climatology file (NATIVE units)."""
    ds = _clim_dataset(domain, var, lvl)
    name = _find_std_name(ds)
    if name is None:
        raise KeyError("std variable not found in climatology file")
    return ds[name]


# ─────────── data access ─────────────────────────────────────────────────────
def find_var(ds: xr.Dataset, short: str) -> str:
    """Return the actual variable name in a dataset for an ERA5 short code.
    Tolerant of VAR_ prefixes and 10 ↔ 10M substitutions."""
    up = short.upper()
    for k in (up, f"VAR_{up}", up.replace("10", "10M")):
        if k in ds:
            return k
    raise KeyError(short)


@st.cache_resource(show_spinner="Loading from RDA…", max_entries=6)
def load_field_cached(url: str, vname: str, plevel: Optional[int],
                      decode_times: bool = True,
                      time_sel: Optional[str] = None,
                      day_sel: Optional[str] = None,
                      stride: int = 1) -> xr.DataArray:
    """Open a remote dataset, slice the requested variable — and, crucially,
    the pressure level and/or time — BEFORE downloading, then eagerly load.

    OPeNDAP translates the .sel()/.isel() calls into server-side subsetting,
    so only the requested slab crosses the network:
      - monthly app: one year of one variable (~50 MB) → month scrubbing on
        the same year is instant in-memory slicing.
      - hourly app: pass `time_sel` (ISO string) to fetch a single hour
        (~4 MB) instead of a whole month of hourly data (~3 GB).
      - animation: pass `day_sel` (ISO date) for all 24 hours of one day,
        with `stride=2` to fetch at 0.5° (~25 MB instead of ~100 MB).

    Cached with st.cache_resource: the array is shared across reruns and
    sessions with NO per-rerun pickle/unpickle copy (all downstream ops
    produce new arrays, nothing mutates in place). max_entries=6 bounds
    worst-case memory at ~6 × 50 MB. No TTL — reanalysis is immutable.
    """
    ds = xr.open_dataset(url, decode_times=decode_times)
    da = ds[find_var(ds, vname)]
    if plevel is not None:
        da = da.sel(level=plevel)
    if day_sel is not None:
        da = da.sel(time=slice(f"{day_sel}T00:00", f"{day_sel}T23:59"))
    elif time_sel is not None:
        da = da.sel(time=time_sel, method="nearest")
    if stride > 1:
        da = da.isel(latitude=slice(None, None, stride),
                     longitude=slice(None, None, stride))
    return da.load().astype(np.float32)


# ─────────── ERA5 LAND-SEA MASK ──────────────────────────────────────────────
# Static field, fetched once from RDA invariants. Values: 1=land, 0=sea,
# fractional at coasts (≥0.5 is ECMWF's standard land definition).
LSM_URL = ("https://thredds.rda.ucar.edu/thredds/dodsC/files/g/d633000/"
           "e5.oper.invariant/197901/e5.oper.invariant.128_172_lsm.ll025sc."
           "1979010100_1979010100.nc")


@st.cache_resource(show_spinner="Fetching land-sea mask…")
def load_lsm() -> xr.DataArray:
    """Load ERA5 land-sea mask (0=sea, 1=land, fractional coasts).
    On the same 0.25° grid as the data, so .where() works directly."""
    ds = xr.open_dataset(LSM_URL)
    da = ds[find_var(ds, "lsm")].squeeze()
    return da.load()


# Memo of the mask re-aligned to each distinct data grid (keyed by the grid's
# fingerprint) so reruns don't repeat the reindex.
_LSM_ALIGNED: dict[tuple, xr.DataArray] = {}


def apply_lsm_mask(da: xr.DataArray, mode: str) -> xr.DataArray:
    """Apply a land-sea mask to a DataArray.  mode is one of:
       "All" / "Land" / "Ocean".  Returns the masked DataArray."""
    if mode == "All":
        return da
    lat, lon = da.latitude.values, da.longitude.values
    key = (float(lat[0]), float(lat[-1]), lat.size,
           float(lon[0]), float(lon[-1]), lon.size)
    lsm_aligned = _LSM_ALIGNED.get(key)
    if lsm_aligned is None:
        lsm = load_lsm()
        try:
            lsm_aligned = lsm.reindex_like(da, method="nearest", tolerance=0.01)
        except Exception:
            lsm_aligned = lsm  # fall through; .where() broadcasts best-effort
        _LSM_ALIGNED[key] = lsm_aligned
    if mode == "Land":
        return da.where(lsm_aligned >= 0.5)
    if mode == "Ocean":
        return da.where(lsm_aligned < 0.5)
    return da


# ─────────── REGION-AWARE COLORBAR AUTOSCALE ─────────────────────────────────
# When the user box-selects a region or picks a preset, recompute the colorbar
# from the 98% quantile of data INSIDE that box. Solves the "Andes/Himalayas
# spikes dominate the scale, ITCZ looks washed out" problem.
def rescale_to_region(da: xr.DataArray,
                      lat_min: float, lat_max: float,
                      lon_min: float, lon_max: float,
                      quantile_lo: float = 0.01,
                      quantile_hi: float = 0.99,
                      symmetric: bool = False) -> tuple[float, float]:
    """Return (cmin, cmax) covering quantile_lo..quantile_hi of the data
    inside the lat/lon box.

    Longitude handling: bounds may be given in either -180..180 or 0..360;
    they are mapped onto the data's own convention. Boxes that wrap across
    Greenwich or the antimeridian (lon_min > lon_max after mapping) select
    the union (lon ≥ lon_min) OR (lon ≤ lon_max). A box spanning the full
    circle skips the longitude filter entirely.

    symmetric=True returns ±max(|q_lo|,|q_hi|) — use for anomaly fields.
    Falls back to global quantiles if the box is empty."""
    lats = da.latitude.values
    lons = da.longitude.values
    vals = da.values

    lat_mask = (lats >= lat_min) & (lats <= lat_max)

    full_circle = (lon_max - lon_min) >= 359.5
    if full_circle:
        lon_mask = np.ones(lons.size, dtype=bool)
    else:
        if lons.max() > 180:            # data on 0..360
            lon_min, lon_max = lon_min % 360, lon_max % 360
        else:                            # data on -180..180
            lon_min = ((lon_min + 180) % 360) - 180
            lon_max = ((lon_max + 180) % 360) - 180
        if lon_min <= lon_max:
            lon_mask = (lons >= lon_min) & (lons <= lon_max)
        else:                            # box wraps the seam
            lon_mask = (lons >= lon_min) | (lons <= lon_max)

    # Ellipsis indexing keeps this correct for both 2-D (lat, lon) fields and
    # 3-D (time, lat, lon) day cubes.
    if lat_mask.any() and lon_mask.any():
        sub = vals[..., lat_mask, :][..., :, lon_mask]
    else:
        sub = vals
    if sub.size == 0 or np.all(np.isnan(sub)):
        sub = vals
    qlo, qhi = np.nanquantile(sub, [quantile_lo, quantile_hi])
    if symmetric:
        m = max(abs(float(qlo)), abs(float(qhi)))
        return -m, m
    return float(qlo), float(qhi)


# Quick-pick region presets — common atmospheric/ocean basins.
# Stored as (lat_min, lat_max, lon_min, lon_max) in -180..180 longitude;
# rescale_to_region maps them onto the data's own convention.
REGION_PRESETS: dict[str, tuple[float, float, float, float]] = {
    "Global":         (-90, 90, -180, 180),
    "Tropics":        (-30, 30, -180, 180),
    "ITCZ band":      (-15, 15, -180, 180),
    "E Pacific (ENP)": (5, 30, -130, -85),
    "Atlantic":        (5, 35, -85, -10),
    "W Pacific":      (5, 35, 110, 180),
    "Indian Ocean":  (-30, 30, 40, 105),
    "N America":     (15, 70, -170, -50),
    "Europe":         (35, 72, -15, 50),
}


# ─────────── coastlines (no cartopy) ─────────────────────────────────────────
@st.cache_resource
def coastlines_trace() -> go.Scatter:
    path = Path(__file__).with_name("coastlines.json")
    data = json.loads(path.read_text())
    return go.Scatter(
        x=data["xs"], y=data["ys"],
        mode="lines",
        # Mid-gray with alpha: readable on light AND dark themes, and doesn't
        # out-shout a diverging colormap's neutral center.
        line=dict(color="rgba(96,102,114,0.55)", width=0.6),
        hoverinfo="skip",
        showlegend=False,
        name="coast",
    )


# ─────────── unit conversions ────────────────────────────────────────────────
def apply_unit_conversions(da: xr.DataArray, vname: str, units: str,
                           anomaly: bool = False) -> tuple[xr.DataArray, str]:
    """Convert a NATIVE-units field for display. Call AFTER any anomaly
    subtraction (see module docstring).

    anomaly=True skips offset-type conversions that cancel in a difference
    (K→°C is a pure shift, so a K anomaly already IS a °C anomaly) but keeps
    scale-type conversions (Pa→hPa, PV→PVU), which also apply to std devs."""
    if vname in {"sstk", "2t", "t"}:
        if not anomaly:
            da = da - 273.15
        units = "°C"
    if vname in {"sp", "msl"}:
        da = da / 100.0
        units = "hPa"
    if vname == "pv":
        da = da * 1.0e6          # 1 PVU = 1e-6 K m² kg⁻¹ s⁻¹
        units = "PVU"
    return da, units


# ─────────── colour-bar controls ─────────────────────────────────────────────
def colourbar_controls(da: xr.DataArray, show_anom: bool,
                      override_default: Optional[tuple[float, float]] = None,
                      override_label: Optional[str] = None):
    """Sidebar sliders + auto-scale button for the colour-bar.

    `override_default`: if provided, these (cmin, cmax) become the slider
    defaults — used by the region-preset picker and the box-select rescale.
    The slider RANGE still spans the full data so the user can pan beyond
    the preset.
    `override_label`: optional small caption shown under the controls."""
    arr = da.values
    if np.all(np.isnan(arr)):
        # Fully masked field (e.g. SST + Land) — the app already explains
        # why the map is empty; give the figure sane dummy bounds.
        st.sidebar.info("Colour-bar disabled — the current mask leaves no data.")
        return 0.0, 1.0
    data_min = float(np.nanmin(arr))
    data_max = float(np.nanmax(arr))

    if override_default is not None:
        default_min, default_max = float(override_default[0]), float(override_default[1])
    elif show_anom:
        m = max(abs(data_min), abs(data_max))
        default_min, default_max = -m, m
    else:
        default_min, default_max = data_min, data_max

    # Slider widget range covers the entire data PLUS the override so the
    # default is always inside the slider range.
    slider_min = min(data_min, default_min)
    slider_max = max(data_max, default_max)
    step = (slider_max - slider_min) / 50 or 1e-6

    # Streamlit keeps a keyed widget's stored value and IGNORES a changed
    # `value=` parameter — so box-select / region-preset rescales would
    # silently do nothing. Instead the sliders are driven purely through
    # session state: whenever the thing that determines the defaults changes
    # (new box, new preset, new field), push the new defaults in.
    fp = ("override", default_min, default_max) if override_default is not None \
         else ("data", data_min, data_max, bool(show_anom))
    if st.session_state.get("_cbar_fp") != fp:
        st.session_state["_cbar_fp"] = fp
        st.session_state["cmin"] = default_min
        st.session_state["cmax"] = default_max
    # Clamp stale values (e.g. from a previous variable) into today's range.
    st.session_state["cmin"] = min(max(st.session_state.get("cmin", default_min), slider_min), slider_max)
    st.session_state["cmax"] = min(max(st.session_state.get("cmax", default_max), slider_min), slider_max)

    if override_label:
        st.sidebar.caption(f"🎯 {override_label}")

    with st.sidebar.expander("Colour-bar limits", expanded=False):
        cmin = st.slider("Min", slider_min, slider_max,
                         step=step, format="%.4g", key="cmin")
        cmax = st.slider("Max", slider_min, slider_max,
                         step=step, format="%.4g", key="cmax")
        if st.button("Auto-scale (98 % of data)", use_container_width=True):
            qmin, qmax = np.nanquantile(arr, [0.01, 0.99])
            # Keep _cbar_fp as-is so the write survives the rerun.
            st.session_state["cmin"] = float(qmin)
            st.session_state["cmax"] = float(qmax)
            st.rerun()

    if cmin >= cmax:
        st.sidebar.error("Min must be less than Max")
        st.stop()
    return cmin, cmax


# ─────────── region picker (sidebar) ─────────────────────────────────────────
def region_picker():
    """Render the region preset selector. Returns the chosen region tuple
    (lat_min, lat_max, lon_min, lon_max) or None for Global / no rescale."""
    st.sidebar.header("Region focus")
    name = st.sidebar.selectbox(
        "Rescale colour-bar to region",
        list(REGION_PRESETS.keys()),
        index=0,
        help="Tunes the colour-bar to the 98% quantile of data inside the "
             "chosen region. The map still shows the whole field — values "
             "outside this range are clipped to the bar ends. "
             "Hint: use Plotly's box-select tool to define a custom region.",
        key="region_name",
    )
    if name == "Global":
        return None, name
    return REGION_PRESETS[name], name


def box_selection_to_bounds(event) -> Optional[tuple[float, float, float, float]]:
    """Extract (lat_min, lat_max, lon_min, lon_max) from a Streamlit
    plotly_chart selection event/state. Returns None if no box."""
    if not event:
        return None
    sel = event.get("selection") if hasattr(event, "get") else None
    if not sel:
        return None
    boxes = sel.get("box") or []
    if not boxes:
        return None
    b = boxes[0]
    xs = b.get("x") or []
    ys = b.get("y") or []
    if len(xs) < 2 or len(ys) < 2:
        return None
    return (float(min(ys)), float(max(ys)), float(min(xs)), float(max(xs)))


# ─────────── statistical significance ────────────────────────────────────────
def add_significance_stipple(fig: go.Figure, anom: xr.DataArray,
                             std: xr.DataArray, z: float = 2.04,
                             stride: int = 8) -> None:
    """Overlay sparse dots where |anom| ≥ z·σ. Default z = t(0.975, 30) ≈ 2.04
    (≈95% confidence for a 31-year base period, assuming year-to-year
    normality). Subsamples BEFORE comparing, so the test runs on 1/stride² of
    the grid — same visual output, ~64× less work at stride=8.
    Pass anom and std in the SAME units (both native or both converted)."""
    sub_a = anom.values[::stride, ::stride]
    sub_s = std.values[::stride, ::stride]
    sig = np.abs(sub_a) >= z * np.abs(sub_s)
    ys, xs = np.where(sig)
    lats = anom.latitude.values[::stride]
    lons = anom.longitude.values[::stride]
    fig.add_trace(go.Scatter(
        x=lons[xs], y=lats[ys],
        mode="markers",
        marker=dict(size=2, color="rgba(96,102,114,0.6)", symbol="circle"),
        hoverinfo="skip", showlegend=False, name=f"sig (|z|≥{z})",
    ))


# ─────────── figure builder ──────────────────────────────────────────────────
def build_figure(da: xr.DataArray, title: str, units: str, cmap: str,
                 cmin: float, cmax: float, show_coast: bool,
                 height: int = 560) -> go.Figure:
    accent_text, _ = _theme_colors()
    fig = px.imshow(
        da,
        origin="lower",
        aspect="auto",
        color_continuous_scale=cmap,
        labels=dict(color=units),
        title=title,
    )
    fig.update_coloraxes(cmin=cmin, cmax=cmax,
                         colorbar=dict(thickness=12, len=0.85, x=1.01,
                                       outlinewidth=0))
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=46, b=0),
        uirevision="keep",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(
            text=title,
            x=0.0, xanchor="left",
            font=dict(size=18, color=accent_text,
                      family="Source Serif 4, Georgia, serif"),
        ),
        xaxis=dict(title="", showgrid=False, zeroline=False, ticks="outside"),
        yaxis=dict(title="", showgrid=False, zeroline=False, ticks="outside"),
    )
    if show_coast:
        fig.add_trace(coastlines_trace())
    return fig


# ─────────── day animation ───────────────────────────────────────────────────
def build_animation_figure(da_day: xr.DataArray, title: str, units: str,
                           cmap: str, cmin: float, cmax: float,
                           show_coast: bool, height: int = 580) -> go.Figure:
    """24-frame animation of one day with client-side play/pause + hour slider.

    Values are colormapped server-side through a 256-entry LUT and shipped as
    PNG-compressed RGB frames (px.imshow binary_string), so a full day is a
    few MB and playback never touches the server. An invisible scatter trace
    carries the colorbar. Hover shows no data values in this mode — that's
    the trade for a payload 20× smaller than raw arrays."""
    import plotly.colors as pc

    times = da_day.time.values
    lats = da_day.latitude.values
    lons = da_day.longitude.values
    vals = da_day.values                      # (time, lat, lon)

    # go.Image draws row i at y0 + i*dy — flip to ascending latitude so a
    # positive dy puts north at the top on a normal (increasing-up) axis.
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        vals = vals[:, ::-1, :]

    colorscale = pc.get_colorscale(cmap)
    samples = pc.sample_colorscale(colorscale, np.linspace(0, 1, 256),
                                   colortype="tuple")
    lut = (np.asarray(samples) * 255).astype(np.uint8)         # (256, 3)

    span = (cmax - cmin) or 1e-9
    idx = (vals - cmin) / span * 255.0
    bad = ~np.isfinite(idx)
    idx8 = np.clip(np.where(bad, 0, idx), 0, 255).astype(np.uint8)
    rgb = lut[idx8]                                            # (t, lat, lon, 3)
    rgb[bad] = 235                                             # masked → light gray

    fig = px.imshow(rgb, animation_frame=0, binary_string=True,
                    binary_compression_level=6)

    # Put the frames back into lon/lat coordinates so coastlines align.
    x0, dx = float(lons[0]), float(lons[1] - lons[0])
    y0, dy = float(lats[0]), float(lats[1] - lats[0])
    fig.update_traces(x0=x0, dx=dx, y0=y0, dy=dy)
    for fr in fig.frames:
        for tr in fr.data:
            tr.update(x0=x0, dx=dx, y0=y0, dy=dy)
    fig.update_xaxes(autorange=True, title="", showgrid=False,
                     zeroline=False, ticks="outside", constrain="domain")
    fig.update_yaxes(autorange=True, title="", showgrid=False,
                     zeroline=False, ticks="outside")

    # Human labels on the frames + slider ("06:00 UTC", not "5").
    labels = [np.datetime_as_string(tv, unit="h")[-2:] + ":00 UTC" for tv in times]
    for fr, lab in zip(fig.frames, labels):
        fr.name = lab
    if fig.layout.sliders:
        sl = fig.layout.sliders[0]
        sl.currentvalue.prefix = ""
        for step_, lab in zip(sl.steps, labels):
            step_.label = lab[:5]
            step_.args = ([lab], step_.args[1])   # args tuples are immutable

    # Faster default playback; no easing between frames.
    if fig.layout.updatemenus:
        btn = fig.layout.updatemenus[0].buttons[0]
        btn.args[1]["frame"]["duration"] = 180
        btn.args[1]["transition"]["duration"] = 0

    if show_coast:
        fig.add_trace(coastlines_trace())

    # Invisible scatter that carries the colorbar for the PNG frames.
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(colorscale=colorscale, cmin=cmin, cmax=cmax,
                    showscale=True,
                    colorbar=dict(thickness=12, len=0.85, x=1.01,
                                  outlinewidth=0, title=dict(text=units))),
        hoverinfo="skip", showlegend=False,
    ))

    accent_text, _ = _theme_colors()
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=46, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text=title, x=0.0, xanchor="left",
                   font=dict(size=18, color=accent_text,
                             family="Source Serif 4, Georgia, serif")),
    )
    return fig


# ─────────── presentation ────────────────────────────────────────────────────
SITE_URL = "https://langosmon.github.io"


def _theme_colors() -> tuple[str, str]:
    """(accent_text, muted) for the ACTIVE theme, chosen to pass WCAG AA.
    Falls back to the light pair on Streamlit versions without st.context.theme."""
    try:
        dark = st.context.theme.type == "dark"
    except Exception:
        dark = False
    if dark:
        return "#e08c66", "#9aa0a6"   # ≥ 5:1 on Streamlit's dark background
    return "#a85028", "#5f6368"       # ≥ 4.5:1 on white / warm paper

_FONT_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  html, body, .stApp, [data-testid="stSidebar"] *, .stMarkdown, button, input, select {
    font-family: "Inter", -apple-system, "Segoe UI", sans-serif;
  }
  h1, h2, h3 { font-family: "Source Serif 4", Georgia, serif; }
  code, pre, [data-testid="stMetricValue"] {
    font-family: "JetBrains Mono", ui-monospace, monospace;
  }
  /* Streamlit renders its UI icons (expander chevrons, etc.) as Material
     Symbols ligatures — the blanket font rule above must NOT touch them,
     or the icons degrade to raw text like "keyboard_double_arrow_right". */
  [data-testid="stIconMaterial"],
  span[class*="material-symbols"] {
    font-family: "Material Symbols Rounded", "Material Symbols Outlined" !important;
  }
</style>
"""


def configure_page(title: str, subtitle: str | None = None,
                   icon: str = "🌀") -> None:
    st.set_page_config(
        page_title=f"{title} · Ocegueda Sanchez",
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get help": SITE_URL,
            "About": f"{title} — Jose A. Ocegueda Sanchez ({SITE_URL}). "
                     f"ERA5 data via NCAR's RDA. Source on GitHub.",
        },
    )
    st.markdown(_FONT_CSS, unsafe_allow_html=True)
    accent_text, muted = _theme_colors()
    st.markdown(
        f"""
        <div style='display:flex; justify-content:space-between; align-items:baseline;
                    border-bottom:1px solid rgba(128,128,128,0.18); padding-bottom:6px;
                    margin-bottom:8px;'>
          <div>
            <div style='font-family:"JetBrains Mono",monospace; font-size:11px;
                        letter-spacing:0.16em; text-transform:uppercase;
                        color:{accent_text};'>ERA5 · NCAR RDA</div>
            <div style='font-family:"Source Serif 4",Georgia,serif; font-size:24px;
                        line-height:1.1; margin-top:4px;'>{title}</div>
            {('<div style="font-size:13px; color:' + muted + '; margin-top:4px;">'
              f'{subtitle}</div>') if subtitle else ''}
          </div>
          <div style='font-family:"JetBrains Mono",monospace; font-size:10px;
                      letter-spacing:0.08em; color:{muted};'>
            <a href='{SITE_URL}' target='_blank' style='color:{accent_text};
                text-decoration:none;'>langosmon.github.io ↗</a>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer(repo_url: str) -> None:
    accent_text, muted = _theme_colors()
    st.markdown(
        f"""
        <div style='margin-top:32px; padding-top:14px;
                    border-top:1px solid rgba(128,128,128,0.18);
                    font-size:12px; color:{muted}; display:flex; gap:16px;
                    flex-wrap:wrap; justify-content:space-between;'>
          <span>Plot built from ERA5 via NCAR's
            <a href='https://rda.ucar.edu/' target='_blank' style='color:{accent_text};'>RDA</a>.
          </span>
          <span>
            <a href='{repo_url}' target='_blank' style='color:{accent_text}; text-decoration:none;'>source ↗</a>
            ·
            <a href='{SITE_URL}' target='_blank' style='color:{accent_text}; text-decoration:none;'>site ↗</a>
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────── reusable sidebar widget ─────────────────────────────────────────
def variable_picker():
    st.sidebar.header("Field")
    field_type = st.sidebar.radio("Domain", ("Surface", "Pressure level"),
                                  horizontal=True, key="field_type")
    if field_type == "Surface":
        choice = st.sidebar.selectbox("Variable", list(SURFACE), key="surface_var")
        domain, code, vname, units, cmap_abs, cmap_anom = SURFACE[choice]
        plevel = None
    else:
        choice = st.sidebar.selectbox("Variable", list(PRESSURE), key="pressure_var")
        domain, code, vname, units, cmap_abs, cmap_anom = PRESSURE[choice]
        plevel = st.sidebar.selectbox("Pressure level (hPa)", COMMON_PLEVELS,
                                      index=COMMON_PLEVELS.index(500),
                                      key="plevel")
    return choice, domain, code, vname, units, cmap_abs, cmap_anom, plevel
