"""TC Diagnostics — vPI, PI, Ventilation Index, GPIv from ERA5.

Built on the lab's tcpyVPI package (Chavas, Camargo & Tippett 2025, J. Clim.;
github.com/drchavas/tcpyVPI). Three modes:

  Climatology   1980–2022 monthly + seasonal (JJA/SON/JJASON) means and
                interannual stds — precomputed, loads instantly.
  Monthly       any month 1980–2022, with anomalies vs the climatology —
                precomputed archive, live-computed fallback.
  Daily (6-h)   any 00/06/12/18 UTC snapshot, computed live from ERA5 via
                RDA THREDDS (~1–2 min first time, then cached), with
                anomalies vs the monthly climatology.

Domain: 40°S–40°N. Archive resolution is set by the published release
(v1 is 1°; a 0.5° rebuild on the cluster can replace it as a new tag);
live computation is 1° by default with a 0.5° option.
"""

import datetime
import numpy as np
import streamlit as st

import _common as C
import tcdiag_data as D


# ─── page setup ───────────────────────────────────────────────────────────────
C.configure_page(
    title="TC Diagnostics · vPI / GPIv",
    subtitle="Ventilated potential intensity and genesis potential, 1980–2022. "
             "Powered by tcpyVPI (Chavas, Camargo & Tippett 2025).",
    icon="🌪️",
)

REPO_URL = "https://github.com/Langosmon/TCdiag_streamlit"
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

HAVE_ARCHIVE = D.archive_available()


# ─── sidebar: diagnostic ─────────────────────────────────────────────────────
st.sidebar.header("Diagnostic")
choice = st.sidebar.selectbox("Field", list(D.DIAGNOSTICS), key="diag")
var, units, cmap_abs, cmap_anom, quantile_default = D.DIAGNOSTICS[choice]

st.sidebar.header("Mode")
mode = st.sidebar.radio(
    "Time aggregation",
    ("Climatology", "Monthly", "Daily (6-hourly)"),
    help="Climatology and Monthly load from the precomputed 1980–2022 "
         "archive. Daily maps are computed live from ERA5 (~1–2 min the "
         "first time, then cached).",
)

show_anom = False
show_std = False
da = None
title = choice
loaded_note = None

# ─── mode: climatology ───────────────────────────────────────────────────────
if mode == "Climatology":
    if not HAVE_ARCHIVE:
        st.info(
            "**The 1980–2022 archive hasn't been published yet.** "
            "Run `tools/build_archive.py` on the cluster and upload the "
            "release (see `tools/README.md`). Until then, the Monthly and "
            "Daily modes still work — they compute live from ERA5."
        )
        st.stop()
    kind = st.sidebar.radio("Period", ("Month", "Season"), horizontal=True)
    show_std = st.sidebar.toggle(
        "Show interannual σ", value=False,
        help="Standard deviation across the 43 years instead of the mean.")
    try:
        if kind == "Month":
            mon = st.sidebar.selectbox("Month", range(1, 13),
                                       format_func=lambda m: MONTH_NAMES[m - 1],
                                       index=8)
            ds = D.load_climatology(var, seasonal=False)
            key = f"{var}_STD" if show_std and f"{var}_STD" in ds else var
            da = ds[key].sel(month=mon)
            title += f" · {MONTH_NAMES[mon - 1]} climatology (1980–2022)"
        else:
            season = st.sidebar.selectbox("Season", D.SEASONS, index=2)
            ds = D.load_climatology(var, seasonal=True)
            key = f"{var}_STD" if show_std and f"{var}_STD" in ds else var
            da = ds[key].sel(season=season)
            title += f" · {season} climatology (1980–2022)"
        if show_std:
            if key.endswith("_STD"):
                title += " · σ"
            else:
                st.warning("This climatology file has no σ variable — "
                           "showing the mean instead.")
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

# ─── mode: monthly ───────────────────────────────────────────────────────────
elif mode == "Monthly":
    col_y, col_m = st.sidebar.columns(2)
    yr = col_y.selectbox("Year", D.YEARS, index=len(D.YEARS) - 1)
    mon = col_m.selectbox("Month", range(1, 13),
                          format_func=lambda m: MONTH_NAMES[m - 1], index=8)
    show_anom = st.sidebar.toggle(
        "Anomaly (vs 1980–2022 climatology)", value=False,
        disabled=not HAVE_ARCHIVE,
        help=None if HAVE_ARCHIVE else "Needs the published climatology.")
    title += f" · {MONTH_NAMES[mon - 1]} {yr}"

    try:
        da = D.load_archive_month(var, yr, mon)
        loaded_note = "archive"
    except Exception:
        # Persist the live-compute choice: an ephemeral button would blank
        # the app on the next widget interaction (every rerun re-raises the
        # archive miss and st.stop()s before the map).
        if st.session_state.get("live_month") != (yr, mon):
            st.info(
                f"**{MONTH_NAMES[mon - 1]} {yr} isn't in the published archive"
                f"{'' if HAVE_ARCHIVE else ' (none published yet)'}.** "
                "It can be computed live from ERA5 instead (about a minute, "
                "then cached)."
            )
            if st.button("🔄 Compute this month live from ERA5", type="primary"):
                st.session_state["live_month"] = (yr, mon)
                st.rerun()
            st.stop()
        with st.status("Computing from ERA5 via RDA…", expanded=False) as s:
            res = D.compute_live(yr, mon, None, None, stride=4)
            s.update(label="Computed (cached for this month).", state="complete")
        da = res[var]
        loaded_note = "live (1°)"

# ─── mode: daily 6-hourly ────────────────────────────────────────────────────
else:
    col_d, col_h = st.sidebar.columns([2, 1])
    sel_date = col_d.date_input(
        "Date (UTC)", value=datetime.date(2023, 10, 25),
        min_value=datetime.date(1940, 1, 1),
        max_value=datetime.date.today() - datetime.timedelta(days=6),
        help="NCAR RDA can lag ERA5 by weeks; very recent dates may not "
             "exist yet and will show a friendly error.")
    sel_hour = col_h.selectbox("Hour", [0, 6, 12, 18], index=1)
    res_choice = st.sidebar.radio("Resolution", ("1° (fast)", "0.5° (slower)"),
                                  horizontal=True)
    stride = 4 if res_choice.startswith("1°") else 2
    show_anom = st.sidebar.toggle(
        "Anomaly (vs monthly climatology)", value=False,
        disabled=not HAVE_ARCHIVE,
        help="Departure from the 1980–2022 mean of the calendar month. "
             "Instantaneous snapshots retain diurnal + synoptic variability."
             if HAVE_ARCHIVE else "Needs the published climatology.")
    title += f" · {sel_date} {sel_hour:02d} UTC"

    with st.status(f"Loading {sel_date} {sel_hour:02d} UTC…", expanded=False) as s:
        st.write("First view of a date pulls ~30–60 MB from NCAR RDA; "
                 "afterwards it's cached and instant.")
        try:
            res = D.compute_live(sel_date.year, sel_date.month, sel_date.day,
                                 sel_hour, stride)
        except Exception as e:
            s.update(label="Failed to load ERA5 inputs.", state="error")
            st.error("**Could not compute this map.** The RDA server may be "
                     "temporarily unreachable, or this date may not exist.")
            with st.expander("Technical details"):
                st.exception(e)
            st.stop()
        s.update(label=f"{sel_date} {sel_hour:02d} UTC ready.", state="complete")
    da = res[var]
    loaded_note = f"live ({'1°' if stride == 4 else '0.5°'})"

# ─── anomaly ─────────────────────────────────────────────────────────────────
cmap = cmap_abs
if show_anom and da is not None:
    try:
        clim = D.load_climatology(var, seasonal=False)
        clim_month = clim[var].sel(month=mon if mode == "Monthly" else sel_date.month)
        # Align grids: archive and live fields may differ in resolution
        # (bilinear, not nearest — nearest would checkerboard-NaN when the
        # grids don't share points). Pad the longitude seam cyclically so
        # interp doesn't leave an all-NaN column at the last source lon.
        if clim_month.sizes != da.sizes:
            import xarray as xr
            wrap = clim_month.isel(longitude=0)
            wrap = wrap.assign_coords(longitude=wrap.longitude + 360.0)
            clim_month = xr.concat([clim_month, wrap], dim="longitude")
            clim_month = clim_month.interp_like(da)
        da = da - clim_month
        cmap = cmap_anom
        units = (units + " " if units != "–" else "") + "anomaly"
        title += " · anomaly"
    except ImportError as e:
        st.error(f"Anomaly regridding needs scipy — deployment bug, not a "
                 f"data problem: {e}")
        show_anom = False
    except Exception as e:
        st.warning(f"Climatology unavailable — showing the full field.\n\n{e}")
        show_anom = False

# ─── display controls ────────────────────────────────────────────────────────
show_coast = st.sidebar.toggle("Coastlines", value=True)

region_bbox, region_name = C.region_picker()

state_box = C.box_selection_to_bounds(st.session_state.get("main_plot"))
if state_box is not None and state_box != st.session_state.get("_dismissed_box"):
    st.session_state["_last_box"] = state_box
last_box = st.session_state.get("_last_box")

override_default = None
override_label = None
if last_box is not None:
    lat_min, lat_max, lon_min, lon_max = last_box
    override_default = C.rescale_to_region(da, lat_min, lat_max, lon_min, lon_max,
                                           symmetric=show_anom)
    override_label = (f"Tuned to box: {lat_min:.1f}–{lat_max:.1f}°N, "
                      f"{lon_min:.1f}–{lon_max:.1f}°E")
elif region_bbox is not None:
    lat_min, lat_max, lon_min, lon_max = region_bbox
    override_default = C.rescale_to_region(da, lat_min, lat_max, lon_min, lon_max,
                                           symmetric=show_anom)
    override_label = f"Tuned to 98% of data in: {region_name}"
elif quantile_default or show_anom:
    vals = da.values
    qlo, qhi = np.nanquantile(vals, [0.01, 0.99])
    if show_anom:
        m = max(abs(float(qlo)), abs(float(qhi)))
        override_default = (-m, m)
    else:
        override_default = (float(qlo), float(qhi))
    override_label = "Default: 98% of data (heavy-tailed field)"

cmin, cmax = C.colourbar_controls(da, show_anom,
                                  override_default=override_default,
                                  override_label=override_label)

# ─── figure ──────────────────────────────────────────────────────────────────
fig = C.build_figure(da, title, units, cmap, cmin, cmax, show_coast, height=440)

st.plotly_chart(
    fig, use_container_width=True,
    on_select="rerun", selection_mode=("box",),
    key="main_plot",
    config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d"]},
)

meta = []
if loaded_note: meta.append(f"source: {loaded_note}")
meta.append("domain: 40°S–40°N")
meta.append("tcpyVPI · Chavas, Camargo & Tippett (2025, J. Clim.)")
st.caption(" · ".join(meta))

col_t, col_r = st.columns([4, 1])
with col_t:
    st.caption(
        "💡 **Box-select** on the map rescales the colour-bar to the 98% "
        "quantile of that region."
    )
with col_r:
    if last_box is not None and st.button("Reset region", use_container_width=True):
        st.session_state["_dismissed_box"] = last_box
        st.session_state["_last_box"] = None
        st.rerun()

with st.expander("What am I looking at?", expanded=False):
    st.markdown(
        "- **PI** — maximum potential intensity a TC could theoretically reach "
        "(Emanuel; computed with [tcpyPI](https://github.com/dgilford/tcpyPI)).\n"
        "- **Ventilation Index** — shear × entropy deficit ÷ PI: how much dry, "
        "sheared air fights the storm (Tang & Emanuel 2012).\n"
        "- **vPI** — PI reduced by ventilation: the *achievable* intensity.\n"
        "- **GPIv** — genesis potential built from vPI and capped low-level "
        "vorticity (Chavas, Camargo & Tippett 2025, *J. Climate*).\n"
        "- Computed from ERA5 via [tcpyVPI](https://github.com/drchavas/tcpyVPI) "
        "— Chavas, Kruskie & Ocegueda Sanchez."
    )

C.render_footer(REPO_URL)
