"""Build the TC-diagnostics archive (GPIv, vPI, PI, VI, VWS, Chi, eta_c) from ERA5.

Compute pipeline for the TCdiag_streamlit apps. Designed for Purdue's Negishi
cluster (see tools/run_archive.slurm) but runs anywhere with a fast pipe to
NCAR RDA. Requires: tcpyVPI, tcpyPI, xarray, netCDF4, numpy, pandas.

Data flow
---------
monthly      ERA5 monthly means (RDA d633001, OPeNDAP) -> lazy subset to the
             tropics band (default 40S-40N at 0.5 deg) -> download -> compute
             GPIv + components with tcpyVPI -> one small per-month file under
             {out}/monthly/, folded into ONE netCDF per variable per decade
             (tcdiag__vPI__1980s.nc, ...) once every month of the decade is on
             disk. float32 + zlib. time = month starts.
climatology  FROM the completed archive (no re-download): per-calendar-month
             mean and std (ddof=1) across years -> tcdiag_clim__{var}.nc with
             dims (month: 12, latitude, longitude); plus seasonal JJA / SON /
             JJASON means and stds -> tcdiag_clim_seasonal__{var}.nc with dims
             (season: 3, latitude, longitude).
verify       sanity checks: month coverage, all-NaN fields, plausible value
             ranges. Prints a table; exits non-zero on problems.

Usage (full runbook in tools/README.md):
    python tools/build_archive.py monthly     --years 1980-2024 --out archive/
    python tools/build_archive.py climatology --years 1980-2024 --archive archive/ --out clim/
    python tools/build_archive.py verify      --archive archive/ --clim clim/

Resumability / crash safety
---------------------------
* A (year, month) is SKIPPED if its per-month file exists OR its timestamp is
  already in all seven per-decade files -- safe to resubmit after a crash.
* Every netCDF is written to a ".tmp" file then atomically renamed, so a
  killed job never leaves a truncated file behind.
* Months that fail (THREDDS hiccup, etc.) are appended to
  {out}/failed_months.txt and the run continues.
* Decade consolidation is guarded by a lock file so concurrent SLURM array
  tasks cannot clobber each other. If a job dies mid-consolidation, delete the
  stale {out}/.consolidate_*.lock and rerun with --consolidate-only.

NOTE: keep --domain and --stride identical across all runs feeding one
archive directory -- grids must match for the decade merge to concat.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import xarray as xr


# ─────────── constants ───────────────────────────────────────────────────────

# Diagnostics produced by tcpyVPI.compute_gpiv_from_dataset (archive order).
VARIABLES = ["GPIv", "vPI", "PI", "ventilation_index", "VWS", "Chi", "eta_c"]

# ERA5 inputs (matches tcpyVPI.era5_loader.load_era5_monthly).
SURFACE_VARS = ["SSTK", "SP"]           # sea-surface temperature, surface pressure
PRESSURE_VARS = ["T", "Q", "U", "V", "VO"]

LAT, LON = "latitude", "longitude"      # ERA5 dim names (latitude DESCENDING 90 -> -90)

MONTHLY_RE = re.compile(r"^tcdiag__(\d{4})-(\d{2})\.nc$")
DECADE_RE = re.compile(r"^tcdiag__.+__(\d{4})s\.nc$")

TIME_ENC = {"units": "days since 1900-01-01", "dtype": "int32", "calendar": "standard"}
VAR_ENC = {"zlib": True, "complevel": 4, "dtype": "float32"}

# Plausible value ranges for `verify` ((None, None) = report only, no pass/fail).
# vPI/PI in m/s over ocean; VWS in m/s; eta_c is capped at +/-3.7e-5 s^-1 by tcpyVPI.
PLAUSIBLE: Dict[str, Tuple[Optional[float], Optional[float]]] = {
    "vPI": (0.0, 150.0),
    "PI": (0.0, 150.0),
    "VWS": (0.0, 80.0),
    "GPIv": (0.0, None),
    "ventilation_index": (0.0, None),
    "eta_c": (-4.0e-5, 4.0e-5),
    "Chi": (None, None),
}

# Seasonal climatology definitions (order = season coordinate order).
SEASONS: List[Tuple[str, List[int]]] = [
    ("JJA", [6, 7, 8]),
    ("SON", [9, 10, 11]),
    ("JJASON", [6, 7, 8, 9, 10, 11]),
]


# ─────────── small helpers ────────────────────────────────────────────────────

def parse_years(spec: str) -> Tuple[int, int]:
    """Parse '1980-2024' (or a single '1980') into an inclusive (y0, y1)."""
    parts = spec.split("-")
    if len(parts) == 1:
        y = int(parts[0])
        return y, y
    y0, y1 = int(parts[0]), int(parts[1])
    if y1 < y0:
        raise ValueError(f"bad year range: {spec}")
    return y0, y1


def parse_domain(spec: str) -> Tuple[float, float]:
    """Parse '40,-40' into (north, south) latitude edges (degrees)."""
    a, b = (float(x) for x in spec.split(","))
    return max(a, b), min(a, b)


def _decade(year: int) -> int:
    return (year // 10) * 10


def _ym(t) -> Tuple[int, int]:
    """(year, month) key from any time value xarray hands back."""
    ts = pd.Timestamp(t)
    return ts.year, ts.month


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _global_attrs(**extra) -> dict:
    attrs = {
        "source": "ERA5 monthly means (NCAR RDA d633001, OPeNDAP) via tcpyVPI",
        "reference": "Chavas, Camargo & Tippett (2025, J. Clim.)",
        "units_convention": "tcpyVPI native",
        "created_by": "TCdiag_streamlit/tools/build_archive.py (Jose Ocegueda Sanchez)",
        "created": _now_iso(),
    }
    attrs.update(extra)
    return attrs


def _atomic_write(ds: xr.Dataset, path: Path, encoding: dict) -> None:
    """Write to .tmp then rename, so a crash never leaves a truncated file."""
    tmp = path.parent / (path.name + ".tmp")
    ds.to_netcdf(tmp, encoding=encoding)
    os.replace(tmp, path)


# ─────────── ERA5 loading (parallel THREDDS opens) ────────────────────────────

def open_month_lazy(year: int, month: int, verbose: bool = True) -> xr.Dataset:
    """Open the 7 ERA5 monthly-mean THREDDS datasets and merge one month, LAZILY.

    Re-implements the loop of tcpyVPI.era5_loader.load_era5_monthly but opens
    the seven remote datasets in parallel threads: sequential opens take ~30 s
    while parallel opens take ~5-8 s (the opens are metadata round-trips, so
    they thread well). Falls back to load_era5_monthly if the private URL
    builder is not importable from the installed tcpyVPI.

    Returns a LAZY dataset -- no field data has been transferred yet. Subset
    (see subset_domain) BEFORE calling .load() so only the tropics band at the
    target stride crosses the wire.
    """
    try:
        from tcpyVPI.era5_loader import _get_monthly_mean_url
    except ImportError:
        from tcpyVPI.era5_loader import load_era5_monthly
        if verbose:
            print("  (tcpyVPI URL builder not importable -> sequential load_era5_monthly)")
        return load_era5_monthly(year, month, verbose=verbose)

    def _open(var: str, is_surface: bool):
        url = _get_monthly_mean_url(var, year, is_surface=is_surface)
        return var, xr.open_dataset(url)  # lazy: reads metadata only

    jobs = [(v, True) for v in SURFACE_VARS] + [(v, False) for v in PRESSURE_VARS]
    opened = {}
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        for fut in [pool.submit(_open, v, s) for v, s in jobs]:
            var, ds = fut.result()
            opened[var] = ds

    idx = month - 1  # yearly files hold 12 months
    return xr.merge([opened[v][v].isel(time=idx) for v in SURFACE_VARS + PRESSURE_VARS])


def subset_domain(ds: xr.Dataset, north: float, south: float, stride: int) -> xr.Dataset:
    """Lazily subset to the latitude band and coarsen by `stride`.

    ERA5 latitude runs DESCENDING (90 -> -90), so the band selection is
    slice(north, south), e.g. slice(40, -40). stride=2 keeps every 2nd point
    of the 0.25 deg grid -> 0.5 deg. Applied before .load(), this cuts the
    transfer by ~18x versus pulling the full global grid.
    """
    ds = ds.sel({LAT: slice(north, south)})
    if stride > 1:
        ds = ds.isel({LAT: slice(None, None, stride), LON: slice(None, None, stride)})
    return ds


def process_month(year: int, month: int, north: float, south: float,
                  stride: int, verbose: bool = True) -> xr.Dataset:
    """Load one (year, month), compute all diagnostics, return a (time=1) Dataset.

    Cost profile (0.5 deg, 40S-40N): transfer dominates at ~90 s per month
    from RDA; the pointwise PI/GPIv compute is a few seconds.
    """
    from tcpyVPI.vpigpiv_module import compute_gpiv_from_dataset

    t0 = time.perf_counter()
    ds = subset_domain(open_month_lazy(year, month, verbose=verbose), north, south, stride)
    t1 = time.perf_counter()
    ds = ds.load()                       # <-- the actual data transfer
    t2 = time.perf_counter()

    res = compute_gpiv_from_dataset(ds, verbose=False)[VARIABLES]

    # Scrub scalar leftovers (input time stamp, level=850 from eta_c, ...) and
    # index by the month start so decade files concat cleanly along time.
    drop = [c for c in ("time", "level", "utc_date") if c in res.coords]
    if drop:
        res = res.drop_vars(drop)
    stamp = np.datetime64(f"{year}-{month:02d}-01")
    res = res.expand_dims(time=[stamp]).astype("float32")

    res.attrs.update(_global_attrs(
        domain=f"latitude {north} to {south}, full longitude",
        stride=stride,
        grid_deg=0.25 * stride,
    ))
    t3 = time.perf_counter()
    if verbose:
        print(f"  {year}-{month:02d}: open {t1 - t0:.0f} s | transfer {t2 - t1:.0f} s "
              f"| compute {t3 - t2:.0f} s "
              f"({res.sizes[LAT]}x{res.sizes[LON]} grid)")
    return res


# ─────────── archive bookkeeping ──────────────────────────────────────────────

def monthly_dir(out: Path) -> Path:
    return Path(out) / "monthly"


def scan_monthly_files(out: Path) -> Dict[Tuple[int, int], Path]:
    """Map (year, month) -> per-month file path for everything under {out}/monthly."""
    found = {}
    mdir = monthly_dir(out)
    if mdir.is_dir():
        for p in mdir.iterdir():
            m = MONTHLY_RE.match(p.name)
            if m:
                found[(int(m.group(1)), int(m.group(2)))] = p
    return found


def decades_on_disk(out: Path) -> Set[int]:
    """Decades for which at least one per-decade file exists."""
    decades = set()
    for p in Path(out).glob("tcdiag__*__*s.nc"):
        m = DECADE_RE.match(p.name)
        if m:
            decades.add(int(m.group(1)))
    return decades


def decade_month_index(out: Path, decade: int) -> Set[Tuple[int, int]]:
    """(year, month) keys present in ALL SEVEN per-decade files for `decade`.

    Intersecting across variables means a month interrupted mid-consolidation
    (some variables written, some not) is treated as absent and gets redone.
    """
    times: Optional[Set[Tuple[int, int]]] = None
    for var in VARIABLES:
        f = Path(out) / f"tcdiag__{var}__{decade}s.nc"
        if not f.exists():
            return set()
        with xr.open_dataset(f) as ds:
            t = {_ym(x) for x in ds["time"].values}
        times = t if times is None else (times & t)
    return times or set()


def archive_month_index(out: Path) -> Set[Tuple[int, int]]:
    """Every (year, month) already done: per-month files + per-decade files."""
    present = set(scan_monthly_files(out))
    for decade in decades_on_disk(out):
        present |= decade_month_index(out, decade)
    return present


def write_month(res: xr.Dataset, out: Path, year: int, month: int) -> Path:
    """Atomically write one per-month file with all 7 variables."""
    mdir = monthly_dir(out)
    mdir.mkdir(parents=True, exist_ok=True)
    path = mdir / f"tcdiag__{year}-{month:02d}.nc"
    enc = {v: dict(VAR_ENC) for v in VARIABLES}
    enc["time"] = dict(TIME_ENC)
    _atomic_write(res, path, enc)
    return path


def log_failure(out: Path, year: int, month: int, err: Exception) -> None:
    """Append a failed (year, month) to {out}/failed_months.txt and keep going."""
    line = f"{year}-{month:02d}\t{_now_iso()}\t{type(err).__name__}: {err}\n"
    with open(Path(out) / "failed_months.txt", "a") as fh:
        fh.write(line)


def consolidate_decade(out: Path, decade: int, years_needed: List[int],
                       verbose: bool = True) -> None:
    """Fold per-month files into the seven per-decade files, if complete.

    Runs only when EVERY month of `years_needed` exists (per-month file or
    already inside the decade files); otherwise it reports what is missing and
    returns -- the last SLURM array task to finish a decade does the merge.
    A lock file serialises concurrent attempts. Per-month files are deleted
    only after all seven variables are safely rewritten (atomic renames), so
    a crash at any point is recoverable by rerunning.
    """
    out = Path(out)
    needed = {(y, m) for y in years_needed for m in range(1, 13)}
    if not needed:
        return
    mfiles = {k: p for k, p in scan_monthly_files(out).items() if _decade(k[0]) == decade}
    dec_present = decade_month_index(out, decade)

    missing = sorted(needed - (set(mfiles) | dec_present))
    if missing:
        if verbose:
            head = ", ".join(f"{y}-{m:02d}" for y, m in missing[:6])
            more = f" (+{len(missing) - 6} more)" if len(missing) > 6 else ""
            print(f"  {decade}s: not complete yet, missing {len(missing)} month(s): {head}{more}")
        return

    lock = out / f".consolidate_{decade}s.lock"
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        if verbose:
            print(f"  {decade}s: {lock.name} exists (another task merging; "
                  f"delete it if stale) -- skipping")
        return

    try:
        to_add = sorted(k for k in (needed & set(mfiles)) if k not in dec_present)
        if to_add:
            if verbose:
                print(f"  {decade}s: folding {len(to_add)} month(s) into per-decade files...")
            new_data = {}
            for k in to_add:
                with xr.open_dataset(mfiles[k]) as dsm:
                    new_data[k] = dsm[VARIABLES].load()

            for var in VARIABLES:
                f = out / f"tcdiag__{var}__{decade}s.nc"
                old = None
                if f.exists():
                    with xr.open_dataset(f) as ds_old:
                        old = ds_old[var].load()
                old_keys = {_ym(t) for t in old["time"].values} if old is not None else set()
                pieces = ([] if old is None else [old])
                pieces += [new_data[k][var] for k in to_add if k not in old_keys]
                da = pieces[0] if len(pieces) == 1 else xr.concat(pieces, dim="time")
                da = da.sortby("time")
                ds_out = da.to_dataset(name=var)
                ds_out.attrs.update(_global_attrs())
                _atomic_write(ds_out, f, {var: dict(VAR_ENC), "time": dict(TIME_ENC)})
                if verbose:
                    print(f"    wrote {f.name}  ({f.stat().st_size / 1e6:.1f} MB, "
                          f"{da.sizes['time']} months)")

        # Delete per-month files now redundantly stored in all 7 decade files.
        dec_present = decade_month_index(out, decade)
        removed = 0
        for k in sorted(needed & set(mfiles)):
            if k in dec_present:
                mfiles[k].unlink()
                removed += 1
        if verbose and removed:
            print(f"  {decade}s: consolidated; removed {removed} per-month file(s)")
    finally:
        os.close(fd)
        try:
            lock.unlink()
        except OSError:
            pass


# ─────────── subcommand: monthly ──────────────────────────────────────────────

def cmd_monthly(args: argparse.Namespace) -> int:
    y0, y1 = parse_years(args.years)
    fp0, fp1 = parse_years(args.full_period)
    north, south = parse_domain(args.domain)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Archive dir : {out.resolve()}")
    print(f"Years       : {y0}-{y1} (full archive period {fp0}-{fp1})")
    print(f"Domain      : latitude {north} to {south}, stride {args.stride} "
          f"({0.25 * args.stride} deg)")

    n_ok = n_skip = n_fail = 0
    if not args.consolidate_only:
        present = archive_month_index(out)
        for year in range(y0, y1 + 1):
            for month in range(1, 13):
                if (year, month) in present:
                    n_skip += 1
                    continue
                try:
                    res = process_month(year, month, north, south, args.stride)
                    write_month(res, out, year, month)
                    n_ok += 1
                except Exception as e:                       # noqa: BLE001 -- keep going
                    n_fail += 1
                    print(f"  ! {year}-{month:02d} FAILED: {e}")
                    log_failure(out, year, month, e)
        print(f"\nMonths: {n_ok} computed, {n_skip} skipped (already done), {n_fail} failed"
              + (f" -> see {out / 'failed_months.txt'}" if n_fail else ""))

    # Consolidate any decade that is now complete (bounded by the full period).
    print("\nConsolidation check:")
    if args.consolidate_only:
        decades = sorted({_decade(y) for y in range(fp0, fp1 + 1)})
    else:
        decades = sorted({_decade(y) for y in range(y0, y1 + 1)})
    for d in decades:
        years_in_decade = [y for y in range(d, d + 10) if fp0 <= y <= fp1]
        consolidate_decade(out, d, years_in_decade)

    return 1 if n_fail else 0


# ─────────── subcommand: climatology ──────────────────────────────────────────

def load_variable_series(archive: Path, var: str, y0: int, y1: int) -> xr.DataArray:
    """Assemble one (time, latitude, longitude) series from the archive on disk.

    Reads the per-decade files plus any not-yet-consolidated per-month files,
    dedupes on time, and clips to the base period. Local reads only.
    """
    pieces = []
    for f in sorted(Path(archive).glob(f"tcdiag__{var}__*s.nc")):
        with xr.open_dataset(f) as ds:
            pieces.append(ds[var].load())
    for _, p in sorted(scan_monthly_files(Path(archive)).items()):
        with xr.open_dataset(p) as ds:
            if var in ds:
                pieces.append(ds[var].load())
    if not pieces:
        raise FileNotFoundError(f"no archive files for '{var}' under {archive}")

    da = pieces[0] if len(pieces) == 1 else xr.concat(pieces, dim="time")
    da = da.sortby("time")
    _, first = np.unique(da["time"].values, return_index=True)
    if first.size != da.sizes["time"]:
        da = da.isel(time=np.sort(first))
    return da.sel(time=slice(f"{y0}-01-01", f"{y1}-12-31"))


def cmd_climatology(args: argparse.Namespace) -> int:
    y0, y1 = parse_years(args.years)
    archive, out = Path(args.archive), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_expected = 12 * (y1 - y0 + 1)
    incomplete = False

    print(f"Climatology base period: {y0}-{y1}  (archive: {archive.resolve()})")
    print(f"Output: {out.resolve()}\n")

    for var in VARIABLES:
        da = load_variable_series(archive, var, y0, y1)
        got = {_ym(t) for t in da["time"].values}
        if len(got) < n_expected:
            missing = sorted({(y, m) for y in range(y0, y1 + 1)
                              for m in range(1, 13)} - got)
            head = ", ".join(f"{y}-{m:02d}" for y, m in missing[:6])
            print(f"  ! {var}: only {len(got)}/{n_expected} months on disk "
                  f"(missing {head}{' ...' if len(missing) > 6 else ''}) -- "
                  f"stats use available months")
            incomplete = True

        # -- per-calendar-month mean and interannual std (sample std, ddof=1) --
        grouped = da.groupby("time.month")
        mean = grouped.mean("time")
        std = grouped.std("time", ddof=1)
        std.attrs = dict(da.attrs,
                         long_name=f"{da.attrs.get('long_name', var)} "
                                   f"(interannual std, ddof=1)")

        attrs = _global_attrs(base_period=f"{y0}-{y1}",
                              n_years=y1 - y0 + 1,
                              std_definition="year-to-year sample std (ddof=1) "
                                             "per calendar month")
        ds_out = xr.Dataset({var: mean.astype("float32"),
                             f"{var}_STD": std.astype("float32")},
                            attrs=attrs)
        f_month = out / f"tcdiag_clim__{var}.nc"
        _atomic_write(ds_out, f_month,
                      {var: dict(VAR_ENC), f"{var}_STD": dict(VAR_ENC)})
        print(f"  wrote {f_month.name}  ({f_month.stat().st_size / 1e6:.1f} MB)")

        # -- seasonal: per-year season mean first, then mean/std across years --
        means, stds = [], []
        for _, months in SEASONS:
            sel = da.sel(time=da["time"].dt.month.isin(months))
            yearly = sel.groupby("time.year").mean("time")
            means.append(yearly.mean("year"))
            stds.append(yearly.std("year", ddof=1))
        season_ix = pd.Index([name for name, _ in SEASONS], name="season")
        smean = xr.concat(means, dim=season_ix)
        sstd = xr.concat(stds, dim=season_ix)
        smean.attrs, sstd.attrs = dict(mean.attrs), dict(std.attrs)

        attrs_s = _global_attrs(base_period=f"{y0}-{y1}",
                                n_years=y1 - y0 + 1,
                                season_definition="; ".join(
                                    f"{n}=months {m}" for n, m in SEASONS),
                                std_definition="std (ddof=1) across per-year "
                                               "seasonal means")
        ds_seas = xr.Dataset({var: smean.astype("float32"),
                              f"{var}_STD": sstd.astype("float32")},
                             attrs=attrs_s)
        f_seas = out / f"tcdiag_clim_seasonal__{var}.nc"
        _atomic_write(ds_seas, f_seas,
                      {var: dict(VAR_ENC), f"{var}_STD": dict(VAR_ENC)})
        print(f"  wrote {f_seas.name}  ({f_seas.stat().st_size / 1e6:.1f} MB)")

    print("\nDone." + (" (WARNING: some months were missing -- see above)"
                       if incomplete else ""))
    return 1 if incomplete else 0


# ─────────── subcommand: verify ───────────────────────────────────────────────

def cmd_verify(args: argparse.Namespace) -> int:
    y0, y1 = parse_years(args.years)
    archive = Path(args.archive)
    expected = {(y, m) for y in range(y0, y1 + 1) for m in range(1, 13)}
    problems: List[str] = []

    print(f"Verifying archive {archive.resolve()}  (expected period {y0}-{y1}, "
          f"{len(expected)} months per variable)\n")
    hdr = (f"{'variable':<20}{'months':>10}{'allNaN':>8}{'min':>13}{'max':>13}"
           f"{'range':>10}")
    print(hdr)
    print("-" * len(hdr))

    for var in VARIABLES:
        files = sorted(archive.glob(f"tcdiag__{var}__*s.nc"))
        files += [p for _, p in sorted(scan_monthly_files(archive).items())]
        got: Set[Tuple[int, int]] = set()
        nan_maps = 0
        vmin, vmax = np.inf, -np.inf
        for f in files:
            with xr.open_dataset(f) as ds:
                if var not in ds:
                    continue
                keep = [i for i, t in enumerate(ds["time"].values)
                        if y0 <= pd.Timestamp(t).year <= y1]
                if not keep:
                    continue
                da = ds[var].isel(time=keep).load()
            got |= {_ym(t) for t in da["time"].values}
            nan_maps += int(da.isnull().all(dim=(LAT, LON)).sum())
            mn, mx = float(da.min()), float(da.max())
            if np.isfinite(mn):
                vmin = min(vmin, mn)
            if np.isfinite(mx):
                vmax = max(vmax, mx)

        n_missing = len(expected - got)
        lo, hi = PLAUSIBLE.get(var, (None, None))
        if lo is None and hi is None:
            range_txt = "--"
        elif not np.isfinite(vmin):
            range_txt = "NO DATA"
        elif (lo is not None and vmin < lo - 1e-9) or (hi is not None and vmax > hi + 1e-9):
            range_txt = "FAIL"
        else:
            range_txt = "ok"

        months_txt = f"{len(got)}/{len(expected)}"
        print(f"{var:<20}{months_txt:>10}{nan_maps:>8}{vmin:>13.4g}{vmax:>13.4g}"
              f"{range_txt:>10}")

        if n_missing:
            problems.append(f"{var}: {n_missing} month(s) missing")
        if nan_maps:
            problems.append(f"{var}: {nan_maps} all-NaN field(s)")
        if range_txt == "FAIL":
            problems.append(f"{var}: values outside plausible range "
                            f"[{lo}, {hi}]: min={vmin:.4g}, max={vmax:.4g}")
        if range_txt == "NO DATA":
            problems.append(f"{var}: no data found")

    if args.clim:
        clim = Path(args.clim)
        print(f"\nVerifying climatology {clim.resolve()}")
        for var in VARIABLES:
            for fname, dim, size in ((f"tcdiag_clim__{var}.nc", "month", 12),
                                     (f"tcdiag_clim_seasonal__{var}.nc", "season", 3)):
                f = clim / fname
                if not f.exists():
                    problems.append(f"clim: {fname} missing")
                    print(f"  {fname:<38} MISSING")
                    continue
                with xr.open_dataset(f) as ds:
                    issues = []
                    if ds.sizes.get(dim) != size:
                        issues.append(f"{dim}={ds.sizes.get(dim)} (want {size})")
                    for v in (var, f"{var}_STD"):
                        if v not in ds:
                            issues.append(f"no {v}")
                        elif bool(ds[v].isnull().all()):
                            issues.append(f"{v} all-NaN")
                    if "base_period" not in ds.attrs:
                        issues.append("no base_period attr")
                if issues:
                    problems.append(f"clim: {fname}: {'; '.join(issues)}")
                    print(f"  {fname:<38} FAIL: {'; '.join(issues)}")
                else:
                    print(f"  {fname:<38} ok")

    print()
    if problems:
        print(f"VERIFY: {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("VERIFY: all checks passed.")
    return 0


# ─────────── CLI ──────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("monthly", help="download+compute monthly diagnostics")
    p.add_argument("--years", required=True,
                   help="inclusive year range to process, e.g. 1980-2024 or 1997")
    p.add_argument("--out", required=True, help="archive output directory")
    p.add_argument("--domain", default="40,-40",
                   help="latitude band 'north,south' (default 40,-40; full longitude)")
    p.add_argument("--stride", type=int, default=2,
                   help="grid stride on the 0.25 deg grid (default 2 = 0.5 deg)")
    p.add_argument("--full-period", default="1980-2024",
                   help="full archive period; decades are consolidated only once "
                        "every month of the decade inside this period is done")
    p.add_argument("--consolidate-only", action="store_true",
                   help="skip downloads; just fold per-month files into decade files")
    p.set_defaults(func=cmd_monthly)

    p = sub.add_parser("climatology", help="monthly+seasonal mean/std from the archive")
    p.add_argument("--years", required=True, help="base period, e.g. 1980-2024")
    p.add_argument("--archive", required=True, help="completed archive directory")
    p.add_argument("--out", required=True, help="climatology output directory")
    p.set_defaults(func=cmd_climatology)

    p = sub.add_parser("verify", help="sanity-check the archive (and climatology)")
    p.add_argument("--archive", required=True, help="archive directory")
    p.add_argument("--clim", default=None, help="climatology directory (optional)")
    p.add_argument("--years", default="1980-2024",
                   help="expected coverage period (default 1980-2024)")
    p.set_defaults(func=cmd_verify)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
