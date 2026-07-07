"""ERA5 fetch helper for the TC-diagnostics app.

One function per THREDDS URL: open, subset server-side (band, stride,
levels, time), load, return plain numpy. Called SEQUENTIALLY on purpose:
the netCDF-C/HDF5 library is not thread-safe for concurrent OPeNDAP opens
(threads crash with DAP/HDF5 errors), and process pools re-import the
Streamlit app in spawn workers. Sequential is fast enough because the
per-variable level subsetting keeps each transfer small.
"""

from __future__ import annotations
import numpy as np


def fetch_field(args: tuple):
    """(name, url, time_spec, levels, lat_max, lat_min, stride) → dict.

    time_spec: ("index", i) for monthly files (decode-free positional pick)
               or ("nearest", iso_string) for hourly snapshots.
    levels:    None (surface), "all", or a (min, max) hPa range / list.
    """
    name, url, time_spec, levels, lat_max, lat_min, stride = args
    import xarray as xr

    ds = xr.open_dataset(url)

    var = None
    for k in (name, name.upper(), f"VAR_{name.upper()}"):
        if k in ds:
            var = ds[k]
            break
    if var is None:
        var = ds[list(ds.data_vars)[0]]

    kind, val = time_spec
    if kind == "index":
        var = var.isel(time=int(val))
    else:
        var = var.sel(time=val, method="nearest")

    if levels is not None and "level" in var.dims:
        if isinstance(levels, tuple):
            var = var.sel(level=slice(levels[0], levels[1]))
        elif isinstance(levels, list):
            var = var.sel(level=levels)

    var = var.sel(latitude=slice(lat_max, lat_min))
    if stride > 1:
        var = var.isel(latitude=slice(None, None, stride),
                       longitude=slice(None, None, stride))

    var = var.load()
    out = {
        "name": name,
        "values": var.values.astype(np.float32),
        "dims": var.dims,
        "latitude": var.latitude.values,
        "longitude": var.longitude.values,
    }
    if "level" in var.dims:
        out["level"] = var.level.values
    ds.close()
    return out
