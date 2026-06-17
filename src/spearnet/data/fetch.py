"""Fetch & store the global mine-segmentation dataset (Methodology v4, Section 5).

Dataset: SimonJasansky/mine-segmentation (Zenodo 10.5281/zenodo.14195737). The 35.6 MB
annotation GeoPackage carries, per mining site, the exact Sentinel-2 STAC id (`s2_tile_id`),
the verified mask polygons, the train/val/test `split`, and the metadata we train on —
`minetype1` (surface/placer/...) and `minetype2` (Artisanal vs Industrial, the headline
scale label).

This module:
  1. downloads the annotation GeoPackage from Zenodo (small, one-time);
  2. (metadata mode) prints split / minetype counts so you can see the ASM signal;
  3. subsets tiles by scale + a per-split cap;
  4. fetches 6-band S2 L2A (B,G,R,NIR,SWIR1,SWIR2) from Microsoft Planetary Computer
     (free, no credentials), rasterizes the mask polygons onto the tile grid, chips
     2048 -> 256, and stores uint16 GeoTIFF image+mask pairs;
  5. writes ``manifest.csv`` (img, mask, split, scale, minetype1, region, has_mining) that
     the training dataset consumes — this decouples loading from any filename scheme and
     carries the per-chip `scale` (for the artisanal/industrial task) and `region` (for the
     geographic leave-one-region-out protocol).

All heavy geospatial imports are lazy so the rest of the package imports without them.
Install on Colab:
    pip install pystac pystac-client planetary-computer stackstac rioxarray \
                geopandas rasterio shapely pyproj pyogrio tqdm
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import time
import urllib.request
import zipfile
from typing import Dict, List, Optional, Tuple

# Zenodo record for SimonJasansky/mine-segmentation. We resolve the actual file names via
# the Zenodo API at runtime (filenames change between versions), then download.
ZENODO_RECORD_ID = "14195737"
ZENODO_API = "https://zenodo.org/api/records/{rid}"

# S2 L2A asset keys on Planetary Computer for our 6 bands, in output order.
S2_ASSETS: List[str] = ["B02", "B03", "B04", "B08", "B11", "B12"]  # B,G,R,NIR,SWIR1,SWIR2

_TILE_LAYER_CANDIDATES = ["tiles"]
_MASK_LAYER_CANDIDATES = ["preferred_polygons", "maus_polygons", "tang_polygons"]
_SCALE_COL_CANDIDATES = ["minetype2", "scale", "minetype_2", "type2"]


# --------------------------------------------------------------------------------------
# Download annotations
# --------------------------------------------------------------------------------------
def download_annotations(dest_dir: str = "mine_data", record_id: str = ZENODO_RECORD_ID,
                         annot_url: Optional[str] = None) -> str:
    """Download the annotation GeoPackage (idempotent). Returns the .gpkg path.

    Resolves the real file name via the Zenodo API (robust to version changes). Pass an
    explicit ``annot_url`` to override (e.g. a mirror or a direct file link).
    """
    os.makedirs(dest_dir, exist_ok=True)

    # already downloaded?
    for f in os.listdir(dest_dir):
        if f.lower().endswith(".gpkg"):
            p = os.path.join(dest_dir, f)
            if os.path.getsize(p) > 1_000_000:
                print(f"[annot] already present: {p}")
                return p

    if annot_url:
        out = os.path.join(dest_dir, os.path.basename(annot_url.split("?")[0]) or "annotations.gpkg")
        print(f"Downloading annotations (manual URL) -> {out} ...")
        _download(annot_url, out)
        return _ensure_gpkg(out, dest_dir)

    files = _zenodo_files(record_id)
    if not files:
        raise RuntimeError(
            f"No files listed for Zenodo record {record_id}. Pass annot_url=... explicitly."
        )
    # Prefer a .gpkg; else a .zip that contains one.
    target = (_first(files, ".gpkg") or _first(files, ".zip")
              or _first(files, ".gpkg.zip") or files[0])
    key = target["key"]
    url = target["links"].get("self") or target["links"].get("download")
    out = os.path.join(dest_dir, key)
    print(f"Downloading annotations from Zenodo: {key} ({target.get('size','?')} bytes) ...")
    _download(url, out)
    return _ensure_gpkg(out, dest_dir)


def _zenodo_files(record_id: str) -> List[Dict]:
    url = ZENODO_API.format(rid=record_id)
    req = urllib.request.Request(url, headers={"User-Agent": "spearnet/0.1"})
    for i in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.load(r)
            return data.get("files", [])
        except Exception as e:  # noqa: BLE001
            print(f"  [zenodo-api] attempt {i+1} failed: {e}")
            time.sleep(2 ** i)
    return []


def _first(files: List[Dict], suffix: str) -> Optional[Dict]:
    for f in files:
        if f.get("key", "").lower().endswith(suffix):
            return f
    return None


def _ensure_gpkg(path: str, dest_dir: str) -> str:
    """If ``path`` is already a gpkg, return it; if it's a zip, extract and find the gpkg."""
    if path.lower().endswith(".gpkg"):
        return path
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            zf.extractall(dest_dir)
        for root, _, files in os.walk(dest_dir):
            for f in files:
                if f.lower().endswith(".gpkg"):
                    return os.path.join(root, f)
    raise FileNotFoundError(f"No .gpkg found in/under {path}")


def _download(url: str, path: str, retries: int = 4) -> None:
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "spearnet/0.1"})
            with urllib.request.urlopen(req, timeout=300) as r, open(path, "wb") as f:
                shutil.copyfileobj(r, f)
            return
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"  [download] attempt {i+1}/{retries} failed: {e}")
            time.sleep(2 ** i)
    raise last


# --------------------------------------------------------------------------------------
# Read / inspect the GeoPackage
# --------------------------------------------------------------------------------------
def _list_layers(gpkg: str) -> List[str]:
    try:
        import fiona

        return list(fiona.listlayers(gpkg))
    except Exception:
        from pyogrio import list_layers

        return [r[0] for r in list_layers(gpkg)]


def _read_layer(gpkg: str, layer: str):
    import geopandas as gpd

    return gpd.read_file(gpkg, layer=layer)


def _pick(cands: List[str], available: List[str]) -> Optional[str]:
    for c in cands:
        if c in available:
            return c
    return None


def load_tiles(gpkg: str):
    """Return (tiles_gdf, mask_gdf, scale_col, layers)."""
    layers = _list_layers(gpkg)
    tile_layer = _pick(_TILE_LAYER_CANDIDATES, layers) or layers[0]
    mask_layer = _pick(_MASK_LAYER_CANDIDATES, layers)
    tiles = _read_layer(gpkg, tile_layer)
    masks = _read_layer(gpkg, mask_layer) if mask_layer else None
    scale_col = _pick(_SCALE_COL_CANDIDATES, list(tiles.columns))
    return tiles, masks, scale_col, layers


def print_summary(gpkg: str) -> None:
    tiles, masks, scale_col, layers = load_tiles(gpkg)
    print(f"GeoPackage : {gpkg}")
    print(f"Layers     : {layers}")
    print(f"Tiles      : {len(tiles)} rows | columns: {list(tiles.columns)}")
    if "split" in tiles:
        print("\n--- split counts ---")
        print(tiles["split"].value_counts())
    if "minetype1" in tiles:
        print("\n--- minetype1 (mine type) ---")
        print(tiles["minetype1"].value_counts())
    if scale_col:
        print(f"\n--- {scale_col} (scale: artisanal vs industrial) ---")
        print(tiles[scale_col].value_counts())
        print(f"\nDetected scale column: {scale_col}")
    else:
        print("\n[warn] no scale column detected (minetype2). 3-class task unavailable.")


# --------------------------------------------------------------------------------------
# Geography helper (for leave-one-region-out)
# --------------------------------------------------------------------------------------
def continent_of(lon: float, lat: float) -> str:
    """Very coarse continent bucket from lon/lat (good enough for region hold-out)."""
    if lat < -60:
        return "Antarctica"
    if -170 <= lon < -30 and lat >= 7:
        return "NorthAmerica"
    if -90 <= lon < -30 and lat < 7:
        return "SouthAmerica"
    if -30 <= lon < 60 and lat >= 35:
        return "Europe"
    if -20 <= lon < 55 and lat < 35:
        return "Africa"
    if 25 <= lon < 60 and 12 <= lat < 45:
        return "MiddleEast"
    if 60 <= lon < 150 and lat >= 5:
        return "Asia"
    return "Oceania"


# --------------------------------------------------------------------------------------
# Fetch one tile's S2 window + rasterize its mask, then chip
# --------------------------------------------------------------------------------------
def _open_catalog():
    import planetary_computer
    import pystac_client

    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )


def _parse_s2_id(s2_id: str) -> Dict[str, str]:
    """Parse an ESA/PC S2 id like S2A_MSIL2A_20180928T022551_R046_T49MHU_20201009T054208."""
    parts = s2_id.split("_")
    out = {"platform": "", "sensing": "", "orbit": "", "tile": "", "prefix": s2_id}
    if len(parts) >= 5:
        out["platform"] = parts[0]
        out["sensing"] = parts[2]                      # 20180928T022551
        out["orbit"] = parts[3]                         # R046
        out["tile"] = parts[4].lstrip("T")             # 49MHU  (PC s2:mgrs_tile has no T)
        out["prefix"] = "_".join(parts[:5])            # acquisition prefix (drops proc-time)
    return out


def _find_item(catalog, s2_id: str, centroid_lonlat, retries: int = 4):
    """Resolve a Planetary Computer S2 L2A item robustly.

    PC re-processes ESA scenes, so the trailing processing timestamp in ``s2_tile_id``
    usually differs from PC's id -> exact-id search fails. We therefore (1) try the exact
    id, then (2) search by MGRS tile + sensing date and match the acquisition prefix,
    falling back to the least-cloudy scene over the tile centroid.
    """
    p = _parse_s2_id(s2_id)
    lon, lat = centroid_lonlat
    last_err = None
    point = {"type": "Point", "coordinates": [lon, lat]}

    def _window(days: int) -> str:
        import datetime as _dt
        d = _dt.datetime.strptime(p["sensing"][:8], "%Y%m%d")
        a = (d - _dt.timedelta(days=days)).strftime("%Y-%m-%d")
        b = (d + _dt.timedelta(days=days)).strftime("%Y-%m-%d")
        return f"{a}T00:00:00Z/{b}T23:59:59Z"

    for attempt in range(retries):
        try:
            # (1) fast path: exact id
            items = list(catalog.search(collections=["sentinel-2-l2a"], ids=[s2_id]).items())
            if items:
                return items[0]

            if p["sensing"]:
                # (2) same acquisition by MGRS tile + sensing-day window
                query = {"s2:mgrs_tile": {"eq": p["tile"]}} if p["tile"] else None
                cands = list(catalog.search(
                    collections=["sentinel-2-l2a"], datetime=_window(2),
                    query=query, intersects=point,
                ).items())
                exact = [c for c in cands if c.id.startswith(p["prefix"])]
                if exact:
                    return exact[0]
                if cands:
                    cands.sort(key=lambda c: c.properties.get("eo:cloud_cover", 100))
                    return cands[0]

                # (3) last resort: any low-cloud scene over the point near that date
                cands = list(catalog.search(
                    collections=["sentinel-2-l2a"], datetime=_window(15),
                    intersects=point, query={"eo:cloud_cover": {"lt": 40}},
                ).items())
                if cands:
                    cands.sort(key=lambda c: c.properties.get("eo:cloud_cover", 100))
                    return cands[0]
            return None
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2 ** attempt)
    if last_err is not None:
        print(f"  [search-error] {s2_id}: {type(last_err).__name__}: {last_err}")
    return None


def fetch_tile_array(catalog, s2_id: str, centroid_lonlat, window_px: int, res: int,
                     retries: int = 4):
    """Return (array[6,H,W] float, transform, epsg) for the S2 window, or None."""
    import numpy as np
    import stackstac
    from pyproj import Transformer

    item = _find_item(catalog, s2_id, centroid_lonlat, retries)
    if item is None:
        return None

    epsg = int(item.properties.get("proj:epsg", 0)) or _utm_epsg(*centroid_lonlat)
    tx = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    cx, cy = tx.transform(centroid_lonlat[0], centroid_lonlat[1])
    half = window_px * res / 2.0
    bounds = (cx - half, cy - half, cx + half, cy + half)

    for i in range(retries):
        try:
            stack = stackstac.stack(
                [item], assets=S2_ASSETS, epsg=epsg, resolution=res,
                bounds=bounds, fill_value=np.nan, rescale=False,
            )  # default float64 (accepts NaN fill); cast to float32 after compute
            arr = stack.squeeze("time").compute()  # (band, y, x)
            data = np.nan_to_num(arr.values).astype("float32")
            transform = arr.rio.transform()
            return data, transform, epsg
        except Exception as e:  # noqa: BLE001
            if i == retries - 1:
                print(f"  [stac-load-error] {s2_id}: {type(e).__name__}: {e}")
            time.sleep(2 ** i)
    return None


def _utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) / 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def rasterize_mask(mask_gdf, tile_id, s2_id, epsg, transform, shape) -> "np.ndarray":
    """Rasterize a tile's mask polygons onto the fetched grid (1 = mining)."""
    import numpy as np
    import rasterio.features

    geoms = _tile_geometries(mask_gdf, tile_id, s2_id, epsg)
    if not geoms:
        return np.zeros(shape, dtype="uint8")
    return rasterio.features.rasterize(
        [(g, 1) for g in geoms], out_shape=shape, transform=transform,
        fill=0, dtype="uint8",
    )


def _tile_geometries(mask_gdf, tile_id, s2_id, epsg) -> List:
    if mask_gdf is None:
        return []
    sub = mask_gdf
    for key, val in (("tile_id", tile_id), ("s2_tile_id", s2_id)):
        if key in mask_gdf.columns:
            m = mask_gdf[mask_gdf[key] == val]
            if len(m):
                sub = m
                break
    sub = sub.to_crs(epsg=epsg)
    out = []
    for g in sub.geometry:
        if g is not None and not g.is_empty:
            out.append(g)
    return out


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def fetch_dataset(
    out_dir: str = "chips",
    annot_dir: str = "mine_data",
    fetch_imagery: bool = False,
    scale_filter: Optional[str] = None,     # None | "artisanal" | "industrial"
    subset_per_split: Optional[int] = 60,   # cap tiles per split (None = all)
    chip_size: int = 256,
    window_px: int = 2048,
    res: int = 10,
    keep_empty_frac: float = 0.3,
    splits: Tuple[str, ...] = ("train", "val", "test"),
    seed: int = 42,
) -> Optional[str]:
    """Run the full pipeline. With ``fetch_imagery=False`` only prints metadata.

    Returns the manifest.csv path when imagery is fetched, else None.
    """
    import random
    import logging

    # Quiet benign GDAL/rasterio chatter during COG reads (CPLE_NotSupported SHARING/WARP_EXTRAS).
    logging.getLogger("rasterio._env").setLevel(logging.ERROR)

    gpkg = download_annotations(annot_dir)
    print_summary(gpkg)
    if not fetch_imagery:
        print("\nFETCH_IMAGERY=False -> metadata only. Set scale_filter/subset, then "
              "re-run with fetch_imagery=True.")
        return None

    tiles, masks, scale_col, _ = load_tiles(gpkg)
    tiles = tiles.to_crs(epsg=4326)
    if scale_filter and scale_col:
        tiles = tiles[tiles[scale_col].str.lower() == scale_filter.lower()]
        print(f"[subset] scale_filter={scale_filter} -> {len(tiles)} tiles")

    rng = random.Random(seed)
    catalog = _open_catalog()
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest.csv")
    write_header = not os.path.exists(manifest_path)
    fout = open(manifest_path, "a", newline="")
    writer = csv.writer(fout)
    if write_header:
        writer.writerow(["img", "mask", "split", "scale", "minetype1", "region",
                         "has_mining", "tile_id", "s2_tile_id"])

    counts = {s: 0 for s in splits}
    for split in splits:
        sub = tiles[tiles["split"] == split] if "split" in tiles else tiles
        rows = list(sub.itertuples())
        rng.shuffle(rows)
        if subset_per_split is not None:
            rows = rows[:subset_per_split]
        print(f"\n[{split}] fetching {len(rows)} tiles ...")
        for ti, r in enumerate(rows):
            n = _process_tile(r, masks, scale_col, catalog, out_dir, split, writer,
                              chip_size, window_px, res, keep_empty_frac, rng)
            counts[split] += n
            fout.flush()
            if n:
                print(f"  [{split} {ti+1}/{len(rows)}] +{n} chips "
                      f"(total {counts[split]})", flush=True)

    fout.close()
    print(f"\nDone. Chips per split: {counts}")
    print(f"Manifest -> {manifest_path}")
    return manifest_path


def _process_tile(r, masks, scale_col, catalog, out_dir, split, writer,
                  chip_size, window_px, res, keep_empty_frac, rng) -> int:
    import numpy as np
    import rasterio
    from rasterio.transform import Affine

    geom = r.geometry
    if geom is None:
        return 0
    centroid = (geom.centroid.x, geom.centroid.y)
    region = continent_of(*centroid)
    scale = (getattr(r, scale_col, "") or "").lower() if scale_col else ""
    minetype1 = getattr(r, "minetype1", "") or ""
    tile_id = getattr(r, "tile_id", getattr(r, "Index", "t"))
    s2_id = getattr(r, "s2_tile_id", "")

    fetched = fetch_tile_array(catalog, s2_id, centroid, window_px, res)
    if fetched is None:
        print(f"  [skip] {tile_id}: no imagery for {s2_id}")
        return 0
    img, transform, epsg = fetched
    H, W = img.shape[-2:]
    mask = rasterize_mask(masks, tile_id, s2_id, epsg, transform, (H, W))

    split_dir = os.path.join(out_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    img_u16 = np.clip(img, 0, 65535).astype("uint16")
    n_written = 0
    for yi in range(0, H - chip_size + 1, chip_size):
        for xi in range(0, W - chip_size + 1, chip_size):
            cmask = mask[yi:yi + chip_size, xi:xi + chip_size]
            has = bool(cmask.any())
            if not has and rng.random() > keep_empty_frac:
                continue
            cimg = img_u16[:, yi:yi + chip_size, xi:xi + chip_size]
            base = f"{tile_id}_{xi}_{yi}"
            ip = os.path.join(split_dir, base + "_img.tif")
            mp = os.path.join(split_dir, base + "_mask.tif")
            ctransform = transform * Affine.translation(xi, yi)
            _write_tif(ip, cimg, epsg, ctransform)
            _write_tif(mp, cmask[None], epsg, ctransform)
            writer.writerow([ip, mp, split, scale, minetype1, region,
                             int(has), tile_id, s2_id])
            n_written += 1
    return n_written


def _write_tif(path: str, arr, epsg: int, transform) -> None:
    import rasterio

    count, h, w = arr.shape
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=count,
        dtype=arr.dtype, crs=f"EPSG:{epsg}", transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(arr)
