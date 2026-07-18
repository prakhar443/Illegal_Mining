#!/usr/bin/env python
"""Phase-A dataset diagnostics for the SPEAR-Net manuscript.

Runs the counts the paper's Table 1 / Section 4.4 must be rebuilt from, and reconciles
the 1210 / 1514 / 1207 site-vs-tile discrepancy. Reads only the annotation GeoPackage and
(if present) the chips manifest -- no model, no GPU.

Usage (Colab, after cell 5 has downloaded the GeoPackage):
    python scripts/diagnostics.py \
        --gpkg mine_data/mining_area_data.gpkg \
        --manifest /content/drive/MyDrive/spearnet_chips/manifest.csv \
        --out diagnostics.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _ct(df, a, b):
    import pandas as pd
    return pd.crosstab(df[a], df[b], margins=True) if a in df and b in df else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpkg", required=True, help="annotation GeoPackage")
    ap.add_argument("--manifest", default=None, help="chips/manifest.csv (optional)")
    ap.add_argument("--out", default="diagnostics.json")
    args = ap.parse_args()

    import geopandas as gpd
    import pandas as pd
    try:
        import fiona
        layers = list(fiona.listlayers(args.gpkg))
    except Exception:
        from pyogrio import list_layers
        layers = [r[0] for r in list_layers(args.gpkg)]

    print("=" * 70)
    print("A1  GeoPackage layers:", layers)

    # pick the tiles layer
    tlayer = "tiles" if "tiles" in layers else layers[0]
    gdf = gpd.read_file(args.gpkg, layer=tlayer)
    n_rows = len(gdf)
    print(f"A1  tiles layer '{tlayer}': {n_rows} rows | columns: {list(gdf.columns)}")

    out = {"layers": layers, "tiles_layer": tlayer, "n_rows": int(n_rows), "tables": {}}

    # A2: split x scale
    print("\nA2  split x minetype2 (scale):")
    t = _ct(gdf, "split", "minetype2"); print(t)
    if t is not None: out["tables"]["split_x_scale"] = t.to_dict()

    # A3: split x mine type
    print("\nA3  split x minetype1 (mine type):")
    t = _ct(gdf, "split", "minetype1"); print(t)
    if t is not None: out["tables"]["split_x_minetype"] = t.to_dict()

    # A4: reconcile counts
    print("\nA4  reconciliation:")
    n_split = int(gdf["split"].notna().sum()) if "split" in gdf else 0
    uniq_s2 = int(gdf["s2_tile_id"].nunique()) if "s2_tile_id" in gdf else None
    print(f"    total annotation rows              = {n_rows}")
    print(f"    rows with a split assigned         = {n_split}")
    print(f"    unique s2_tile_id (distinct scenes)= {uniq_s2}")
    if "minetype2" in gdf:
        print(f"    scale value counts                 = {gdf['minetype2'].value_counts().to_dict()}")
    print("    -> Use 'annotation tiles' for the row count; report unique scenes as 'sites'.")
    out["n_rows"] = int(n_rows); out["n_with_split"] = n_split; out["n_unique_scenes"] = uniq_s2

    # A5/A6: chips manifest
    if args.manifest and os.path.exists(args.manifest):
        m = pd.read_csv(args.manifest)
        print(f"\nA5  chips manifest: {len(m)} rows | columns {list(m.columns)}")
        t = _ct(m, "split", "scale"); print("\nA5  split x scale (chips):\n", t)
        if t is not None: out["tables"]["chips_split_x_scale"] = t.to_dict()
        t = _ct(m, "region", "scale"); print("\nA6  region x scale (chips):\n", t)
        if t is not None: out["tables"]["chips_region_x_scale"] = t.to_dict()
        if "has_mining" in m:
            print("\nA5  mining vs empty chips:\n", m["has_mining"].value_counts().to_dict())
            out["chips_has_mining"] = m["has_mining"].value_counts().to_dict()
        out["n_chips"] = int(len(m))
    else:
        print("\nA5/A6  manifest not found -> run the chip-fetch first, then re-run with --manifest")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved -> {args.out}")
    print("Use A2 for Table 1 (scale), A3 for the mine-type block, A5 for chips-per-split,")
    print("A6 to rewrite Section 4.4, and A4 to fix every count in paper/README.")


if __name__ == "__main__":
    main()
