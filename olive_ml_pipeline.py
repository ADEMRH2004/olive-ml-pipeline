"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   GEE Olive Harvest — Tunisia  |  Step 2: ML Pipeline                       ║
║   Two-stage per-zone classifier: olive detection + intensity classification  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  STAGE 1 — Binary:     olive (1) vs non-olive (0)   per zone               ║
║  STAGE 2 — Multiclass: extensif(1) / intensif(2) / hyper_intensif(3)       ║
║            only on olive pixels, SMOTE-balanced, per zone                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ENVIRONMENT                                                                 ║
║   Recommended : Mac M1/M2  (MPS backend, 16 GB unified memory)             ║
║   Alternative : Windows + GTX 1650  (CUDA, works but slower on tabular)    ║
║   Fallback    : Google Colab free tier  (reduce N_ESTIMATORS to 200)       ║
║                                                                              ║
║  INSTALL                                                                     ║
║   pip install lightgbm scikit-learn imbalanced-learn shapely pandas         ║
║               numpy matplotlib seaborn joblib tqdm                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import json
import warnings
import hashlib
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tqdm import tqdm

from shapely.geometry import Point, Polygon, mapping
from shapely.ops import unary_union

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, roc_auc_score, ConfusionMatrixDisplay,
)
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

import lightgbm as lgb
import joblib

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    # ── Paths ──────────────────────────────────────────────────────────────
    "csv_path":        "./data/gee_outputs/OliveTunisia_TrainingSamples_AllZones.csv",
    "gt_json_path":    "./data/ground_truth/parcels.json",   # your parcel JSON
    "output_dir":      "./models",
    "plots_dir":       "./plots",

    # ── Feature sets ───────────────────────────────────────────────────────
    # 16 GEE bands + 2 engineered (EVI, NDVI_cv) + 2 new layers (canopy_*)
    "spectral_features": [
        "NDVI_mean", "NDVI_max", "NDVI_min", "NDVI_amplitude",
        "NDVI_p10",  "NDVI_p50", "NDVI_p90",
        "NDWI_mean", "NDRE_mean", "FAPAR_mean", "LAI_mean",
        "B3", "B4", "B8", "B5", "B11",
        "EVI",        # engineered
        "NDVI_cv",    # engineered
        "canopy_height",   # Meta/WRI layer (added in GEE Step 1 update)
        "canopy_cover",    # derived in GEE Step 1 update
    ],
    "binary_features":    None,   # filled at runtime = spectral_features
    "intensity_features": None,   # filled at runtime = spectral_features

    # ── Intensity thresholds (rule-based labelling, tuned per zone) ────────
    # Overridden by ground-truth spatial join where parcels overlap
    "intensity_rules": {
        "extensif":       {"canopy_cover_max": 0.20, "ndvi_mean_max": 0.22},
        "hyper_intensif": {"ndwi_mean_min": -0.05,
                           "ndvi_amplitude_max": 0.12,
                           "canopy_height_max": 6.0},
        # intensif = everything else among olive pixels
    },

    # ── Split strategy ─────────────────────────────────────────────────────
    "train_years": [2020, 2021, 2022],
    "test_year":   2023,

    # ── Model hyperparameters ──────────────────────────────────────────────
    # Tuned for ≤ 1.5 h total on M1 / Colab free tier
    "lgbm_binary_params": {
        "objective":        "binary",
        "metric":           "auc",
        "n_estimators":     400,
        "learning_rate":    0.05,
        "num_leaves":       63,
        "max_depth":        -1,
        "min_child_samples": 20,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "reg_alpha":        0.1,
        "reg_lambda":       1.0,
        "n_jobs":           -1,
        "random_state":     42,
        "verbose":          -1,
    },
    "lgbm_intensity_params": {
        "objective":        "multiclass",
        "num_class":        3,          # extensif / intensif / hyper
        "metric":           "multi_logloss",
        "n_estimators":     400,
        "learning_rate":    0.05,
        "num_leaves":       63,
        "max_depth":        -1,
        "min_child_samples": 10,        # smaller: hyper class is rare
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "class_weight":     "balanced",
        "n_jobs":           -1,
        "random_state":     42,
        "verbose":          -1,
    },

    # ── SMOTE ──────────────────────────────────────────────────────────────
    "smote_k_neighbors": 5,
    "smote_random_state": 42,

    # ── Misc ───────────────────────────────────────────────────────────────
    "random_state":  42,
    "zones":         ["SFAX", "SAHEL", "MIDWEST", "NORTH", "SOUTH"],
    "intensity_map": {1: "extensif", 2: "intensif", 3: "hyper_intensif"},
    "intensity_inv": {"extensif": 1, "intensif": 2, "hyper_intensif": 3},
}

CFG["binary_features"]    = CFG["spectral_features"]
CFG["intensity_features"] = CFG["spectral_features"]

Path(CFG["output_dir"]).mkdir(parents=True, exist_ok=True)
Path(CFG["plots_dir"]).mkdir(parents=True, exist_ok=True)

np.random.seed(CFG["random_state"])

# ─────────────────────────────────────────────────────────────────────────────
# 2.  MOCK DATA GENERATOR
#     Produces a realistic CSV + parcel JSON when real GEE data isn't ready.
#     Matches exactly the schema Step 1 exports.
# ─────────────────────────────────────────────────────────────────────────────

def _zone_spectral_profile(zone: str, intensity: int, n: int, rng) -> dict:
    """
    Return per-column arrays mimicking real spectral profiles for each
    zone × intensity combination.  Values calibrated against published
    Sentinel-2 statistics for Tunisian olive orchards.
    """
    # Base NDVI by intensity (0=non-olive, 1=extensif, 2=intensif, 3=hyper)
    ndvi_base = {0: 0.08, 1: 0.18, 2: 0.32, 3: 0.48}[intensity]
    ndvi_std  = {0: 0.04, 1: 0.06, 2: 0.05, 3: 0.04}[intensity]

    # Zone-level adjustment (SOUTH drier → lower NDVI)
    zone_adj = {"SFAX": 0.0, "SAHEL": 0.03, "MIDWEST": 0.02,
                "NORTH": 0.05, "SOUTH": -0.04}[zone]

    ndvi_mean = np.clip(rng.normal(ndvi_base + zone_adj, ndvi_std, n), 0.01, 0.85)
    ndvi_amp  = np.clip(rng.normal(
        {0: 0.05, 1: 0.20, 2: 0.14, 3: 0.08}[intensity], 0.03, n), 0.01, 0.45)

    ndwi_base = {0: -0.30, 1: -0.22, 2: -0.12, 3: -0.03}[intensity]
    ndwi_mean = np.clip(rng.normal(ndwi_base, 0.05, n), -0.6, 0.2)

    ndre_mean = np.clip(ndvi_mean * 0.55 + rng.normal(0, 0.02, n), 0.0, 0.5)
    lai_mean  = np.clip(ndvi_mean * 4.5  + rng.normal(0, 0.3,  n), 0.0, 8.0)
    fapar     = np.clip(ndvi_mean * 1.24 - 0.168, 0, 1)

    b8  = np.clip(ndvi_mean * 0.35 + 0.15 + rng.normal(0, 0.02, n), 0.05, 0.9)
    b4  = np.clip(b8 * (1 - ndvi_mean) / (1 + ndvi_mean), 0.01, 0.5)
    b3  = b4 * rng.uniform(0.85, 1.05, n)
    b5  = b4 * 1.15 + rng.normal(0, 0.01, n)
    b11 = np.clip(b8 - ndwi_mean * b8, 0.01, 0.8)

    evi = np.clip(
        2.5 * (b8 - b4) / (b8 + 6 * b4 - 7.5 * b3 + 1 + 1e-8), -1, 1)
    ndvi_cv = np.where(ndvi_mean > 0.01, ndvi_amp / ndvi_mean, 0.0)

    ch_base = {0: 1.0, 1: 5.5, 2: 4.5, 3: 3.5}[intensity]
    canopy_h = np.clip(rng.normal(ch_base, 0.8, n), 0.5, 12.0)
    canopy_c = np.clip((ndvi_mean - 0.10) / (0.65 - 0.10), 0.0, 1.0)

    return {
        "NDVI_mean": ndvi_mean, "NDVI_max": ndvi_mean + ndvi_amp * 0.6,
        "NDVI_min":  ndvi_mean - ndvi_amp * 0.4, "NDVI_amplitude": ndvi_amp,
        "NDVI_p10":  ndvi_mean - ndvi_amp * 0.35,
        "NDVI_p50":  ndvi_mean,
        "NDVI_p90":  ndvi_mean + ndvi_amp * 0.35,
        "NDWI_mean": ndwi_mean, "NDRE_mean": ndre_mean,
        "FAPAR_mean": fapar,    "LAI_mean":  lai_mean,
        "B3": b3, "B4": b4, "B8": b8, "B5": b5, "B11": b11,
        "EVI": evi, "NDVI_cv": ndvi_cv,
        "canopy_height": canopy_h, "canopy_cover": canopy_c,
    }


def generate_mock_csv(path: str, n_per_zone_year: int = 1500) -> pd.DataFrame:
    """
    Generate a realistic mock CSV matching the Step 1 GEE export schema.
    intensity column: 0=non-olive, 1=extensif, 2=intensif, 3=hyper_intensif
    """
    rng   = np.random.default_rng(42)
    rows  = []
    years = [2020, 2021, 2022, 2023]

    # Zone-level intensity distribution (reflects real Tunisia density map)
    zone_intensity_dist = {
        "SFAX":    {0: 0.50, 1: 0.40, 2: 0.08, 3: 0.02},
        "SAHEL":   {0: 0.50, 1: 0.25, 2: 0.20, 3: 0.05},
        "MIDWEST": {0: 0.50, 1: 0.20, 2: 0.22, 3: 0.08},
        "NORTH":   {0: 0.50, 1: 0.30, 2: 0.18, 3: 0.02},
        "SOUTH":   {0: 0.50, 1: 0.45, 2: 0.04, 3: 0.01},
    }
    zone_ids = {"SFAX": 1, "SAHEL": 2, "MIDWEST": 3, "NORTH": 4, "SOUTH": 5}

    # Approximate zone bounding boxes for fake coordinates
    zone_bbox = {
        "SFAX":    (10.0, 33.9, 11.05, 35.25),
        "SAHEL":   (10.3, 35.15, 11.1, 36.25),
        "MIDWEST": (8.8, 34.4, 10.4, 36.1),
        "NORTH":   (8.4, 35.9, 11.2, 37.4),
        "SOUTH":   (8.1, 30.2, 11.6, 34.3),
    }

    for zone, zone_id in zone_ids.items():
        dist  = zone_intensity_dist[zone]
        bbox  = zone_bbox[zone]
        for year in years:
            counts = {k: int(v * n_per_zone_year) for k, v in dist.items()}
            for intensity, n in counts.items():
                if n == 0:
                    continue
                spec = _zone_spectral_profile(zone, intensity, n, rng)
                lats = rng.uniform(bbox[1], bbox[3], n)
                lngs = rng.uniform(bbox[0], bbox[2], n)
                for i in range(n):
                    row = {
                        "zone_name": zone,
                        "zone_id":   zone_id,
                        "year":      year,
                        "label":     int(intensity > 0),
                        "intensity": intensity,
                        "wc_class":  10 if intensity > 0 else 60,
                        ".geo":      json.dumps({
                            "type": "Point",
                            "coordinates": [float(lngs[i]), float(lats[i])]
                        }),
                    }
                    for feat, arr in spec.items():
                        row[feat] = float(arr[i])
                    rows.append(row)

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  ✅  Mock CSV written → {path}  ({len(df):,} rows)")
    return df


def generate_mock_gt_json(path: str) -> dict:
    """
    Generate a mock ground-truth parcel JSON in the exact format
    described (id, name, crop, owner, notes, area_ha, coordinates).
    50 parcels spread across zones with known intensity.
    """
    rng = np.random.default_rng(7)
    intensity_labels = ["extensif", "intensif", "hyper_intensif"]

    # One small cluster per zone, known intensity
    zone_clusters = {
        "SFAX":    {"lat": 34.60, "lng": 10.40, "intensity": "extensif",      "n": 12},
        "SAHEL":   {"lat": 35.70, "lng": 10.65, "intensity": "intensif",      "n": 10},
        "MIDWEST": {"lat": 35.10, "lng": 9.50,  "intensity": "hyper_intensif","n": 10},
        "NORTH":   {"lat": 36.60, "lng": 9.20,  "intensity": "intensif",      "n": 8 },
        "SOUTH":   {"lat": 33.50, "lng": 9.80,  "intensity": "extensif",      "n": 10},
    }

    parcels = []
    pid = 1
    for zone, meta in zone_clusters.items():
        for _ in range(meta["n"]):
            clat = meta["lat"] + rng.uniform(-0.05, 0.05)
            clng = meta["lng"] + rng.uniform(-0.05, 0.05)
            # Small square parcel (~2–10 ha)
            half = rng.uniform(0.003, 0.010)
            coords = [
                {"lat": clat - half, "lng": clng - half},
                {"lat": clat - half, "lng": clng + half},
                {"lat": clat + half, "lng": clng + half},
                {"lat": clat + half, "lng": clng - half},
                {"lat": clat - half, "lng": clng - half},  # close ring
            ]
            area = (2 * half * 111) ** 2 * 100  # rough ha
            parcels.append({
                "id":          f"P{pid:04d}",
                "name":        f"Parcelle {zone} {pid}",
                "crop":        "olive",
                "owner":       f"Owner_{pid}",
                "notes":       f"Zone {zone} — {meta['intensity']} olive grove",
                "area_ha":     round(float(area), 2),
                "created_at":  "2023-01-15T10:00:00Z",
                "modified_at": "2023-06-01T10:00:00Z",
                "coordinates": coords,
                # ground truth label — present in your app data
                "_gt_intensity": meta["intensity"],
            })
            pid += 1

    gt = {
        "generated_at": datetime.now().isoformat(),
        "parcel_count": len(parcels),
        "parcels":      parcels,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(gt, f, indent=2)
    print(f"  ✅  Mock GT JSON written → {path}  ({len(parcels)} parcels)")
    return gt


# ─────────────────────────────────────────────────────────────────────────────
# 3.  DATA LOADING & CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def load_and_clean(csv_path: str) -> pd.DataFrame:
    """
    Load GEE CSV, run all cleaning steps, return clean DataFrame.

    Cleaning steps
    ──────────────
    1. Drop geometry column (.geo) — JSON string, useless for sklearn
    2. Drop rows with any NaN in feature columns (cloud-mask bleed-through)
    3. Remove reflectance outliers: B-bands must be in (0, 1]
    4. Flag NDVI_amplitude == 0 (dead pixel artifact) → drop
    5. Verify label balance per zone — warn if deviation > 55/45
    6. Parse lat/lng from .geo for spatial join (stored separately)
    """
    print("\n━━━  LOADING & CLEANING  ━━━")
    df = pd.read_csv(csv_path)
    print(f"  Loaded  {len(df):,} rows × {df.shape[1]} cols")

    # ── Extract coordinates before dropping .geo ──────────────────────────
    if ".geo" in df.columns:
        def _parse_geo(g):
            try:
                c = json.loads(g)["coordinates"]
                return float(c[1]), float(c[0])  # lat, lng
            except Exception:
                return np.nan, np.nan

        df[["lat", "lng"]] = pd.DataFrame(
            df[".geo"].apply(_parse_geo).tolist(), index=df.index
        )
        df.drop(columns=[".geo"], inplace=True)
    else:
        # If coordinates already split
        if "lat" not in df.columns:
            df["lat"] = np.nan
            df["lng"] = np.nan

    # ── Drop non-feature metadata columns temporarily for NaN check ───────
    meta_cols = ["zone_name", "zone_id", "year", "label",
                 "intensity", "wc_class", "lat", "lng"]
    feat_cols = [c for c in CFG["spectral_features"] if c in df.columns]
    missing_feats = [c for c in CFG["spectral_features"] if c not in df.columns]
    if missing_feats:
        print(f"  ⚠️   Missing feature columns (will be skipped): {missing_feats}")

    # ── 1. NaN rows ───────────────────────────────────────────────────────
    before = len(df)
    df.dropna(subset=feat_cols, inplace=True)
    print(f"  NaN drop        : {before - len(df):,} rows removed")

    # ── 2. Reflectance outliers (raw band values must be 0–1) ────────────
    band_cols = ["B3", "B4", "B5", "B8", "B11"]
    band_cols = [c for c in band_cols if c in df.columns]
    if band_cols:
        mask_valid = (df[band_cols] >= 0).all(axis=1) & (df[band_cols] <= 1).all(axis=1)
        before = len(df)
        df = df[mask_valid].copy()
        print(f"  Band outlier    : {before - len(df):,} rows removed")

    # ── 3. Dead pixel artifact (NDVI_amplitude == 0) ─────────────────────
    if "NDVI_amplitude" in df.columns:
        before = len(df)
        df = df[df["NDVI_amplitude"] > 0].copy()
        print(f"  Dead pixel drop : {before - len(df):,} rows removed")

    # ── 4. Label balance check ────────────────────────────────────────────
    print("\n  Label balance per zone:")
    for zone in CFG["zones"]:
        z = df[df["zone_name"] == zone]
        if len(z) == 0:
            continue
        olive_pct = z["label"].mean() * 100
        flag = " ⚠️  IMBALANCED" if abs(olive_pct - 50) > 5 else ""
        print(f"    {zone:<10} {len(z):>5,} samples  |  "
              f"olive {olive_pct:.1f}%  non-olive {100-olive_pct:.1f}%{flag}")

    print(f"\n  ✅  Clean dataset: {len(df):,} rows")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add EVI and NDVI_cv if not already present (they may already be
    in the CSV if added during GEE export update).
    """
    print("\n━━━  FEATURE ENGINEERING  ━━━")

    if "EVI" not in df.columns:
        if all(c in df.columns for c in ["B8", "B4", "B3"]):
            df["EVI"] = 2.5 * (df["B8"] - df["B4"]) / (
                df["B8"] + 6 * df["B4"] - 7.5 * df["B3"] + 1 + 1e-8
            )
            df["EVI"] = df["EVI"].clip(-1, 1)
            print("  ✅  EVI computed")
        else:
            print("  ⚠️   EVI skipped (missing B3/B4/B8)")

    if "NDVI_cv" not in df.columns:
        if all(c in df.columns for c in ["NDVI_amplitude", "NDVI_mean"]):
            df["NDVI_cv"] = np.where(
                df["NDVI_mean"] > 0.01,
                df["NDVI_amplitude"] / df["NDVI_mean"],
                0.0,
            )
            print("  ✅  NDVI_cv computed")

    # canopy_cover may not be in CSV if GEE Step 1 not yet updated
    if "canopy_cover" not in df.columns and "NDVI_mean" in df.columns:
        df["canopy_cover"] = ((df["NDVI_mean"] - 0.10) / (0.65 - 0.10)).clip(0, 1)
        print("  ✅  canopy_cover derived from NDVI_mean (GEE layer not present)")

    if "canopy_height" not in df.columns:
        # Rough proxy from LAI when GEE layer absent
        if "LAI_mean" in df.columns:
            df["canopy_height"] = (df["LAI_mean"] * 1.2 + 1.5).clip(0.5, 12)
            print("  ✅  canopy_height proxied from LAI_mean (GEE layer not present)")

    feat_cols = [c for c in CFG["spectral_features"] if c in df.columns]
    print(f"  Total features available: {len(feat_cols)} / {len(CFG['spectral_features'])}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5.  INTENSITY LABELLING
#     Rule-based → then overridden by ground-truth spatial join
# ─────────────────────────────────────────────────────────────────────────────

def _infer_intensity_from_notes(notes: str) -> str | None:
    """
    Parse intensity from parcel notes / crop field.
    Looks for keywords in French or English.
    """
    if not isinstance(notes, str):
        return None
    notes_l = notes.lower()
    # Hyper-intensive signals
    if any(k in notes_l for k in [
        "hyper", "hyper-intensif", "hyper intensif",
        "super high density", "shd", "ultra"
    ]):
        return "hyper_intensif"
    # Intensive
    if any(k in notes_l for k in [
        "intensif", "intensive", "semi-intensif",
        "high density", "irrigu", "irrigated"
    ]):
        return "intensif"
    # Extensive
    if any(k in notes_l for k in [
        "extensif", "extensive", "traditional", "traditionnel",
        "low density", "pluvial", "rainfed"
    ]):
        return "extensif"
    return None


def build_gt_index(gt_json_path: str) -> list[dict]:
    """
    Load the ground truth JSON and build a list of
    { polygon: Polygon, intensity_int: int }
    for fast spatial lookup.
    """
    with open(gt_json_path) as f:
        gt = json.load(f)

    index = []
    for p in gt["parcels"]:
        coords = p.get("coordinates", [])
        if len(coords) < 3:
            continue

        # Build Shapely polygon (lng, lat order for geographic ops)
        try:
            poly = Polygon([(c["lng"], c["lat"]) for c in coords])
            if not poly.is_valid:
                poly = poly.buffer(0)
        except Exception:
            continue

        # Determine intensity: check _gt_intensity first (mock), then notes
        intensity_str = p.get("_gt_intensity") or _infer_intensity_from_notes(
            (p.get("notes") or "") + " " + (p.get("crop") or "")
        )
        if intensity_str not in CFG["intensity_inv"]:
            continue

        index.append({
            "polygon":       poly,
            "intensity_int": CFG["intensity_inv"][intensity_str],
            "parcel_id":     p.get("id", "?"),
        })

    print(f"  ✅  GT index: {len(index)} valid parcels loaded")
    return index


def rule_based_intensity(row: pd.Series) -> int:
    """
    Assign intensity class from spectral thresholds.
    Returns 0 for non-olive pixels (label == 0).
    """
    if row["label"] == 0:
        return 0

    r = CFG["intensity_rules"]
    cc  = row.get("canopy_cover", 0.0)
    ndvi = row.get("NDVI_mean", 0.0)
    ndwi = row.get("NDWI_mean", -0.5)
    amp  = row.get("NDVI_amplitude", 0.5)
    ch   = row.get("canopy_height", 5.0)

    if cc < r["extensif"]["canopy_cover_max"] or ndvi < r["extensif"]["ndvi_mean_max"]:
        return 1  # extensif

    if (ndwi  > r["hyper_intensif"]["ndwi_mean_min"] and
        amp   < r["hyper_intensif"]["ndvi_amplitude_max"] and
        ch    < r["hyper_intensif"]["canopy_height_max"]):
        return 3  # hyper_intensif

    return 2  # intensif


def assign_intensity_labels(df: pd.DataFrame, gt_json_path: str) -> pd.DataFrame:
    """
    1. Apply rule-based intensity to ALL rows.
    2. Spatial join: override with ground-truth where a sample point
       falls inside a known parcel polygon.
    Reports how many labels were overridden.
    """
    print("\n━━━  INTENSITY LABELLING  ━━━")

    # Step 1: rule-based
    if "intensity" not in df.columns:
        print("  Applying rule-based intensity labels …")
        df["intensity"] = df.apply(rule_based_intensity, axis=1)
    else:
        print("  'intensity' column already present in CSV — skipping rule-based step")

    # Step 2: ground-truth override via spatial join
    if not Path(gt_json_path).exists():
        print(f"  ⚠️   GT JSON not found at {gt_json_path} — skipping spatial join")
        return df

    gt_index = build_gt_index(gt_json_path)
    if not gt_index:
        return df

    # Only check olive pixels that have coordinates
    olive_mask = (df["label"] == 1) & df["lat"].notna() & df["lng"].notna()
    olive_idx  = df.index[olive_mask]

    n_overridden = 0
    print(f"  Spatial joining {olive_mask.sum():,} olive samples against "
          f"{len(gt_index)} GT parcels …")

    for idx in tqdm(olive_idx, desc="  Spatial join", unit="pt"):
        pt = Point(df.at[idx, "lng"], df.at[idx, "lat"])
        for entry in gt_index:
            if entry["polygon"].contains(pt):
                df.at[idx, "intensity"] = entry["intensity_int"]
                n_overridden += 1
                break

    print(f"  ✅  {n_overridden:,} labels overridden by ground truth "
          f"({n_overridden/olive_mask.sum()*100:.1f}% of olive pixels)")

    # Intensity distribution summary
    print("\n  Intensity distribution (olive pixels only):")
    olive_df = df[df["label"] == 1]
    for code, name in CFG["intensity_map"].items():
        n = (olive_df["intensity"] == code).sum()
        pct = n / len(olive_df) * 100
        bar = "█" * int(pct / 2)
        print(f"    {name:<18} {n:>5,}  ({pct:5.1f}%)  {bar}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6.  ZONE-YEAR SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def split_by_year(df: pd.DataFrame, zone: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Temporal split for a single zone.
    Train: CFG['train_years']  |  Test: CFG['test_year']
    Returns (train_df, test_df).
    """
    z = df[df["zone_name"] == zone].copy()
    train = z[z["year"].isin(CFG["train_years"])].copy()
    test  = z[z["year"] == CFG["test_year"]].copy()
    return train, test


# ─────────────────────────────────────────────────────────────────────────────
# 7.  STAGE 1 — BINARY CLASSIFIER  (olive vs non-olive, per zone)
# ─────────────────────────────────────────────────────────────────────────────

def train_binary_zone(
    train: pd.DataFrame,
    test:  pd.DataFrame,
    zone:  str,
    feat_cols: list[str],
) -> tuple[lgb.LGBMClassifier, dict, StandardScaler]:
    """
    Train and evaluate a binary LightGBM for one zone.
    Returns (model, metrics_dict, scaler).
    """
    X_train = train[feat_cols].values
    y_train = train["label"].values
    X_test  = test[feat_cols].values
    y_test  = test["label"].values

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model = lgb.LGBMClassifier(**CFG["lgbm_binary_params"])
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(period=-1)],
    )

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "zone":      zone,
        "stage":     "binary",
        "n_train":   len(train),
        "n_test":    len(test),
        "f1_binary": round(f1_score(y_test, y_pred, average="binary"), 4),
        "auc":       round(roc_auc_score(y_test, y_proba), 4),
        "report":    classification_report(y_test, y_pred,
                         target_names=["non-olive", "olive"]),
    }
    return model, metrics, scaler


# ─────────────────────────────────────────────────────────────────────────────
# 8.  STAGE 2 — INTENSITY CLASSIFIER  (per zone, olive pixels only)
# ─────────────────────────────────────────────────────────────────────────────

def train_intensity_zone(
    train: pd.DataFrame,
    test:  pd.DataFrame,
    zone:  str,
    feat_cols: list[str],
) -> tuple[lgb.LGBMClassifier | None, dict, StandardScaler | None]:
    """
    Train and evaluate a 3-class LightGBM on olive-only pixels.
    Classes: 1=extensif, 2=intensif, 3=hyper_intensif → remapped to 0/1/2.
    SMOTE applied to training fold only.
    """
    # Filter olive pixels only
    tr_olive = train[train["label"] == 1].copy()
    te_olive = test[test["label"] == 1].copy()

    if len(tr_olive) < 30 or te_olive["intensity"].nunique() < 2:
        print(f"    ⚠️   {zone}: insufficient olive data for intensity model — skipping")
        return None, {"zone": zone, "stage": "intensity", "skipped": True}, None

    # Remap labels 1/2/3 → 0/1/2  (LightGBM multiclass needs 0-based)
    label_map     = {1: 0, 2: 1, 3: 2}
    label_map_inv = {0: 1, 1: 2, 2: 3}

    X_train = tr_olive[feat_cols].values
    y_train = tr_olive["intensity"].map(label_map).values
    X_test  = te_olive[feat_cols].values
    y_test  = te_olive["intensity"].map(label_map).values

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # SMOTE — applied to training only, never leaks into test
    unique, counts = np.unique(y_train, return_counts=True)
    min_samples = counts.min()
    k = min(CFG["smote_k_neighbors"], min_samples - 1)

    if k >= 1 and len(unique) >= 2:
        try:
            smote = SMOTE(k_neighbors=k,
                          random_state=CFG["smote_random_state"])
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"    SMOTE applied ({zone}): "
                  f"{dict(zip(*np.unique(y_train, return_counts=True)))}")
        except Exception as e:
            print(f"    ⚠️   SMOTE failed for {zone}: {e} — continuing without")

    model = lgb.LGBMClassifier(**CFG["lgbm_intensity_params"])
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(period=-1)],
    )

    y_pred = model.predict(X_test)
    target_names = ["extensif", "intensif", "hyper_intensif"]

    # Only report classes present in test
    labels_present = sorted(np.unique(np.concatenate([y_test, y_pred])))

    metrics = {
        "zone":       zone,
        "stage":      "intensity",
        "n_train_smote": len(X_train),
        "n_test":     len(te_olive),
        "macro_f1":   round(f1_score(y_test, y_pred, average="macro",
                                     labels=labels_present), 4),
        "report":     classification_report(
                          y_test, y_pred,
                          labels=labels_present,
                          target_names=[target_names[l] for l in labels_present],
                          zero_division=0,
                      ),
        "cm":         confusion_matrix(y_test, y_pred, labels=labels_present),
        "cm_labels":  [target_names[l] for l in labels_present],
    }
    return model, metrics, scaler


# ─────────────────────────────────────────────────────────────────────────────
# 9.  PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(model, feat_cols: list[str],
                            title: str, path: str) -> None:
    imp = pd.Series(model.feature_importances_, index=feat_cols).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    imp.plot.barh(ax=ax, color="#2ca25f")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Feature importance (split count)")
    ax.axvline(imp.mean(), color="red", linestyle="--", linewidth=0.8,
               label=f"mean = {imp.mean():.0f}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, labels: list[str],
                          title: str, path: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Greens", colorbar=False)
    ax.set_title(title, fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_zone_summary(all_metrics: list[dict], path: str) -> None:
    """
    Side-by-side bar chart: binary AUC and intensity macro-F1 per zone.
    """
    zones   = CFG["zones"]
    auc_map = {m["zone"]: m.get("auc", 0)
               for m in all_metrics if m["stage"] == "binary"}
    mf1_map = {m["zone"]: m.get("macro_f1", 0)
               for m in all_metrics
               if m["stage"] == "intensity" and not m.get("skipped")}

    x   = np.arange(len(zones))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(9, 4))
    b1  = ax.bar(x - w/2, [auc_map.get(z, 0) for z in zones],
                 w, label="Stage-1 AUC (binary)", color="#2171b5")
    b2  = ax.bar(x + w/2, [mf1_map.get(z, 0) for z in zones],
                 w, label="Stage-2 Macro-F1 (intensity)", color="#2ca25f")
    ax.set_xticks(x)
    ax.set_xticklabels(zones)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-Zone Performance Summary", fontweight="bold")
    ax.axhline(0.85, color="red", linestyle="--", linewidth=0.7, label="0.85 target")
    ax.legend(fontsize=8)
    ax.bar_label(b1, fmt="%.2f", fontsize=7, padding=2)
    ax.bar_label(b2, fmt="%.2f", fontsize=7, padding=2)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📊  Zone summary plot saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 10.  PREDICTION PIPELINE
#      Takes a new CSV (e.g. 2024 export) → outputs GeoJSON with predictions
# ─────────────────────────────────────────────────────────────────────────────

def predict_new_csv(
    csv_path: str,
    model_dir: str,
    output_geojson: str,
) -> None:
    """
    Load a new GEE export CSV, run per-zone two-stage prediction,
    and write a GeoJSON FeatureCollection with predicted class fields.

    Output GeoJSON properties per feature:
        zone_name, year, lat, lng,
        pred_binary (0/1),
        pred_intensity (0=non-olive / 1=extensif / 2=intensif / 3=hyper)
        pred_intensity_label (string)
    """
    print(f"\n━━━  PREDICTION on {csv_path}  ━━━")
    df = pd.read_csv(csv_path)
    df = engineer_features(df)

    # Parse .geo if present
    if ".geo" in df.columns:
        def _pg(g):
            try:
                c = json.loads(g)["coordinates"]
                return float(c[1]), float(c[0])
            except Exception:
                return np.nan, np.nan
        df[["lat", "lng"]] = pd.DataFrame(
            df[".geo"].apply(_pg).tolist(), index=df.index)
        df.drop(columns=[".geo"], inplace=True)

    features_out = []
    model_dir    = Path(model_dir)

    for zone in CFG["zones"]:
        z_df = df[df["zone_name"] == zone].copy()
        if len(z_df) == 0:
            continue

        # Load stage-1 artifacts
        bin_model_path  = model_dir / f"{zone}_stage1_binary.pkl"
        bin_scaler_path = model_dir / f"{zone}_stage1_scaler.pkl"
        if not bin_model_path.exists():
            print(f"  ⚠️   No binary model for {zone} — skipping")
            continue

        bin_model  = joblib.load(bin_model_path)
        bin_scaler = joblib.load(bin_scaler_path)

        feat_cols = [c for c in CFG["binary_features"] if c in z_df.columns]
        X = bin_scaler.transform(z_df[feat_cols].values)
        z_df["pred_binary"] = bin_model.predict(X)

        # Load stage-2 artifacts
        int_model_path  = model_dir / f"{zone}_stage2_intensity.pkl"
        int_scaler_path = model_dir / f"{zone}_stage2_scaler.pkl"

        z_df["pred_intensity"]       = 0
        z_df["pred_intensity_label"] = "non-olive"

        if int_model_path.exists():
            int_model  = joblib.load(int_model_path)
            int_scaler = joblib.load(int_scaler_path)
            olive_mask = z_df["pred_binary"] == 1
            if olive_mask.sum() > 0:
                feat_cols_i = [c for c in CFG["intensity_features"] if c in z_df.columns]
                Xi = int_scaler.transform(z_df.loc[olive_mask, feat_cols_i].values)
                preds_raw = int_model.predict(Xi)  # 0/1/2 → remap to 1/2/3
                preds     = preds_raw + 1
                z_df.loc[olive_mask, "pred_intensity"] = preds
                z_df.loc[olive_mask, "pred_intensity_label"] = [
                    CFG["intensity_map"][p] for p in preds]

        # Build GeoJSON features
        for _, row in z_df.iterrows():
            if pd.isna(row.get("lat")) or pd.isna(row.get("lng")):
                continue
            features_out.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(row["lng"]), float(row["lat"])],
                },
                "properties": {
                    "zone_name":            row.get("zone_name", ""),
                    "year":                 int(row.get("year", 0)),
                    "pred_binary":          int(row["pred_binary"]),
                    "pred_intensity":       int(row["pred_intensity"]),
                    "pred_intensity_label": str(row["pred_intensity_label"]),
                },
            })

    geojson = {"type": "FeatureCollection", "features": features_out}
    with open(output_geojson, "w") as f:
        json.dump(geojson, f)
    print(f"  ✅  {len(features_out):,} predictions written → {output_geojson}")


# ─────────────────────────────────────────────────────────────────────────────
# 11.  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_ml_pipeline():
    t0 = datetime.now()
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║   GEE Olive Tunisia — Step 2: ML Pipeline                        ║
║   {t0.strftime('%Y-%m-%d  %H:%M:%S')}                                          ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    # ── 0.  Generate mock data if real files absent ───────────────────────
    if not Path(CFG["csv_path"]).exists():
        print("  ℹ️   Real CSV not found → generating mock data …")
        generate_mock_csv(CFG["csv_path"])
    if not Path(CFG["gt_json_path"]).exists():
        print("  ℹ️   GT JSON not found → generating mock parcels …")
        generate_mock_gt_json(CFG["gt_json_path"])

    # ── 1.  Load & clean ──────────────────────────────────────────────────
    df = load_and_clean(CFG["csv_path"])

    # ── 2.  Feature engineering ───────────────────────────────────────────
    df = engineer_features(df)

    # ── 3.  Intensity labelling ───────────────────────────────────────────
    df = assign_intensity_labels(df, CFG["gt_json_path"])

    # ── 4.  Per-zone training loop ────────────────────────────────────────
    all_metrics  = []
    model_store  = {}   # zone → {stage1: model, stage2: model}
    output_dir   = Path(CFG["output_dir"])
    plots_dir    = Path(CFG["plots_dir"])

    feat_cols_b = [c for c in CFG["binary_features"]    if c in df.columns]
    feat_cols_i = [c for c in CFG["intensity_features"] if c in df.columns]

    print(f"\n━━━  TRAINING  ({len(CFG['zones'])} zones × 2 stages)  ━━━")
    print(f"  Binary features  : {len(feat_cols_b)}")
    print(f"  Intensity features: {len(feat_cols_i)}")
    print(f"  Train years      : {CFG['train_years']}")
    print(f"  Test year        : {CFG['test_year']}\n")

    for zone in CFG["zones"]:
        print(f"\n{'─'*60}")
        print(f"  ZONE: {zone}")
        print(f"{'─'*60}")

        train, test = split_by_year(df, zone)
        print(f"  Train: {len(train):,}  |  Test: {len(test):,}")

        if len(train) < 50 or len(test) < 10:
            print(f"  ⚠️   Insufficient data for {zone} — skipping")
            continue

        model_store[zone] = {}

        # ── Stage 1: Binary ───────────────────────────────────────────────
        print(f"\n  [Stage 1] Binary classifier …")
        m1, met1, sc1 = train_binary_zone(train, test, zone, feat_cols_b)
        all_metrics.append(met1)
        model_store[zone]["stage1"] = (m1, sc1)

        print(f"    AUC = {met1['auc']:.4f}   F1 = {met1['f1_binary']:.4f}")
        print(met1["report"])

        # Save stage-1 artifacts
        joblib.dump(m1,  output_dir / f"{zone}_stage1_binary.pkl")
        joblib.dump(sc1, output_dir / f"{zone}_stage1_scaler.pkl")

        # Feature importance plot
        plot_feature_importance(
            m1, feat_cols_b,
            f"{zone} — Stage 1 Binary Feature Importance",
            str(plots_dir / f"{zone}_stage1_feature_importance.png"),
        )

        # ── Stage 2: Intensity ────────────────────────────────────────────
        print(f"\n  [Stage 2] Intensity classifier (olive pixels only) …")
        m2, met2, sc2 = train_intensity_zone(train, test, zone, feat_cols_i)
        all_metrics.append(met2)

        if m2 is not None:
            model_store[zone]["stage2"] = (m2, sc2)
            print(f"    Macro-F1 = {met2['macro_f1']:.4f}")
            print(met2["report"])

            joblib.dump(m2,  output_dir / f"{zone}_stage2_intensity.pkl")
            joblib.dump(sc2, output_dir / f"{zone}_stage2_scaler.pkl")

            # Confusion matrix
            plot_confusion_matrix(
                met2["cm"], met2["cm_labels"],
                f"{zone} — Stage 2 Intensity Confusion Matrix",
                str(plots_dir / f"{zone}_stage2_confusion_matrix.png"),
            )
            plot_feature_importance(
                m2, feat_cols_i,
                f"{zone} — Stage 2 Intensity Feature Importance",
                str(plots_dir / f"{zone}_stage2_feature_importance.png"),
            )

    # ── 5.  Summary ───────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  OVERALL RESULTS")
    print(f"{'═'*60}")

    print("\n  Stage 1 — Binary (olive detection):")
    for m in all_metrics:
        if m["stage"] == "binary":
            print(f"    {m['zone']:<10}  AUC={m.get('auc',0):.3f}  "
                  f"F1={m.get('f1_binary',0):.3f}  "
                  f"(train={m['n_train']:,}  test={m['n_test']:,})")

    print("\n  Stage 2 — Intensity classification:")
    for m in all_metrics:
        if m["stage"] == "intensity":
            if m.get("skipped"):
                print(f"    {m['zone']:<10}  SKIPPED")
            else:
                print(f"    {m['zone']:<10}  macro-F1={m.get('macro_f1',0):.3f}  "
                      f"(test={m['n_test']:,})")

    # Summary plot
    plot_zone_summary(all_metrics,
                      str(plots_dir / "zone_performance_summary.png"))

    # Save metrics JSON
    metrics_path = output_dir / "metrics.json"
    serializable = []
    for m in all_metrics:
        sm = {k: v for k, v in m.items() if k != "cm"}
        if "cm" in m:
            sm["cm"] = m["cm"].tolist()
        serializable.append(sm)
    with open(metrics_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  📋  Metrics saved → {metrics_path}")

    elapsed = (datetime.now() - t0).seconds
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  PIPELINE COMPLETE                                               ║
║  Elapsed: {elapsed//60:02d}m {elapsed%60:02d}s                                         ║
║                                                                  ║
║  Models saved  → ./models/                                       ║
║    <ZONE>_stage1_binary.pkl   × {len(CFG['zones'])} zones                 ║
║    <ZONE>_stage2_intensity.pkl× {len(CFG['zones'])} zones                 ║
║                                                                  ║
║  Plots saved   → ./plots/                                        ║
║    zone_performance_summary.png                                  ║
║    <ZONE>_stage1_feature_importance.png                          ║
║    <ZONE>_stage2_confusion_matrix.png                            ║
║                                                                  ║
║  NEXT — run predict_new_csv() with your 2024 GEE export          ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    return model_store, all_metrics


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    models, metrics = run_ml_pipeline()

    # ── Uncomment to predict on a new year's export ───────────────────────
    # predict_new_csv(
    #     csv_path       = "./data/gee_outputs/OliveTunisia_2024_export.csv",
    #     model_dir      = "./models",
    #     output_geojson = "./data/predictions/olive_predictions_2024.geojson",
    # )
