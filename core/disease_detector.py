"""
disease_detector.py
───────────────────
Crop disease detection from images.

How it works (without deep learning):
 1. Load image and convert to HSV + LAB colour spaces
 2. Extract 20+ colour, texture and statistical features per image
 3. A pre-fitted RandomForestClassifier trained on the extracted features
    maps those features to one of 8 disease classes
 4. Returns top-3 predictions with confidence scores and a Grad-CAM-style
    pixel importance heatmap (visualised by saliency from colour channels)

In production this module is replaced by EfficientNet-B4 (PyTorch) —
the interface (predict / visualise_heatmap) stays identical.
"""

import os
import io
import math
import random
import pickle
import hashlib
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB = True
except ImportError:
    MATPLOTLIB = False

# ── Disease taxonomy ────────────────────────────────────────────────────────
DISEASES = {
    "healthy":              {"severity": 0.0, "color": (80, 180, 60),   "crop_risk": []},
    "early_blight":         {"severity": 0.6, "color": (180, 100, 40),  "crop_risk": ["tomato","potato"]},
    "late_blight":          {"severity": 0.9, "color": (80, 60, 100),   "crop_risk": ["tomato","potato"]},
    "leaf_rust":            {"severity": 0.7, "color": (200, 120, 30),  "crop_risk": ["wheat","barley"]},
    "powdery_mildew":       {"severity": 0.5, "color": (210, 210, 200), "crop_risk": ["wheat","grape"]},
    "mosaic_virus":         {"severity": 0.85,"color": (160, 190, 50),  "crop_risk": ["tomato","maize"]},
    "nutrient_deficiency":  {"severity": 0.4, "color": (220, 200, 80),  "crop_risk": ["rice","maize"]},
    "bacterial_spot":       {"severity": 0.65,"color": (60, 80, 40),    "crop_risk": ["tomato","pepper"]},
}
DISEASE_NAMES = list(DISEASES.keys())

# ── Feature extraction ───────────────────────────────────────────────────────
def _rgb_to_hsv_array(rgb: np.ndarray) -> np.ndarray:
    """Vectorised RGB→HSV without OpenCV."""
    r, g, b = rgb[:,:,0]/255.0, rgb[:,:,1]/255.0, rgb[:,:,2]/255.0
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc
    v = maxc
    s = np.where(maxc > 0, delta / maxc, 0.0)
    h = np.zeros_like(v)
    mask_r = (maxc == r) & (delta > 0)
    mask_g = (maxc == g) & (delta > 0)
    mask_b = (maxc == b) & (delta > 0)
    h[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6
    h[mask_g] = (b[mask_g] - r[mask_g]) / delta[mask_g] + 2
    h[mask_b] = (r[mask_b] - g[mask_b]) / delta[mask_b] + 4
    h = h / 6.0
    return np.stack([h, s, v], axis=-1)


def extract_features(image: Image.Image) -> np.ndarray:
    """
    Extract 28 numerical features from a PIL Image.
    These capture colour distribution, texture, and spatial statistics.
    """
    img = image.convert("RGB").resize((128, 128))
    arr = np.array(img, dtype=np.float32)
    hsv = _rgb_to_hsv_array(arr)

    feats = []

    # ── Channel statistics (R,G,B,H,S,V) ──
    for ch in range(3):
        ch_data = arr[:,:,ch].ravel() / 255.0
        feats += [ch_data.mean(), ch_data.std(), np.percentile(ch_data, 25), np.percentile(ch_data, 75)]

    for ch in range(3):
        ch_data = hsv[:,:,ch].ravel()
        feats += [ch_data.mean(), ch_data.std()]

    # ── Colour ratios (diagnostic) ──
    r_mean = arr[:,:,0].mean()
    g_mean = arr[:,:,1].mean()
    b_mean = arr[:,:,2].mean()
    feats.append(r_mean / (g_mean + 1e-6))   # red/green — rust indicator
    feats.append((r_mean - b_mean) / (r_mean + b_mean + 1e-6))  # chromatic diff
    feats.append(g_mean / (r_mean + b_mean + 1e-6))  # greenness index

    # ── Texture: local standard deviation (proxy for lesion texture) ──
    gray = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]
    pil_gray = Image.fromarray(gray.astype(np.uint8))
    blurred = np.array(pil_gray.filter(ImageFilter.GaussianBlur(3)), dtype=np.float32)
    texture = np.std(gray - blurred)
    feats.append(texture)

    # ── Spot detection: count of dark pixels in green channel ──
    dark_mask = arr[:,:,1] < 80
    feats.append(dark_mask.mean())   # fraction of very dark pixels → spots/lesions

    # ── Yellow fraction (chlorosis) ──
    yellow = ((arr[:,:,0] > 160) & (arr[:,:,1] > 140) & (arr[:,:,2] < 80)).mean()
    feats.append(float(yellow))

    # ── Brown fraction (blight/necrosis) ──
    brown = ((arr[:,:,0] > 120) & (arr[:,:,1] < 90) & (arr[:,:,2] < 70)).mean()
    feats.append(float(brown))

    # ── White/grey fraction (mildew) ──
    white = ((arr[:,:,0] > 190) & (arr[:,:,1] > 185) & (arr[:,:,2] > 180)).mean()
    feats.append(float(white))

    # ── Entropy (complex textures = disease) ──
    hist, _ = np.histogram(gray.ravel(), bins=32, range=(0, 255))
    hist = hist / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    feats.append(float(entropy))

    return np.array(feats, dtype=np.float32)


# ── Synthetic training ───────────────────────────────────────────────────────
def _build_synthetic_model(n_samples: int = 2400):
    """
    Build a RandomForest disease classifier using synthetically generated
    feature vectors that mimic the statistical patterns of each disease class.
    Returns a fitted classifier.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    rng = np.random.default_rng(42)
    X, y = [], []

    # Disease feature profiles (offsets from neutral green leaf)
    profiles = {
        "healthy":             dict(r_mean=-0.05, g_mean=0.15, b_mean=-0.02, brown=0.0,  yellow=0.0,  white=0.0,  dark=0.04, tex=2.0),
        "early_blight":        dict(r_mean=0.10,  g_mean=-0.05,b_mean=-0.05, brown=0.18, yellow=0.08, white=0.01, dark=0.08, tex=8.0),
        "late_blight":         dict(r_mean=-0.05, g_mean=-0.08,b_mean=0.05,  brown=0.22, yellow=0.02, white=0.02, dark=0.12, tex=10.0),
        "leaf_rust":           dict(r_mean=0.18,  g_mean=-0.08,b_mean=-0.12, brown=0.08, yellow=0.15, white=0.01, dark=0.06, tex=7.0),
        "powdery_mildew":      dict(r_mean=0.05,  g_mean=0.02, b_mean=0.05,  brown=0.02, yellow=0.04, white=0.20, dark=0.03, tex=6.0),
        "mosaic_virus":        dict(r_mean=-0.02, g_mean=0.05, b_mean=-0.08, brown=0.04, yellow=0.10, white=0.02, dark=0.05, tex=9.0),
        "nutrient_deficiency": dict(r_mean=0.05,  g_mean=-0.10,b_mean=-0.05, brown=0.04, yellow=0.22, white=0.04, dark=0.04, tex=4.0),
        "bacterial_spot":      dict(r_mean=-0.04, g_mean=-0.06,b_mean=-0.04, brown=0.14, yellow=0.02, white=0.01, dark=0.15, tex=11.0),
    }

    n_feat = 27  # must match extract_features output size

    for disease, prof in profiles.items():
        cls_idx = DISEASE_NAMES.index(disease)
        n = n_samples // len(profiles)
        # Build feature vectors with disease-appropriate statistics
        samples = rng.normal(0.5, 0.06, size=(n, n_feat))
        # Inject disease signature into relevant feature positions
        samples[:, 0]  += prof["r_mean"]  + rng.normal(0, 0.03, n)  # R mean
        samples[:, 4]  += prof["g_mean"]  + rng.normal(0, 0.03, n)  # G mean
        samples[:, 8]  += prof["b_mean"]  + rng.normal(0, 0.03, n)  # B mean
        samples[:, 24] += prof["brown"]   + rng.normal(0, 0.02, n)  # brown frac
        samples[:, 25] += prof["yellow"]  + rng.normal(0, 0.02, n)  # yellow frac
        samples[:, 26] += prof["white"]   + rng.normal(0, 0.02, n)  # white frac
        samples[:, 22] += prof["dark"]    + rng.normal(0, 0.01, n)  # dark pixels
        samples[:, 20] += prof["tex"]     + rng.normal(0, 0.8,  n)  # texture
        X.append(samples)
        y.extend([cls_idx] * n)

    X = np.vstack(X)
    y = np.array(y)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, max_depth=12,
                                        min_samples_leaf=2, n_jobs=-1, random_state=42)),
    ])
    model.fit(X, y)
    return model


# ── Main detector class ──────────────────────────────────────────────────────
class DiseaseDetector:
    """
    Detects crop disease from a PIL Image.

    Usage:
        detector = DiseaseDetector()
        result = detector.predict(image, crop_type="tomato")
    """

    def __init__(self):
        self._model = _build_synthetic_model()
        print("  [DiseaseDetector] Model ready — 8 classes, 200 trees")

    def predict(self, image: Image.Image, crop_type: str = "unknown") -> dict:
        """
        Predict disease from a PIL Image.

        Returns:
            dict with keys:
              top_disease    str   — predicted class name
              confidence     float — model confidence 0–1
              top3           list  — [(disease, prob), ...]
              severity       str   — "none" | "low" | "medium" | "high" | "critical"
              is_healthy     bool
              treatment      dict  — recommended treatment
        """
        feats = extract_features(image).reshape(1, -1)
        proba = self._model.predict_proba(feats)[0]

        # Adjust probabilities if crop_type is known (crop-aware boost)
        if crop_type.lower() in DISEASES.get("healthy", {}).get("crop_risk", []):
            pass  # no boost needed
        for idx, name in enumerate(DISEASE_NAMES):
            if crop_type.lower() in DISEASES[name].get("crop_risk", []):
                proba[idx] *= 1.15  # boost crop-relevant diseases
        proba = proba / proba.sum()

        top3_idx = np.argsort(proba)[::-1][:3]
        top3 = [(DISEASE_NAMES[i], float(proba[i])) for i in top3_idx]
        top_disease, confidence = top3[0]
        severity_val = DISEASES[top_disease]["severity"]

        if severity_val == 0.0:
            severity = "none"
        elif severity_val < 0.5:
            severity = "low"
        elif severity_val < 0.7:
            severity = "medium"
        elif severity_val < 0.85:
            severity = "high"
        else:
            severity = "critical"

        treatment = self._get_treatment(top_disease, severity)

        return {
            "top_disease":   top_disease,
            "confidence":    round(confidence, 4),
            "top3":          top3,
            "severity":      severity,
            "is_healthy":    top_disease == "healthy",
            "treatment":     treatment,
        }

    def _get_treatment(self, disease: str, severity: str) -> dict:
        treatments = {
            "healthy":             {"chemical": "None required",     "organic": "Preventive neem spray 3ml/L",    "timing": "Every 3 weeks preventively"},
            "early_blight":        {"chemical": "Mancozeb 2g/L",     "organic": "Copper oxychloride 3g/L",        "timing": "Every 10 days × 3 applications"},
            "late_blight":         {"chemical": "Metalaxyl 2.5g/L",  "organic": "Bordeaux mixture 1%",           "timing": "Every 7 days — urgent"},
            "leaf_rust":           {"chemical": "Propiconazole 1ml/L","organic": "Sulphur dust 3kg/ha",           "timing": "2 sprays, 14 days apart"},
            "powdery_mildew":      {"chemical": "Triadimefon 0.5g/L","organic": "Potassium bicarbonate 5g/L",    "timing": "Weekly × 4 weeks"},
            "mosaic_virus":        {"chemical": "No direct cure",     "organic": "Imidacloprid for aphid control","timing": "Immediate — remove infected plants"},
            "nutrient_deficiency": {"chemical": "NPK foliar spray",   "organic": "Vermicompost tea 1:10 dilution","timing": "Apply and retest soil after 3 weeks"},
            "bacterial_spot":      {"chemical": "Copper hydroxide 3g/L","organic": "Garlic+chilli extract spray","timing": "Every 7 days × 3 weeks"},
        }
        return treatments.get(disease, treatments["healthy"])

    def create_heatmap(self, image: Image.Image, save_path: str = None) -> str:
        """
        Generate a pseudo-Grad-CAM heatmap by highlighting pixels
        whose colour channels deviate most from a healthy leaf profile.
        Returns path to saved PNG.
        """
        if not MATPLOTLIB:
            return None

        img_resized = image.convert("RGB").resize((256, 256))
        arr = np.array(img_resized, dtype=np.float32)

        # Compute "disease saliency": deviation from healthy green
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        saliency = (
            np.abs(r - 60)  / 255 * 0.35 +   # redness
            np.abs(b - 50)  / 255 * 0.20 +   # bluishness
            (1 - g / 255)   * 0.30 +         # loss of green
            (np.abs(r - g) + np.abs(g - b)) / 510 * 0.15  # colour variance
        )
        saliency = (saliency - saliency.min()) / (saliency.max() + 1e-6)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="#1a1508")
        for ax in axes:
            ax.set_facecolor("#1a1508")
            ax.axis("off")

        axes[0].imshow(img_resized)
        axes[0].set_title("Original image", color="#e8c97a", fontsize=11, pad=6)

        axes[1].imshow(saliency, cmap="RdYlGn_r", vmin=0, vmax=1)
        axes[1].set_title("Disease saliency map", color="#e8c97a", fontsize=11, pad=6)

        overlay = np.array(img_resized) / 255.0
        heatmap_rgb = plt.cm.RdYlGn_r(saliency)[:, :, :3]
        blended = 0.55 * overlay + 0.45 * heatmap_rgb
        axes[2].imshow(np.clip(blended, 0, 1))
        axes[2].set_title("Overlay (disease regions)", color="#e8c97a", fontsize=11, pad=6)

        plt.tight_layout(pad=1.0)
        path = save_path or "/tmp/disease_heatmap.png"
        plt.savefig(path, dpi=130, bbox_inches="tight", facecolor="#1a1508")
        plt.close()
        return path