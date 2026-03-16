"""
yield_predictor.py
──────────────────
Crop yield prediction using a stacked ensemble:
  - GradientBoostingRegressor (base learner)
  - Ridge meta-learner on CV out-of-fold predictions
  - Feature importance via permutation (no SHAP dependency)

Trained on 3,000 synthetic farm records that encode:
  - Liebig's Law of the Minimum
  - Quadratic temperature optimum
  - Rainfall logistic response curve
  - Soil pH × nutrient interaction effects
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Optional


# ── Synthetic dataset ────────────────────────────────────────────────────────
CROP_META = {
    "wheat":     {"base": 4.5,  "N_opt": 120, "T_opt": 18, "R_opt": 40},
    "rice":      {"base": 5.0,  "N_opt": 140, "T_opt": 26, "R_opt": 80},
    "maize":     {"base": 6.0,  "N_opt": 160, "T_opt": 24, "R_opt": 60},
    "tomato":    {"base": 40.0, "N_opt": 130, "T_opt": 22, "R_opt": 55},
    "potato":    {"base": 25.0, "N_opt": 90,  "T_opt": 17, "R_opt": 50},
    "cotton":    {"base": 2.5,  "N_opt": 120, "T_opt": 26, "R_opt": 50},
    "soybean":   {"base": 3.0,  "N_opt": 60,  "T_opt": 22, "R_opt": 55},
    "sugarcane": {"base": 70.0, "N_opt": 180, "T_opt": 28, "R_opt": 100},
}


def _generate_training_data(n: int = 3000):
    """
    Generate realistic synthetic farm records.
    Yield is computed from agronomically-grounded equations.
    """
    from sklearn.preprocessing import LabelEncoder
    rng = np.random.default_rng(42)
    crops = list(CROP_META.keys())
    rows = []

    for _ in range(n):
        crop = rng.choice(crops)
        meta = CROP_META[crop]
        cidx = crops.index(crop)

        N   = rng.uniform(10, 200)
        P   = rng.uniform(5,  100)
        K   = rng.uniform(10, 200)
        pH  = rng.uniform(4.5, 8.5)
        OM  = rng.uniform(0.5, 5.0)
        T   = rng.uniform(10, 38)
        R   = rng.uniform(5,  150)
        H   = rng.uniform(30, 95)
        sun = rng.uniform(4,  12)

        # Feature engineering (mirrors soil_weather.py)
        ph_avail   = max(0.05, 1.0 - abs(pH - 6.5) * 0.22)
        soil_h     = (min(1, N/meta["N_opt"])*0.30 + min(1, P/(meta["N_opt"]*0.35))*0.20 +
                      min(1, K/meta["N_opt"])*0.20 + max(0, 1-abs(pH-6.5)/3)*0.20 + min(1, OM/4)*0.10)
        t_fac      = max(0.05, 1.0 - ((T - meta["T_opt"])**2) / 150)
        water_sc   = min(1.2, R / meta["R_opt"])
        climate_sc = t_fac * 0.45 + min(1, water_sc) * 0.35 + min(1, sun/10) * 0.20
        liebig     = min(N/meta["N_opt"], P/(meta["N_opt"]*0.35), K/meta["N_opt"])
        aridity    = R / (T * 20 + 1)
        vpd        = T * (1 - H / 100)
        gdd        = max(0, T - 10)
        npk_total  = N + P + K
        n_p_ratio  = N / (P + 1e-6)
        om_score   = min(1, OM / 4.0)

        # Yield formula (multiplicative, agronomically grounded)
        r_excess = max(0.0, (R - meta["R_opt"]*1.4) / meta["R_opt"] * 0.3)
        r_fac    = max(0.1, 1/(1+math.exp(-0.06*(R-meta["R_opt"]*0.5))) - r_excess)
        y = (meta["base"] * liebig * ph_avail * (1 + OM*0.04) * t_fac * r_fac * (0.7 + 0.03*sun))
        y = max(0.01, y * rng.normal(1.0, 0.08))

        rows.append([
            soil_h, climate_sc, ph_avail, aridity, vpd, gdd,
            water_sc, t_fac, npk_total, n_p_ratio, N/meta["N_opt"],
            liebig, om_score, float(cidx),
            y
        ])

    X = np.array(rows)[:, :-1]
    y = np.array(rows)[:, -1]
    return X, y


FEATURE_NAMES = [
    "soil_health_score", "climate_score", "ph_availability",
    "aridity_index", "vpd_proxy", "gdd_proxy", "water_score",
    "temp_factor", "npk_total", "n_p_ratio", "n_ratio",
    "liebig_minimum", "om_score", "crop_index",
]


def _build_yield_model():
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict

    X, y = _generate_training_data(3000)

    # Base learner
    gb = Pipeline([
        ("scaler", StandardScaler()),
        ("gb", GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.06,
            max_depth=5, subsample=0.8,
            min_samples_leaf=5, random_state=42,
        )),
    ])

    # Stacking: use CV predictions as meta-features
    oof_preds = cross_val_predict(gb, X, y, cv=3)
    gb.fit(X, y)

    meta = Ridge(alpha=1.0)
    meta.fit(oof_preds.reshape(-1, 1), y)

    return gb, meta, X, y


class YieldPredictor:
    """
    Predicts crop yield range (min, expected, max) in t/ha.

    Also provides:
      - Feature importance ranking
      - Key limiting factor identification
      - Improvement potential estimate
    """

    def __init__(self):
        self._gb, self._meta, self._X, self._y = _build_yield_model()
        self._importance = self._compute_importance()
        print("  [YieldPredictor] GBM + Ridge stacked model ready")

    def predict(self, features: dict, crop: str = "wheat") -> dict:
        """
        Predict yield from engineered features dict.

        Args:
            features: dict from SoilWeatherAnalyser.analyse()["engineered_features"]
            crop:     crop name string

        Returns:
            dict with predicted_yield, range, confidence, limiting_factor, improvement_potential
        """
        crops = list(CROP_META.keys())
        crop_idx = crops.index(crop) if crop in crops else 0

        x = np.array([[
            features.get("soil_health_score", 0.6),
            features.get("climate_score",     0.6),
            features.get("ph_availability",   0.8),
            features.get("aridity_index",     0.3),
            features.get("vpd_proxy",         6.0),
            features.get("gdd_proxy",         12.0),
            features.get("water_score",       0.8),
            features.get("temp_factor",       0.8),
            features.get("npk_total",         180.0),
            features.get("n_p_ratio",         2.0),
            features.get("soil_health_score", 0.6),   # n_ratio approx
            features.get("liebig_minimum",    0.6),
            features.get("om_score",          0.5),
            float(crop_idx),
        ]])

        base_pred = self._gb.predict(x)[0]
        meta_pred = self._meta.predict([[base_pred]])[0]
        prediction = max(0.01, float(meta_pred))

        # Confidence interval: ±8% + inverse liebig penalty
        liebig = features.get("liebig_minimum", 0.6)
        ci_pct = 0.08 + (1 - liebig) * 0.04
        lower  = max(0.0, prediction * (1 - ci_pct))
        upper  = prediction * (1 + ci_pct)

        # Limiting factor
        nutrient_scores = {
            "Nitrogen (N)":    features.get("soil_health_score", 1),
            "Phosphorus (P)":  features.get("ph_availability", 1),
            "Water supply":    features.get("water_score", 1),
            "Temperature":     features.get("temp_factor", 1),
            "Soil organic matter": features.get("om_score", 1),
        }
        limiting = min(nutrient_scores, key=nutrient_scores.get)
        limiting_score = nutrient_scores[limiting]

        # Improvement potential
        base_yield = CROP_META.get(crop, {}).get("base", prediction)
        improvement = max(0.0, base_yield - prediction)

        return {
            "predicted_yield":    round(prediction, 2),
            "yield_range":        (round(lower, 2), round(upper, 2)),
            "unit":               "t/ha",
            "confidence":         round(1.0 - ci_pct, 3),
            "limiting_factor":    limiting,
            "limiting_score":     round(limiting_score, 3),
            "improvement_potential": round(improvement, 2),
            "max_potential":      round(base_yield, 2),
            "grade":              self._grade(prediction, base_yield),
        }

    def _grade(self, y: float, base: float) -> str:
        ratio = y / base
        if ratio >= 0.90: return "Excellent"
        if ratio >= 0.75: return "Good"
        if ratio >= 0.55: return "Fair"
        return "Poor — intervention needed"

    def _compute_importance(self) -> dict:
        """Permutation importance on training data."""
        from sklearn.metrics import r2_score
        X_s, y_s = self._X, self._y
        base_r2  = r2_score(y_s, self._gb.predict(X_s))
        imps = {}
        for i, name in enumerate(FEATURE_NAMES):
            X_perm = X_s.copy()
            np.random.default_rng(i).shuffle(X_perm[:, i])
            perm_r2 = r2_score(y_s, self._gb.predict(X_perm))
            imps[name] = round(max(0, base_r2 - perm_r2), 5)
        total = sum(imps.values()) + 1e-10
        return {k: round(v/total, 4) for k, v in sorted(imps.items(), key=lambda x: -x[1])}

    @property
    def feature_importance(self) -> dict:
        return self._importance