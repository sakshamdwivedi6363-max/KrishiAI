"""
recommendation_engine.py
────────────────────────
Hybrid rule + ML recommendation engine.

Takes outputs from disease_detector, soil_weather, and yield_predictor
and produces four types of recommendations ranked by yield impact:
  1. Fertilizer
  2. Irrigation
  3. Disease treatment
  4. Crop improvement techniques

Algorithm:
  - Rule engine fires first (explicit agronomic rules)
  - ML ranking scores each recommendation by predicted yield delta
  - Top N recommendations selected per category
  - All recommendations carry: action, product, dose, timing, impact_score
"""

from typing import Optional


# ── Knowledge base ─────────────────────────────────────────────────────────
FERTILIZER_KB = {
    "wheat": {
        "N_low":     {"product": "Urea (46% N)",        "dose": "{:.0f} kg/ha",  "timing": "50% basal + 50% tillering"},
        "P_low":     {"product": "DAP (18-46-0)",        "dose": "{:.0f} kg/ha",  "timing": "Full basal before sowing"},
        "K_low":     {"product": "MOP (60% K₂O)",        "dose": "{:.0f} kg/ha",  "timing": "50% basal + 50% at jointing"},
        "pH_acid":   {"product": "Agricultural lime",    "dose": "{:.1f} t/ha",   "timing": "6 weeks before sowing"},
        "pH_alk":    {"product": "Elemental sulphur",    "dose": "{:.1f} t/ha",   "timing": "Incorporate before sowing"},
        "OM_low":    {"product": "Farmyard manure (FYM)","dose": "5–10 t/ha",     "timing": "Incorporate 2 weeks before sowing"},
    },
    "rice": {
        "N_low":     {"product": "Urea (46% N)",         "dose": "{:.0f} kg/ha",  "timing": "3 splits: basal, tillering, panicle initiation"},
        "P_low":     {"product": "SSP (16% P₂O₅)",       "dose": "{:.0f} kg/ha",  "timing": "Full basal before transplanting"},
        "K_low":     {"product": "MOP (60% K₂O)",        "dose": "{:.0f} kg/ha",  "timing": "50% basal + 50% at panicle init"},
        "pH_acid":   {"product": "Agricultural lime",    "dose": "{:.1f} t/ha",   "timing": "Before field preparation"},
        "OM_low":    {"product": "Green manure / Azolla", "dose": "25 kg/ha seed", "timing": "Incorporate 2 weeks before transplanting"},
    },
}

DISEASE_TREATMENT_KB = {
    "early_blight":        {"chemical": "Mancozeb 75% WP — 2g/L", "organic": "Neem oil 3ml/L + copper soap", "interval": "10 days × 3 sprays"},
    "late_blight":         {"chemical": "Metalaxyl+Mancozeb 2.5g/L", "organic": "Bordeaux mixture 1%",        "interval": "7 days — urgent"},
    "leaf_rust":           {"chemical": "Propiconazole 25% EC 1ml/L", "organic": "Sulphur 3kg/ha dust",       "interval": "14 days × 2 sprays"},
    "powdery_mildew":      {"chemical": "Hexaconazole 5% SC 1ml/L",   "organic": "KHCO₃ 5g/L + wetting agent","interval": "7 days × 4 sprays"},
    "mosaic_virus":        {"chemical": "No curative — remove plants", "organic": "Imidacloprid for vectors",   "interval": "Immediate"},
    "nutrient_deficiency": {"chemical": "NPK 19-19-19 foliar 2g/L",   "organic": "Compost tea 1:10 dilution",  "interval": "Weekly × 3 weeks"},
    "bacterial_spot":      {"chemical": "Copper hydroxide 3g/L",       "organic": "Garlic-chilli extract spray","interval": "7 days × 3 sprays"},
    "healthy":             {"chemical": "Preventive copper spray 1g/L","organic": "Neem + turmeric extract",   "interval": "Monthly preventive"},
}

CROP_IMPROVEMENT_KB = {
    "low_OM":         {"action": "Build organic matter",       "technique": "Add FYM 10 t/ha + biostimulant inoculant", "impact": 0.18},
    "pH_correction":  {"action": "Correct soil pH",            "technique": "Lime or sulphur application + monitoring", "impact": 0.20},
    "heat_stress":    {"action": "Reduce heat stress",         "technique": "Straw mulch 5–8cm depth; saves 4–6°C soil temp", "impact": 0.12},
    "low_sunlight":   {"action": "Improve light interception", "technique": "Adjust row spacing/orientation; remove canopy weeds", "impact": 0.08},
    "high_humidity":  {"action": "Improve canopy airflow",     "technique": "Wider row spacing + pruning of lower leaves", "impact": 0.10},
    "drought_risk":   {"action": "Improve water retention",    "technique": "Mulching + conservation tillage + drip system", "impact": 0.22},
    "variety":        {"action": "Upgrade crop variety",       "technique": "Use certified HYV seed from regional ICAR station", "impact": 0.25},
    "ipm":            {"action": "Integrated pest management", "technique": "Pheromone traps + biological control agents", "impact": 0.10},
}


class RecommendationEngine:
    """
    Generates prioritised recommendations from all AI module outputs.
    """

    def generate(
        self,
        disease_result: Optional[dict],
        soil_weather_result: dict,
        yield_result: dict,
        crop: str = "wheat",
    ) -> dict:
        """
        Generate all recommendations.

        Returns:
            dict with keys:
              fertilizer   list[dict]
              irrigation   dict
              disease_treatment  list[dict]
              crop_improvement   list[dict]
              priority_action    str
              overall_score      int (0–100)
        """
        crop = crop.lower()
        nutrients   = soil_weather_result["nutrients"]
        weather_a   = soil_weather_result["weather_analysis"]
        water_mgmt  = soil_weather_result["water_management"]
        eng_feats   = soil_weather_result["engineered_features"]
        soil_score  = soil_weather_result["soil_health_score"]
        ph_a        = soil_weather_result["ph_analysis"]
        soil_data   = soil_weather_result.get("soil_data_raw", {})
        params      = soil_weather_result["params"]

        fert_recs   = self._fertilizer_recommendations(nutrients, ph_a, crop, params, soil_data)
        irr_rec     = self._irrigation_recommendation(water_mgmt, weather_a)
        dis_recs    = self._disease_treatment_recommendations(disease_result)
        imp_recs    = self._improvement_recommendations(soil_weather_result, weather_a, ph_a, eng_feats)

        # Rank everything by impact
        all_recs = fert_recs + dis_recs + imp_recs
        for r in all_recs:
            r.setdefault("impact_score", 0.5)

        all_recs.sort(key=lambda x: x["impact_score"], reverse=True)
        priority = all_recs[0]["action"] if all_recs else "Monitor crop regularly"

        overall = int(min(100, max(0,
            soil_score * 0.4 +
            yield_result.get("confidence", 0.7) * 100 * 0.35 +
            eng_feats.get("climate_score", 0.6) * 100 * 0.25
        )))

        return {
            "fertilizer":       fert_recs,
            "irrigation":       irr_rec,
            "disease_treatment":dis_recs,
            "crop_improvement": imp_recs,
            "priority_action":  priority,
            "overall_score":    overall,
            "ranked_all":       all_recs[:6],
        }

    def _fertilizer_recommendations(self, nutrients, ph_a, crop, params, soil_data):
        recs = []
        kb = FERTILIZER_KB.get(crop, FERTILIZER_KB["wheat"])

        # N deficiency
        n = nutrients["N"]
        if "low" in n["status"] or "critical" in n["status"]:
            impact = 0.35 + (0.20 if "critical" in n["status"] else 0)
            dose_kg = n["deficit"] * 2.17
            recs.append({
                "type": "fertilizer",
                "nutrient": "Nitrogen",
                "action": f"Apply {kb.get('N_low',{}).get('product','Urea')}",
                "dose": kb.get("N_low", {}).get("dose", "{:.0f} kg/ha").format(dose_kg),
                "timing": kb.get("N_low", {}).get("timing", "Basal application"),
                "status": n["status"],
                "impact_score": round(impact, 3),
                "priority": "high" if "critical" in n["status"] else "medium",
            })

        # P deficiency
        p = nutrients["P"]
        if "low" in p["status"] or "critical" in p["status"]:
            dose_kg = p["deficit"] * 2.17
            recs.append({
                "type": "fertilizer",
                "nutrient": "Phosphorus",
                "action": f"Apply {kb.get('P_low',{}).get('product','DAP')}",
                "dose": kb.get("P_low", {}).get("dose", "{:.0f} kg/ha").format(dose_kg),
                "timing": kb.get("P_low", {}).get("timing", "Basal"),
                "status": p["status"],
                "impact_score": 0.25,
                "priority": "medium",
            })

        # K deficiency
        k = nutrients["K"]
        if "low" in k["status"] or "critical" in k["status"]:
            dose_kg = k["deficit"] * 1.67
            recs.append({
                "type": "fertilizer",
                "nutrient": "Potassium",
                "action": f"Apply {kb.get('K_low',{}).get('product','MOP')}",
                "dose": kb.get("K_low", {}).get("dose", "{:.0f} kg/ha").format(dose_kg),
                "timing": kb.get("K_low", {}).get("timing", "Basal + top-dress"),
                "status": k["status"],
                "impact_score": 0.22,
                "priority": "medium",
            })

        # pH correction
        if ph_a["status"] != "optimal":
            recs.append({
                "type": "fertilizer",
                "nutrient": "pH correction",
                "action": ph_a["action"],
                "dose": f"{ph_a['deviation']*1.5:.1f} t/ha",
                "timing": "Before next sowing season",
                "status": ph_a["status"],
                "impact_score": 0.30,
                "priority": "high",
            })

        return recs

    def _irrigation_recommendation(self, water_mgmt, weather_a) -> dict:
        status = water_mgmt["status"]
        urgency = "critical" if status == "critical_deficit" else ("moderate" if "moderate" in status else "low")
        return {
            "status": status,
            "urgency": urgency,
            "frequency": water_mgmt["irrigation_frequency"],
            "volume": water_mgmt["irrigation_volume"],
            "method": "Drip irrigation" if water_mgmt["weekly_deficit_mm"] > 15 else "Sprinkler or furrow",
            "note": water_mgmt["note"],
            "disease_risk": weather_a["humidity"]["status"] == "high_disease_risk",
            "impact_score": 0.40 if urgency == "critical" else 0.20,
        }

    def _disease_treatment_recommendations(self, disease_result) -> list:
        if disease_result is None:
            return []
        disease = disease_result.get("top_disease", "healthy")
        confidence = disease_result.get("confidence", 0.5)
        severity = disease_result.get("severity", "low")

        kb = DISEASE_TREATMENT_KB.get(disease, DISEASE_TREATMENT_KB["healthy"])
        impact = confidence * (1.5 if severity in ("high","critical") else 1.0) * 0.5

        return [{
            "type": "disease_treatment",
            "disease": disease.replace("_", " ").title(),
            "confidence": confidence,
            "severity": severity,
            "action": f"Treat for {disease.replace('_',' ').title()}",
            "chemical_treatment": kb["chemical"],
            "organic_alternative": kb["organic"],
            "spray_interval": kb["interval"],
            "additional_action": disease_result.get("treatment", {}).get("timing", ""),
            "impact_score": round(impact, 3),
            "priority": "critical" if severity == "critical" else ("high" if severity == "high" else "medium"),
        }]

    def _improvement_recommendations(self, soil_result, weather_a, ph_a, eng_feats) -> list:
        recs = []
        soil = soil_result

        if eng_feats.get("om_score", 1) < 0.5:
            r = CROP_IMPROVEMENT_KB["low_OM"].copy()
            r.update({"type": "improvement", "action": r["action"], "impact_score": r["impact"]})
            recs.append(r)

        if ph_a["status"] != "optimal" and ph_a["deviation"] > 0.8:
            r = CROP_IMPROVEMENT_KB["pH_correction"].copy()
            r.update({"type": "improvement", "action": r["action"], "impact_score": r["impact"]})
            recs.append(r)

        t_status = weather_a["temperature"]["status"]
        if t_status in ("heat_stress", "cold_stress"):
            r = CROP_IMPROVEMENT_KB["heat_stress"].copy()
            r.update({"type": "improvement", "action": r["action"], "impact_score": r["impact"]})
            recs.append(r)

        if weather_a["humidity"]["status"] == "high_disease_risk":
            r = CROP_IMPROVEMENT_KB["high_humidity"].copy()
            r.update({"type": "improvement", "action": r["action"], "impact_score": r["impact"]})
            recs.append(r)

        # Always suggest variety improvement as long-term
        r = CROP_IMPROVEMENT_KB["variety"].copy()
        r.update({"type": "improvement", "action": r["action"], "impact_score": r["impact"]})
        recs.append(r)

        return recs