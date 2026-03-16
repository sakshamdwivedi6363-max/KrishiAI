"""
soil_weather.py
───────────────
Soil nutrient and weather condition analysis.

Takes raw farmer inputs (N, P, K, pH, OM, temperature, rainfall,
humidity, sunlight) and produces:
  - Soil health score (0–100)
  - 13 engineered features for the yield model
  - Deficiency diagnoses with remedies
  - Water stress assessment
  - Growing conditions report
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SoilData:
    N:  float = 80.0    # Nitrogen  kg/ha
    P:  float = 40.0    # Phosphorus kg/ha
    K:  float = 60.0    # Potassium  kg/ha
    pH: float = 6.5     # Soil pH
    OM: float = 2.5     # Organic matter %
    texture: str = "loamy"  # sandy | loamy | clay | silty

@dataclass
class WeatherData:
    temperature: float = 25.0   # °C
    rainfall:    float = 30.0   # mm/week
    humidity:    float = 65.0   # %
    sunlight:    float = 8.0    # hours/day
    season:      str   = "kharif"  # kharif | rabi | zaid


# ── Crop reference parameters ────────────────────────────────────────────────
CROP_PARAMS = {
    "wheat":     {"N_opt": 120, "P_opt": 45, "K_opt": 80, "T_opt": 18, "R_opt": 40, "base_yield": 4.5,  "pH_min": 6.0, "pH_max": 7.5},
    "rice":      {"N_opt": 140, "P_opt": 50, "K_opt": 90, "T_opt": 26, "R_opt": 80, "base_yield": 5.0,  "pH_min": 5.5, "pH_max": 7.0},
    "maize":     {"N_opt": 160, "P_opt": 60, "K_opt": 100,"T_opt": 24, "R_opt": 60, "base_yield": 6.0,  "pH_min": 5.8, "pH_max": 7.0},
    "tomato":    {"N_opt": 130, "P_opt": 55, "K_opt": 120,"T_opt": 22, "R_opt": 55, "base_yield": 40.0, "pH_min": 6.0, "pH_max": 7.0},
    "potato":    {"N_opt": 90,  "P_opt": 60, "K_opt": 150,"T_opt": 17, "R_opt": 50, "base_yield": 25.0, "pH_min": 5.5, "pH_max": 6.5},
    "cotton":    {"N_opt": 120, "P_opt": 50, "K_opt": 80, "T_opt": 26, "R_opt": 50, "base_yield": 2.5,  "pH_min": 6.0, "pH_max": 7.5},
    "soybean":   {"N_opt": 60,  "P_opt": 40, "K_opt": 60, "T_opt": 22, "R_opt": 55, "base_yield": 3.0,  "pH_min": 6.0, "pH_max": 7.0},
    "sugarcane": {"N_opt": 180, "P_opt": 60, "K_opt": 160,"T_opt": 28, "R_opt": 100,"base_yield": 70.0, "pH_min": 6.0, "pH_max": 7.5},
}


class SoilWeatherAnalyser:
    """
    Analyses soil + weather inputs and produces a structured report
    with engineered features for downstream yield prediction.
    """

    def analyse(self, soil: SoilData, weather: WeatherData, crop: str = "wheat") -> dict:
        crop = crop.lower()
        params = CROP_PARAMS.get(crop, CROP_PARAMS["wheat"])

        # ── 1. Nutrient analysis ───────────────────────────────────────────
        nutrients = self._analyse_nutrients(soil, params)

        # ── 2. pH analysis ─────────────────────────────────────────────────
        ph_analysis = self._analyse_ph(soil.pH, params)

        # ── 3. Weather analysis ────────────────────────────────────────────
        weather_analysis = self._analyse_weather(weather, params)

        # ── 4. Soil health score ──────────────────────────────────────────
        soil_health = self._compute_soil_health(soil, params, nutrients)

        # ── 5. Engineered features for yield model ─────────────────────────
        features = self._engineer_features(soil, weather, params)

        # ── 6. Water management ────────────────────────────────────────────
        water = self._water_management(weather, params)

        return {
            "soil_health_score": soil_health,
            "nutrients":         nutrients,
            "ph_analysis":       ph_analysis,
            "weather_analysis":  weather_analysis,
            "water_management":  water,
            "engineered_features": features,
            "crop":              crop,
            "params":            params,
        }

    def _analyse_nutrients(self, soil: SoilData, params: dict) -> dict:
        def status(val, opt):
            ratio = val / opt
            if ratio < 0.40: return "critical_low"
            if ratio < 0.65: return "low"
            if ratio < 0.85: return "slightly_low"
            if ratio <= 1.20: return "optimal"
            return "excess"

        def remedy(nutrient, stat):
            if "low" in stat or "critical" in stat:
                remedies = {
                    "N": "Apply urea (46% N) — dose: {:.0f} kg/ha",
                    "P": "Apply DAP (46% P₂O₅) — dose: {:.0f} kg/ha",
                    "K": "Apply MOP (60% K₂O) — dose: {:.0f} kg/ha",
                }
                return remedies.get(nutrient, "Consult soil laboratory")
            return "No action needed"

        n_stat = status(soil.N, params["N_opt"])
        p_stat = status(soil.P, params["P_opt"])
        k_stat = status(soil.K, params["K_opt"])

        return {
            "N": {"value": soil.N, "optimal": params["N_opt"], "status": n_stat,
                  "deficit": max(0, params["N_opt"] - soil.N),
                  "remedy": remedy("N", n_stat).format(max(0, params["N_opt"] - soil.N) * 2.17)},
            "P": {"value": soil.P, "optimal": params["P_opt"], "status": p_stat,
                  "deficit": max(0, params["P_opt"] - soil.P),
                  "remedy": remedy("P", p_stat).format(max(0, params["P_opt"] - soil.P) * 2.17)},
            "K": {"value": soil.K, "optimal": params["K_opt"], "status": k_stat,
                  "deficit": max(0, params["K_opt"] - soil.K),
                  "remedy": remedy("K", k_stat).format(max(0, params["K_opt"] - soil.K) * 1.67)},
            "limiting_nutrient": min(
                [("N", soil.N/params["N_opt"]), ("P", soil.P/params["P_opt"]), ("K", soil.K/params["K_opt"])],
                key=lambda x: x[1]
            )[0],
        }

    def _analyse_ph(self, ph: float, params: dict) -> dict:
        ph_min, ph_max = params["pH_min"], params["pH_max"]
        deviation = 0.0
        if ph < ph_min:
            deviation = ph_min - ph
            status = "too_acidic"
            action = f"Apply agricultural lime {deviation * 1.5:.1f} t/ha — raise pH by {deviation:.1f} units"
        elif ph > ph_max:
            deviation = ph - ph_max
            status = "too_alkaline"
            action = f"Apply elemental sulphur {deviation * 0.8:.1f} t/ha — lower pH by {deviation:.1f} units"
        else:
            status = "optimal"
            action = "No pH correction needed"

        # Nutrient availability factor (0–1) — peaks at pH 6.5
        availability = max(0.1, 1.0 - abs(ph - 6.5) * 0.22)
        return {
            "value": ph, "status": status, "deviation": round(deviation, 2),
            "action": action, "nutrient_availability_factor": round(availability, 3),
        }

    def _analyse_weather(self, weather: WeatherData, params: dict) -> dict:
        # Temperature stress (quadratic penalty away from optimum)
        t_factor = max(0.05, 1.0 - ((weather.temperature - params["T_opt"]) ** 2) / 150)
        if weather.temperature < params["T_opt"] - 8:
            t_status = "cold_stress"
        elif weather.temperature > params["T_opt"] + 8:
            t_status = "heat_stress"
        else:
            t_status = "optimal"

        # Humidity assessment
        if weather.humidity > 85:
            h_status = "high_disease_risk"
        elif weather.humidity < 35:
            h_status = "desiccation_risk"
        else:
            h_status = "suitable"

        # Sunlight
        if weather.sunlight < 5:
            s_status = "insufficient"
        elif weather.sunlight > 12:
            s_status = "excessive"
        else:
            s_status = "adequate"

        # Growing Degree Days proxy
        gdd = max(0.0, weather.temperature - 10) * 7  # per week

        return {
            "temperature":    {"value": weather.temperature, "optimal": params["T_opt"],
                               "status": t_status, "stress_factor": round(t_factor, 3)},
            "humidity":       {"value": weather.humidity, "status": h_status},
            "sunlight":       {"value": weather.sunlight, "status": s_status},
            "season":         weather.season,
            "GDD_week":       round(gdd, 1),
        }

    def _compute_soil_health(self, soil: SoilData, params: dict, nutrients: dict) -> float:
        n_score = min(100, soil.N / params["N_opt"] * 100) * 0.25
        p_score = min(100, soil.P / params["P_opt"] * 100) * 0.20
        k_score = min(100, soil.K / params["K_opt"] * 100) * 0.20
        ph_score = max(0, 100 - abs(soil.pH - 6.5) * 25) * 0.20
        om_score = min(100, soil.OM / 4.0 * 100) * 0.15
        return round(n_score + p_score + k_score + ph_score + om_score, 1)

    def _water_management(self, weather: WeatherData, params: dict) -> dict:
        deficit = max(0.0, params["R_opt"] - weather.rainfall)
        excess  = max(0.0, weather.rainfall - params["R_opt"] * 1.4)
        if deficit > 25:
            status = "critical_deficit"
            freq   = "every 2 days"
            vol    = f"{deficit * 0.6:.0f} mm/session"
        elif deficit > 12:
            status = "moderate_deficit"
            freq   = "every 3–4 days"
            vol    = f"{deficit * 0.5:.0f} mm/session"
        elif excess > 0:
            status = "excess"
            freq   = "No irrigation — consider drainage"
            vol    = "—"
        else:
            status = "adequate"
            freq   = "every 6–7 days"
            vol    = "15 mm/session"

        et = max(0, weather.temperature * 0.12 + weather.sunlight * 0.08 - weather.humidity * 0.02)
        return {
            "status": status,
            "weekly_deficit_mm": round(deficit, 1),
            "irrigation_frequency": freq,
            "irrigation_volume": vol,
            "evapotranspiration_proxy": round(et, 2),
            "note": "Use drip irrigation to reduce evaporation by 35–40%" if deficit > 15 else "Current moisture near adequate",
        }

    def _engineer_features(self, soil: SoilData, weather: WeatherData, params: dict) -> dict:
        """Derive composite agronomic features for the yield model."""
        ph_dev = abs(soil.pH - 6.5)
        ph_avail = max(0.1, 1.0 - ph_dev * 0.22)

        soil_health = (
            min(1, soil.N / params["N_opt"]) * 0.30 +
            min(1, soil.P / params["P_opt"]) * 0.20 +
            min(1, soil.K / params["K_opt"]) * 0.20 +
            max(0, 1 - ph_dev / 3) * 0.20 +
            min(1, soil.OM / 4.0) * 0.10
        )

        aridity = weather.rainfall / (weather.temperature * 20 + 1)
        vpd = weather.temperature * (1 - weather.humidity / 100)
        gdd = max(0, weather.temperature - 10)
        water_score = min(1.2, weather.rainfall / params["R_opt"])
        t_factor = max(0.05, 1.0 - ((weather.temperature - params["T_opt"]) ** 2) / 150)
        climate_score = t_factor * 0.45 + min(1, water_score) * 0.35 + min(1, weather.sunlight / 10) * 0.20
        npk_total = soil.N + soil.P + soil.K
        n_p_ratio = soil.N / (soil.P + 1e-6)
        n_k_ratio = soil.N / (soil.K + 1e-6)
        liebig_min = min(soil.N / params["N_opt"], soil.P / params["P_opt"], soil.K / params["K_opt"])

        return {
            "soil_health_score": round(soil_health, 4),
            "climate_score":     round(climate_score, 4),
            "ph_availability":   round(ph_avail, 4),
            "aridity_index":     round(aridity, 4),
            "vpd_proxy":         round(vpd, 4),
            "gdd_proxy":         round(gdd, 4),
            "water_score":       round(water_score, 4),
            "temp_factor":       round(t_factor, 4),
            "npk_total":         round(npk_total, 2),
            "n_p_ratio":         round(n_p_ratio, 3),
            "n_k_ratio":         round(n_k_ratio, 3),
            "liebig_minimum":    round(liebig_min, 4),
            "om_score":          round(min(1, soil.OM / 4.0), 4),
        }