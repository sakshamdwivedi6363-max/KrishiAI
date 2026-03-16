"""
smart_agri_system.py
─────────────────────
Smart Agriculture AI — Full Prototype System
============================================

Integrates all five modules into a single pipeline:

  Module 1 → DiseaseDetector    (crop image analysis)
  Module 2 → SoilWeatherAnalyser (soil + weather features)
  Module 3 → YieldPredictor      (ensemble regression)
  Module 4 → RecommendationEngine (rule + ML advisory)
  Module 5 → MultilingualOutput  (i18n localised report)

Architecture:
  ┌─────────────────────────────────────────────────────┐
  │  Farmer Input                                       │
  │   • Crop image (PIL.Image or file path)             │
  │   • Soil: N, P, K, pH, OM                          │
  │   • Weather: temp, rainfall, humidity, sunlight     │
  │   • Crop type & language preference                 │
  └───────────────────┬─────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
  DiseaseDetector  SoilWeather  [same soil+weather]
  (CNN features)   Analyser       ─────────────────
        │             │                │
        │             ▼                │
        │       Feature Engineering    │
        │       (13 derived features)  │
        │             │                │
        └─────────────┼────────────────┘
                      ▼
               YieldPredictor
               (GBM stacked ensemble)
                      │
                      ▼
           RecommendationEngine
           (Rule + ML ranked advisory)
                      │
                      ▼
           MultilingualOutput
           (i18n localised report)
                      │
                      ▼
           ┌──────────────────────┐
           │  Printed report      │
           │  JSON full result    │
           │  Visualisation PNG   │
           └──────────────────────┘

Usage:
    python smart_agri_system.py
    → runs full demo in 7 languages + saves charts
"""

import os
import sys
import json
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Add package root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

from core.disease_detector    import DiseaseDetector
from core.soil_weather        import SoilWeatherAnalyser, SoilData, WeatherData
from core.yield_predictor     import YieldPredictor, CROP_META
from core.recommendation_engine import RecommendationEngine
from core.multilingual        import MultilingualOutput, SUPPORTED_LANGUAGES

# ── Results directory ─────────────────────────────────────────────────────
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
# PIPELINE ORCHESTRATOR
# ════════════════════════════════════════════════════════════════════════════

class SmartAgriPipeline:
    """
    Full Smart Agriculture AI pipeline.

    Usage:
        pipeline = SmartAgriPipeline()
        result = pipeline.run(
            image=Image.open("leaf.jpg"),
            soil=SoilData(N=80, P=40, K=60, pH=6.2),
            weather=WeatherData(temperature=28, rainfall=25),
            crop="wheat",
            lang="hi",
        )
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        print("\n" + "═"*60)
        print("  Initialising Smart Agriculture AI System...")
        print("═"*60)
        t0 = time.perf_counter()

        self.detector    = DiseaseDetector()
        self.soil_wx     = SoilWeatherAnalyser()
        self.yield_pred  = YieldPredictor()
        self.rec_engine  = RecommendationEngine()

        elapsed = time.perf_counter() - t0
        print(f"  System ready in {elapsed:.2f}s\n")

    def run(
        self,
        image:   Optional[Image.Image],
        soil:    SoilData,
        weather: WeatherData,
        crop:    str = "wheat",
        lang:    str = "en",
    ) -> dict:
        """
        Run the full pipeline.

        Returns a dict with keys:
          disease_result, soil_weather_result, yield_result,
          recommendations, localised_report, crop, lang
        """
        crop = crop.lower()
        t0   = time.perf_counter()

        # ── Step 1: Disease detection ──────────────────────────────
        disease_result = None
        if image is not None:
            if self.verbose:
                print(f"  [1/4] Analysing crop image ({image.width}×{image.height}px)...")
            disease_result = self.detector.predict(image, crop_type=crop)
            if self.verbose:
                d = disease_result["top_disease"]
                c = disease_result["confidence"]
                print(f"        → {d.replace('_',' ').title()} ({c:.1%} confidence)")
        else:
            if self.verbose:
                print("  [1/4] No image provided — skipping disease detection")

        # ── Step 2: Soil & weather analysis ───────────────────────
        if self.verbose:
            print("  [2/4] Analysing soil nutrients and weather conditions...")
        soil_weather_result = self.soil_wx.analyse(soil, weather, crop)
        soil_weather_result["soil_data_raw"] = {
            "N": soil.N, "P": soil.P, "K": soil.K,
            "pH": soil.pH, "OM": soil.OM,
        }
        if self.verbose:
            sh = soil_weather_result["soil_health_score"]
            print(f"        → Soil health score: {sh}/100")

        # ── Step 3: Yield prediction ───────────────────────────────
        if self.verbose:
            print("  [3/4] Predicting crop yield...")
        features     = soil_weather_result["engineered_features"]
        yield_result = self.yield_pred.predict(features, crop)
        if self.verbose:
            y  = yield_result["predicted_yield"]
            lo = yield_result["yield_range"][0]
            hi = yield_result["yield_range"][1]
            print(f"        → {y} t/ha  (range: {lo}–{hi} t/ha)")

        # ── Step 4: Recommendations ───────────────────────────────
        if self.verbose:
            print("  [4/4] Generating recommendations...")
        recommendations = self.rec_engine.generate(
            disease_result, soil_weather_result, yield_result, crop
        )
        if self.verbose:
            n = len(recommendations.get("ranked_all", []))
            print(f"        → {n} recommendations generated")

        # ── Step 5: Multilingual output ───────────────────────────
        full_result = {
            "disease_result":      disease_result,
            "soil_weather_result": soil_weather_result,
            "yield_result":        yield_result,
            "recommendations":     recommendations,
            "crop":                crop,
            "lang":                lang,
        }
        ml_out = MultilingualOutput(lang=lang)
        localised = ml_out.localise_report(full_result)
        full_result["localised_report"] = localised
        full_result["_ml_out"] = ml_out

        elapsed = time.perf_counter() - t0
        if self.verbose:
            print(f"\n  Pipeline completed in {elapsed:.2f}s")

        return full_result


# ════════════════════════════════════════════════════════════════════════════
# SYNTHETIC IMAGE GENERATOR (for demo without real photos)
# ════════════════════════════════════════════════════════════════════════════

def _create_demo_image(disease_type: str = "early_blight") -> Image.Image:
    """
    Create a realistic synthetic leaf image for demo purposes.
    Uses PIL to draw a stylised leaf with appropriate colour patterns
    for the requested disease type.
    """
    img = Image.new("RGB", (400, 400), (45, 95, 30))  # dark green background
    draw = ImageDraw.Draw(img)

    # Base leaf shape
    for r in range(180, 0, -2):
        green = min(255, 60 + r // 2)
        draw.ellipse([(200 - r, 200 - r), (200 + r, 200 + r)],
                     fill=(20, green, 15))

    # Disease-specific visual patterns
    if disease_type == "early_blight":
        # Brown concentric ring spots
        for cx, cy in [(150,160),(230,220),(170,260),(290,190)]:
            for rr, col in [(28,(100,55,20)),(18,(130,75,30)),(10,(160,100,40))]:
                draw.ellipse([(cx-rr,cy-rr),(cx+rr,cy+rr)], fill=col)
    elif disease_type == "late_blight":
        # Dark irregular patches
        for _ in range(5):
            import random; random.seed(42)
            cx, cy = random.randint(80,320), random.randint(80,320)
            draw.ellipse([(cx-30,cy-22),(cx+30,cy+22)], fill=(40,35,60))
    elif disease_type == "leaf_rust":
        # Orange-brown pustules
        for cx, cy in [(140,140),(200,170),(260,200),(180,240),(310,160)]:
            draw.ellipse([(cx-8,cy-8),(cx+8,cy+8)], fill=(190,105,20))
    elif disease_type == "powdery_mildew":
        # White powdery patches
        for cx, cy in [(160,150),(240,180),(180,250),(280,220)]:
            draw.ellipse([(cx-18,cy-18),(cx+18,cy+18)], fill=(215,212,205))
    elif disease_type == "nutrient_deficiency":
        # Yellowing — replace green with yellow-green
        for y in range(0, 400, 4):
            for x in range(0, 400, 4):
                px = img.getpixel((x, y))
                if px[1] > 60:
                    img.putpixel((x, y), (px[0]+40, px[1], px[2]-15))
    else:
        # Healthy — add vein texture
        for i in range(10, 180, 14):
            draw.line([(200, 200), (200 - i*1.3, 200 + i*0.8)], fill=(30, 85, 25), width=1)
            draw.line([(200, 200), (200 + i*1.2, 200 + i*0.9)], fill=(30, 85, 25), width=1)

    # Slight blur for realism
    img = img.filter(ImageFilter.GaussianBlur(0.7))
    return img


# ════════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ════════════════════════════════════════════════════════════════════════════

def create_full_dashboard(full_result: dict, save_path: str) -> str:
    """
    Generate a comprehensive visual dashboard PNG with 6 panels:
      (A) Crop image + disease heatmap
      (B) Soil nutrient bar chart
      (C) Yield prediction with confidence
      (D) Feature importance
      (E) Recommendation priority chart
      (F) Weather conditions radar
    """
    if not HAS_PLOT:
        print("  [Viz] matplotlib not available — skipping dashboard")
        return None

    DARK   = "#1a1508"
    EARTH  = "#2d2210"
    WHEAT  = "#e8c97a"
    STRAW  = "#c4a35a"
    LEAF   = "#4a9a3f"
    SPROUT = "#7dce6e"
    SKY    = "#4a8db5"
    CORAL  = "#e85c2a"
    MIST   = "#a8cfe0"

    fig = plt.figure(figsize=(18, 12), facecolor=DARK)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32,
                            left=0.06, right=0.97, top=0.92, bottom=0.06)

    yr   = full_result.get("yield_result", {})
    sw   = full_result.get("soil_weather_result", {})
    dr   = full_result.get("disease_result") or {}
    rr   = full_result.get("recommendations", {})
    crop = full_result.get("crop", "wheat")

    # ── Title ──────────────────────────────────────────────────────────────
    fig.suptitle(
        f"KhetAI — Smart Agriculture Report  |  {crop.title()}  |  "
        f"Yield: {yr.get('predicted_yield','—')} t/ha  |  Score: {rr.get('overall_score','—')}/100",
        fontsize=13, color=WHEAT, fontweight="bold", y=0.97,
    )

    def style_ax(ax, title):
        ax.set_facecolor(EARTH)
        ax.tick_params(colors=STRAW, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#3d2f16")
        ax.set_title(title, color=WHEAT, fontsize=10, pad=8, fontweight="normal")
        ax.title.set_fontfamily("serif")

    # ── Panel A: Disease detection ─────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    style_ax(ax_a, "Disease Detection")
    disease = dr.get("top_disease", "healthy")
    top3    = dr.get("top3", [("healthy", 1.0)])
    names   = [t[0].replace("_"," ").title()[:20] for t in top3]
    probs   = [t[1] for t in top3]
    colours = [LEAF if "healthy" in t[0] else CORAL for t in top3]
    bars    = ax_a.barh(names, probs, color=colours, height=0.5)
    ax_a.set_xlim(0, 1.05)
    ax_a.set_xlabel("Confidence", color=STRAW, fontsize=8)
    ax_a.tick_params(axis="y", labelsize=8, colors=WHEAT)
    ax_a.tick_params(axis="x", labelsize=7, colors=STRAW)
    for bar, prob in zip(bars, probs):
        ax_a.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                  f"{prob:.1%}", va="center", ha="left", color=STRAW, fontsize=8)
    ax_a.spines[["top","right"]].set_visible(False)

    # ── Panel B: Soil nutrients ────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    style_ax(ax_b, "Soil Nutrients vs Optimal")
    nutrients = sw.get("nutrients", {})
    params    = sw.get("params", {})
    n_items = [(k, v.get("value",0), params.get(f"{k}_opt", 100))
               for k, v in nutrients.items() if k != "limiting_nutrient"]
    if n_items:
        labels    = [x[0] for x in n_items]
        actual    = [x[1] for x in n_items]
        optimal   = [x[2] for x in n_items]
        x_pos     = np.arange(len(labels))
        width     = 0.35
        ax_b.bar(x_pos - width/2, actual,  width, label="Actual",  color=LEAF,  alpha=0.85, zorder=3)
        ax_b.bar(x_pos + width/2, optimal, width, label="Optimal", color=STRAW, alpha=0.5, zorder=3)
        ax_b.set_xticks(x_pos)
        ax_b.set_xticklabels(labels, color=WHEAT, fontsize=9)
        ax_b.set_ylabel("kg/ha", color=STRAW, fontsize=8)
        ax_b.legend(fontsize=7, facecolor=EARTH, edgecolor="#3d2f16", labelcolor=STRAW)
        ax_b.yaxis.grid(True, alpha=0.2, color=STRAW)
        ax_b.set_axisbelow(True)
    sh = sw.get("soil_health_score", 0)
    ax_b.set_title(f"Soil Nutrients vs Optimal  (Health: {sh}/100)", color=WHEAT, fontsize=10, pad=8)
    ax_b.spines[["top","right"]].set_visible(False)

    # ── Panel C: Yield prediction ──────────────────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    style_ax(ax_c, "Yield Forecast")
    crops_list = list(CROP_META.keys())
    pred       = yr.get("predicted_yield", 0)
    lo, hi     = yr.get("yield_range", (pred*0.92, pred*1.08))
    base       = yr.get("max_potential", pred * 1.3)
    categories = ["Predicted", "Lower bound", "Upper bound", "Max potential"]
    values     = [pred, lo, hi, base]
    bar_colors = [LEAF, SKY, SPROUT, STRAW]
    bars2 = ax_c.bar(categories, values, color=bar_colors, width=0.55)
    ax_c.set_ylabel("t/ha", color=STRAW, fontsize=8)
    ax_c.tick_params(axis="x", labelsize=7.5, colors=WHEAT)
    for bar, val in zip(bars2, values):
        ax_c.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                  f"{val:.2f}", ha="center", va="bottom", color=WHEAT, fontsize=8)
    ax_c.yaxis.grid(True, alpha=0.2, color=STRAW)
    ax_c.set_axisbelow(True)
    ax_c.spines[["top","right"]].set_visible(False)
    grade = yr.get("grade","—")
    ax_c.set_title(f"Yield Forecast — {grade}", color=WHEAT, fontsize=10, pad=8)

    # ── Panel D: Feature importance ────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 0])
    style_ax(ax_d, "Feature Importance (Yield Model)")
    imp = full_result.get("yield_predictor_importance", {})
    if not imp:
        # use a cached one from the pipeline if available
        pass
    if imp:
        top_feats = list(imp.items())[:8]
        feat_names = [x[0].replace("_"," ").title() for x in top_feats]
        feat_vals  = [x[1] for x in top_feats]
        ax_d.barh(feat_names[::-1], feat_vals[::-1], color=SKY, alpha=0.8)
        ax_d.set_xlabel("Permutation importance", color=STRAW, fontsize=8)
        ax_d.tick_params(axis="y", labelsize=7.5, colors=MIST)
        ax_d.spines[["top","right"]].set_visible(False)
    else:
        ax_d.text(0.5, 0.5, "No feature data", ha="center", va="center",
                  color=STRAW, fontsize=10, transform=ax_d.transAxes)

    # ── Panel E: Recommendation priorities ────────────────────────────────
    ax_e = fig.add_subplot(gs[1, 1])
    style_ax(ax_e, "Top Recommendations by Yield Impact")
    ranked = rr.get("ranked_all", [])
    if ranked:
        r_labels  = [r.get("action","—")[:32] for r in ranked[:6]]
        r_impacts = [r.get("impact_score", 0) * 100 for r in ranked[:6]]
        r_colors  = [CORAL if "disease" in r.get("type","") else
                     (SKY if "irrig" in r.get("type","").lower() else
                      (LEAF if "fert" in r.get("type","") else STRAW)) for r in ranked[:6]]
        ax_e.barh(r_labels[::-1], r_impacts[::-1], color=r_colors[::-1], height=0.5)
        ax_e.set_xlabel("Impact score", color=STRAW, fontsize=8)
        ax_e.tick_params(axis="y", labelsize=7, colors=WHEAT)
        ax_e.set_xlim(0, 65)
        ax_e.spines[["top","right"]].set_visible(False)

    # ── Panel F: Weather radar ─────────────────────────────────────────────
    ax_f = fig.add_subplot(gs[1, 2], polar=True)
    ax_f.set_facecolor(EARTH)
    ax_f.tick_params(colors=STRAW, labelsize=7)
    ax_f.set_title("Weather Conditions", color=WHEAT, fontsize=10, pad=14)
    for spine in ax_f.spines.values():
        spine.set_edgecolor("#3d2f16")
    wdata = sw.get("weather_analysis", {})
    temp_norm = min(1, wdata.get("temperature", {}).get("value", 25) / 45)
    rain_norm = min(1, sw.get("water_management", {}).get("weekly_deficit_mm", 0) / 50)
    hum_norm  = min(1, wdata.get("humidity", {}).get("value", 65) / 100)
    sun_norm  = min(1, wdata.get("sunlight", {}).get("value", 8) / 14)
    gh_norm   = min(1, sw.get("soil_health_score", 50) / 100)
    cats   = ["Temp", "Water\nStress", "Humidity", "Sunlight", "Soil\nHealth"]
    vals   = [temp_norm, rain_norm, hum_norm, sun_norm, gh_norm]
    N_cat  = len(cats)
    angles = [n / float(N_cat) * 2 * np.pi for n in range(N_cat)]
    angles += angles[:1]
    vals   += vals[:1]
    ax_f.plot(angles, vals, "o-", color=LEAF, linewidth=1.8)
    ax_f.fill(angles, vals, alpha=0.25, color=LEAF)
    ax_f.set_xticks(angles[:-1])
    ax_f.set_xticklabels(cats, size=7, color=WHEAT)
    ax_f.set_ylim(0, 1)
    ax_f.yaxis.set_tick_params(labelsize=6, labelcolor=STRAW)
    ax_f.grid(color="#3d2f16", alpha=0.7)

    plt.savefig(save_path, dpi=130, bbox_inches="tight", facecolor=DARK)
    plt.close()
    return save_path


# ════════════════════════════════════════════════════════════════════════════
# DEMO SCENARIOS
# ════════════════════════════════════════════════════════════════════════════

DEMO_SCENARIOS = [
    {
        "name":    "Wheat — Rust disease, nitrogen deficit",
        "crop":    "wheat",
        "disease": "leaf_rust",
        "soil":    SoilData(N=55, P=30, K=50, pH=6.1, OM=1.4),
        "weather": WeatherData(temperature=22, rainfall=18, humidity=78, sunlight=7, season="rabi"),
        "lang":    "en",
    },
    {
        "name":    "Rice — Blight risk, adequate soil",
        "crop":    "rice",
        "disease": "early_blight",
        "soil":    SoilData(N=110, P=55, K=80, pH=6.4, OM=3.0),
        "weather": WeatherData(temperature=29, rainfall=70, humidity=88, sunlight=8, season="kharif"),
        "lang":    "hi",
    },
    {
        "name":    "Tomato — Late blight, water deficit",
        "crop":    "tomato",
        "disease": "late_blight",
        "soil":    SoilData(N=90, P=40, K=100, pH=6.8, OM=2.2),
        "weather": WeatherData(temperature=24, rainfall=12, humidity=82, sunlight=9, season="rabi"),
        "lang":    "ta",
    },
    {
        "name":    "Maize — Healthy, heat stress",
        "crop":    "maize",
        "disease": "healthy",
        "soil":    SoilData(N=130, P=65, K=90, pH=6.3, OM=2.8),
        "weather": WeatherData(temperature=37, rainfall=35, humidity=55, sunlight=11, season="kharif"),
        "lang":    "sw",
    },
    {
        "name":    "Potato — Nutrient deficiency",
        "crop":    "potato",
        "disease": "nutrient_deficiency",
        "soil":    SoilData(N=30, P=20, K=40, pH=5.4, OM=1.1),
        "weather": WeatherData(temperature=18, rainfall=28, humidity=65, sunlight=8, season="rabi"),
        "lang":    "te",
    },
]


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "█"*60)
    print("  SMART AGRICULTURE AI — FULL PROTOTYPE SYSTEM")
    print("  Python • sklearn • PIL • matplotlib")
    print("█"*60)

    pipeline = SmartAgriPipeline(verbose=True)

    all_results = []
    print("\n" + "═"*60)
    print("  Running demo scenarios...")
    print("═"*60)

    for i, scenario in enumerate(DEMO_SCENARIOS, 1):
        print(f"\n{'━'*60}")
        print(f"  Scenario {i}/{len(DEMO_SCENARIOS)}: {scenario['name']}")
        print(f"  Language: {scenario['lang'].upper()}")
        print(f"{'━'*60}")

        # Create demo image
        demo_img = _create_demo_image(scenario["disease"])

        # Run pipeline
        result = pipeline.run(
            image   = demo_img,
            soil    = scenario["soil"],
            weather = scenario["weather"],
            crop    = scenario["crop"],
            lang    = scenario["lang"],
        )

        # Attach feature importance
        result["yield_predictor_importance"] = pipeline.yield_pred.feature_importance

        # Print localised report
        pipeline.verbose = True
        ml_out = result.get("_ml_out") or MultilingualOutput(scenario["lang"])
        ml_out.print_report(result["localised_report"])

        # Save dashboard
        dash_path = str(RESULTS_DIR / f"dashboard_{i:02d}_{scenario['crop']}.png")
        saved = create_full_dashboard(result, dash_path)
        if saved:
            print(f"  [Viz] Dashboard saved → {saved}")

        # Save heatmap
        heatmap_path = str(RESULTS_DIR / f"heatmap_{i:02d}_{scenario['crop']}.png")
        try:
            saved_hm = pipeline.detector.create_heatmap(demo_img, heatmap_path)
            if saved_hm:
                print(f"  [Viz] Heatmap saved  → {saved_hm}")
        except Exception as e:
            print(f"  [Viz] Heatmap skipped: {e}")

        # Clean result for JSON (remove non-serialisable keys)
        clean = {k: v for k, v in result.items() if k not in ("_ml_out",)}
        if "soil_weather_result" in clean:
            clean["soil_weather_result"].pop("params", None)
        if "disease_result" in clean and clean["disease_result"]:
            clean["disease_result"].pop("treatment", None)
        all_results.append(clean)

    # Save JSON results
    json_path = str(RESULTS_DIR / "all_results.json")
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n  Results saved → {json_path}")
    except Exception as e:
        print(f"\n  JSON save: {e}")

    # ── Final multilingual demo ────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  MULTILINGUAL OUTPUT DEMO")
    print("  Same analysis — 4 languages")
    print("═"*60)

    demo_soil    = SoilData(N=95, P=38, K=70, pH=6.0, OM=1.8)
    demo_weather = WeatherData(temperature=26, rainfall=20, humidity=74, sunlight=8)
    demo_img     = _create_demo_image("early_blight")

    for lang in ["en", "hi", "ta", "bn"]:
        pipeline.verbose = False
        result = pipeline.run(
            image   = demo_img,
            soil    = demo_soil,
            weather = demo_weather,
            crop    = "wheat",
            lang    = lang,

        )
        pipeline.verbose = True
        ml_out = result.get("_ml_out") or MultilingualOutput(lang)
        print(f"\n{'▶'*3} Language: {lang.upper()} ({'─'*40}")
        ml_out.print_report(result["localised_report"])

    print("\n" + "█"*60)
    print(f"  All outputs saved to: {RESULTS_DIR}")
    print("  Files:")
    for f in sorted(RESULTS_DIR.iterdir()):
        size = f.stat().st_size
        print(f"    {f.name:<45} {size/1024:>6.1f} KB")
    print("█"*60 + "\n")


if __name__ == "__main__":
    main()