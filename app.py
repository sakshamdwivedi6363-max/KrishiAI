
from flask import Flask, request, jsonify
from core.soil_weather import SoilData, WeatherData
from core.disease_detector import DiseaseDetector
from core.soil_weather import SoilWeatherAnalyser
from core.yield_predictor import YieldPredictor
from core.recommendation_engine import RecommendationEngine
from core.multilingual import MultilingualOutput
from PIL import Image
import io, base64

app = Flask(__name__)

detector   = DiseaseDetector()
soil_wx    = SoilWeatherAnalyser()
yield_pred = YieldPredictor()
rec_engine = RecommendationEngine()

# ─────────────────────────────────────────────────────────────
# CROP SPECIFIC DISEASE
# ─────────────────────────────────────────────────────────────
CROP_DISEASE = {
    'wheat':     'Leaf Rust',
    'rice':      'Rice Blast',
    'maize':     'Northern Blight',
    'tomato':    'Early Blight',
    'potato':    'Late Blight',
    'cotton':    'Bollworm',
    'soybean':   'Soybean Rust',
    'sugarcane': 'Red Rot',
}

# ─────────────────────────────────────────────────────────────
# CROP SPECIFIC FERTILIZER
# ─────────────────────────────────────────────────────────────
CROP_FERT = {
    'wheat': [
        {'action': 'Apply Urea (46% N)',   'timing': '50% basal before sowing + 50% at tillering stage'},
        {'action': 'Apply DAP (18-46-0)',   'timing': 'Full basal dose before sowing'},
        {'action': 'Apply MOP (60% K₂O)',   'timing': '50% basal + 50% at jointing stage'},
    ],
    'rice': [
        {'action': 'Apply Urea (46% N)',       'timing': '3 splits: basal + tillering + panicle initiation'},
        {'action': 'Apply MOP (60% K₂O)',      'timing': '50% basal + 50% at panicle initiation'},
        {'action': 'Apply Zinc Sulphate',       'timing': '25 kg/ha as basal — rice needs zinc'},
        {'action': 'Apply SSP (16% P₂O₅)',     'timing': 'Full basal before transplanting'},
    ],
    'maize': [
        {'action': 'Apply NPK 10-26-26',        'timing': '150 kg/ha as basal dose at sowing'},
        {'action': 'Apply Urea (top dress)',     'timing': 'At knee-high stage (30 days after sowing)'},
        {'action': 'Apply Sulphur 90%',         'timing': '20 kg/ha basal — improves protein content'},
    ],
    'tomato': [
        {'action': 'Apply Calcium Nitrate',     'timing': 'Fertigation every 5 days after transplanting'},
        {'action': 'Apply K₂SO₄ (SOP)',         'timing': 'Weekly after fruit set — improves quality'},
        {'action': 'Apply Boron 0.1% spray',    'timing': 'At flowering to prevent blossom drop'},
        {'action': 'Apply Magnesium Sulphate',  'timing': '1% foliar spray to prevent yellowing'},
    ],
    'potato': [
        {'action': 'Apply NPK 12-32-16',        'timing': '100 kg/ha at planting time'},
        {'action': 'Apply Urea (top dress)',     'timing': '80 kg/ha at earthing up stage (30 days)'},
        {'action': 'Apply MOP (60% K₂O)',       'timing': '120 kg/ha — potato needs high potassium'},
        {'action': 'Apply Calcium Nitrate',     'timing': '2% foliar spray to prevent hollow heart'},
    ],
    'cotton': [
        {'action': 'Apply DAP (18-46-0)',        'timing': '50 kg/ha as basal before planting'},
        {'action': 'Apply Urea (46% N)',         'timing': '60 kg/ha at squaring stage (45 days)'},
        {'action': 'Apply MOP (60% K₂O)',        'timing': '50 kg/ha at boll development stage'},
        {'action': 'Apply Boron 0.2% spray',     'timing': 'At flowering — reduces boll shedding'},
    ],
    'soybean': [
        {'action': 'Apply Rhizobium inoculant',  'timing': 'Seed treatment 24 hours before sowing'},
        {'action': 'Apply SSP (16% P₂O₅)',       'timing': '200 kg/ha as basal — soybean needs P'},
        {'action': 'Apply MOP (60% K₂O)',        'timing': '80 kg/ha at pod filling stage'},
    ],
    'sugarcane': [
        {'action': 'Apply Urea (46% N)',         'timing': '180 kg/ha in 3 equal splits over season'},
        {'action': 'Apply DAP (18-46-0)',         'timing': '100 kg/ha as basal at planting'},
        {'action': 'Apply MOP (60% K₂O)',        'timing': '120 kg/ha split in 2 doses'},
        {'action': 'Apply Pressmud + FYM',       'timing': '10 t/ha at planting for organic matter'},
    ],
}

# ─────────────────────────────────────────────────────────────
# CROP SPECIFIC IMPROVEMENT TIPS
# ─────────────────────────────────────────────────────────────
CROP_IMPROVE = {
    'wheat': [
        'Use certified HD-2967 or WH-1105 variety for 15% higher yield',
        'Timely sowing Nov 1–15 is critical — late sowing reduces yield 30 kg/ha per day delay',
        'Seed rate: 100 kg/ha for normal sowing, 125 kg/ha for late sowing',
        'Ensure proper irrigation at crown root initiation (21 days) and tillering (45 days)',
    ],
    'rice': [
        'Use SRI method (System of Rice Intensification) to save 30% water and increase yield',
        'Transplant at 21-day seedling age for maximum tillering and yield',
        'Maintain 2–3 cm water level during tillering — drain at maturity stage',
        'Use IR-64 or BPT-5204 varieties for high yield in irrigated conditions',
    ],
    'maize': [
        'Use single cross hybrid seeds — 40% more yield than open pollinated varieties',
        'Maintain 60×20 cm spacing for optimal plant population of 83,000 plants/ha',
        'Apply weedicide atrazine 1.5 kg/ha within 3 days of sowing for weed control',
        'Detassel female rows in seed production plots for hybrid seed purity',
    ],
    'tomato': [
        'Use drip irrigation + plastic mulch to prevent soil splash and reduce blight by 60%',
        'Stake or trellis plants at 30 cm height to prevent lodging and disease',
        'Remove suckers weekly below first flower cluster to improve fruit size',
        'Use grafted seedlings on disease-resistant rootstock for soil-borne disease control',
    ],
    'potato': [
        'Use certified seed tubers — avoid saved seed to prevent virus buildup',
        'Hill up soil 2–3 times to prevent tuber greening and improve yield',
        'Harvest when 75% vines turn yellow — delay causes skin damage',
        'Apply copper fungicide every 7 days in humid conditions to prevent late blight',
    ],
    'cotton': [
        'Use Bt cotton varieties to reduce bollworm pesticide cost by 70%',
        'Top the plant at 90 days to stop vegetative growth and improve boll set',
        'Install pheromone traps at 5 per acre for early bollworm monitoring',
        'Avoid excessive nitrogen — causes vegetative growth and boll shedding',
    ],
    'soybean': [
        'Inoculate seeds with Bradyrhizobium — saves 60–80 kg/ha urea cost',
        'Avoid waterlogging — soybean roots rot within 48 hours of flooding',
        'Harvest at 95% pod maturity to avoid pod shattering losses',
        'Intercrop with maize 4:2 ratio for better land utilization and income',
    ],
    'sugarcane': [
        'Use single budded setts — saves 40% seed cane cost vs whole stalk planting',
        'Trash mulching after harvest retains soil moisture and suppresses weeds',
        'Ratoon crop: apply fertilizer within 7 days of harvest for quick regrowth',
        'Install drip irrigation to save 40% water and increase sugar recovery',
    ],
}

# ─────────────────────────────────────────────────────────────
# CROP SPECIFIC IRRIGATION METHOD
# ─────────────────────────────────────────────────────────────
CROP_IRRIGATION = {
    'wheat':     'Furrow irrigation — avoid wetting leaves to prevent rust',
    'rice':      'Flood irrigation — maintain 3–5 cm standing water during tillering',
    'maize':     'Furrow or drip irrigation — critical at tasseling and grain fill',
    'tomato':    'Drip irrigation only — never wet foliage, prevents blight spread',
    'potato':    'Sprinkler or drip — avoid waterlogging, damages tubers',
    'cotton':    'Furrow irrigation — critical at flowering and boll development',
    'soybean':   'Sprinkler — critical at flowering and pod filling stages',
    'sugarcane': 'Flood or drip — water-loving crop, needs regular irrigation',
}

# ─────────────────────────────────────────────────────────────
# SEASON SPECIFIC DATA
# ─────────────────────────────────────────────────────────────
SEASON_DATA = {
    'kharif': {
        'crops':        ['rice', 'maize', 'cotton', 'soybean', 'sugarcane'],
        'months':       'June to October',
        'tip':          'Kharif season — monsoon crops. Sow after first good rain (50–75 mm).',
        'warning':      'High humidity in monsoon — monitor for fungal diseases every week.',
        'irrigation':   'Monsoon rains usually sufficient. Irrigate only during 2-week dry spells.',
        'fert_timing':  'Apply basal dose at sowing. Top dress nitrogen after 30 days.',
        'special':      'Drain excess water immediately — waterlogging kills roots within 48 hours.',
    },
    'rabi': {
        'crops':        ['wheat', 'potato', 'tomato', 'chickpea', 'mustard'],
        'months':       'October to March',
        'tip':          'Rabi season — winter crops. Sow after monsoon withdrawal (Oct 15–Nov 15).',
        'warning':      'Cold nights below 5°C can damage young crop — monitor temperature.',
        'irrigation':   'No monsoon support — must irrigate at crown root (21d), tillering (45d), flowering (65d).',
        'fert_timing':  'Apply full basal at sowing. Top dress nitrogen at 21 days after emergence.',
        'special':      'Fog and dew in Dec–Jan increase disease risk — apply preventive fungicide.',
    },
    'zaid': {
        'crops':        ['maize', 'cucumber', 'watermelon', 'vegetables', 'fodder'],
        'months':       'March to June',
        'tip':          'Zaid season — summer crops. Short duration 60–90 days. High temperature.',
        'warning':      'Heat stress above 40°C — mulch soil and irrigate in early morning only.',
        'irrigation':   'Critical season — irrigate every 2–3 days. Drip system strongly recommended.',
        'fert_timing':  'Use quick-release fertilizers — crop duration is short, fast nutrition needed.',
        'special':      'Use shade nets for vegetables. Harvest early morning to maintain quality.',
    },
}

# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────
@app.route('/')
def home():
    with open('farmer_dashboard.html', 'r', encoding='utf-8') as f:
        return f.read()


@app.route('/api/analyse', methods=['POST'])
def analyse():
    data   = request.json
    crop   = data.get('crop',   'wheat').lower()
    lang   = data.get('lang',   'en')
    season = data.get('season', 'rabi').lower()

    # ── Image ────────────────────────────────────────────────
    image = None
    if data.get('image'):
        try:
            img_bytes = base64.b64decode(data['image'].split(',')[1])
            image     = Image.open(io.BytesIO(img_bytes))
        except Exception:
            image = None

    # ── Soil inputs ──────────────────────────────────────────
    soil = SoilData(
        N  = float(data.get('N',   80)),
        P  = float(data.get('P',   40)),
        K  = float(data.get('K',   60)),
        pH = float(data.get('pH',  6.5)),
        OM = float(data.get('OM',  2.5)),
    )

    # ── Weather inputs ────────────────────────────────────────
    weather = WeatherData(
        temperature = float(data.get('temperature', 25)),
        rainfall    = float(data.get('rainfall',    30)),
        humidity    = float(data.get('humidity',    65)),
        sunlight    = float(data.get('sunlight',     8)),
        season      = season,
    )

    # ── Run AI modules ────────────────────────────────────────
    disease_result  = detector.predict(image, crop) if image else None
    sw_result       = soil_wx.analyse(soil, weather, crop)
    yield_result    = yield_pred.predict(sw_result['engineered_features'], crop)
    recommendations = rec_engine.generate(disease_result, sw_result, yield_result, crop)

    # ── Disease — crop specific ───────────────────────────────
    disease_name = CROP_DISEASE.get(crop, 'Unknown Disease')
    humidity     = weather.humidity
    temperature  = weather.temperature
    disease_risk = humidity > 75 and temperature > 18
    disease_conf = round(55 + humidity * 0.25 + (temperature - 18) * 0.8, 1) if disease_risk else 0

    # ── Fertilizer — crop + deficit specific ─────────────────
    params = sw_result['params']
    N_def  = max(0, params['N_opt'] - soil.N)
    P_def  = max(0, params['P_opt'] - soil.P)
    K_def  = max(0, params['K_opt'] - soil.K)

    final_ferts = []

    # pH correction — highest priority
    ph_status = sw_result['ph_analysis']['status']
    if ph_status != 'optimal':
        correction = 'Agricultural Lime' if soil.pH < 6.0 else 'Elemental Sulphur'
        amt        = round(abs(soil.pH - 6.5) * 1.5, 1)
        final_ferts.append({
            'action': f'Apply {correction} — FIRST PRIORITY',
            'dose':   f'{amt} t/ha',
            'timing': '6 weeks before sowing — nutrients unavailable at wrong pH',
        })

    for f in CROP_FERT.get(crop, CROP_FERT['wheat']):
        dose = 'As recommended'
        if 'Urea'   in f['action'] and N_def > 0:
            dose = f'{round(N_def * 2.17)} kg/ha'
        elif 'DAP'  in f['action'] and P_def > 0:
            dose = f'{round(P_def * 2.17)} kg/ha'
        elif 'MOP'  in f['action'] and K_def > 0:
            dose = f'{round(K_def * 1.67)} kg/ha'
        elif 'SSP'  in f['action'] and P_def > 0:
            dose = f'{round(P_def * 6.25)} kg/ha'
        elif 'NPK'  in f['action']:
            dose = f['action'].split('Apply ')[-1] + ' — standard dose'
            dose = 'As per label'
        final_ferts.append({
            'action': f['action'],
            'dose':   dose,
            'timing': f['timing'],
        })

    # ── Irrigation — crop + season specific ──────────────────
    irr         = recommendations['irrigation']
    irr['method'] = CROP_IRRIGATION.get(crop, 'Drip irrigation')

    season_info         = SEASON_DATA.get(season, SEASON_DATA['rabi'])
    irr['note']         = season_info['irrigation']
    irr['season_tip']   = season_info['special']

    # ── Season warning if wrong crop ─────────────────────────
    season_warning = ''
    if crop not in season_info['crops']:
        best = ', '.join(season_info['crops'][:3])
        season_warning = (
            f"{crop.title()} is not ideal for {season.title()} season. "
            f"Best crops for {season.title()}: {best}. "
            f"Yield may be 20–30% lower than optimal season."
        )

    # ── Improvement tips — crop specific ─────────────────────
    improve_tips = CROP_IMPROVE.get(crop, [])

    # Add season specific tip at top
    improve_tips = [season_info['tip']] + improve_tips

    # ── Priority action ───────────────────────────────────────
    soil_score = sw_result['soil_health_score']

    if disease_risk and image:
        priority = (
            f"Treat {disease_name} immediately — "
            f"apply fungicide within 48 hours. "
            f"Confidence: {disease_conf}%"
        )
    elif soil.pH < 5.8:
        priority = (
            f"URGENT: Soil pH is {soil.pH} — too acidic. "
            f"Apply lime before any fertilizer. "
            f"Nutrients are locked at this pH."
        )
    elif N_def > params['N_opt'] * 0.40:
        priority = (
            f"Critical nitrogen deficiency for {crop.title()}. "
            f"Apply Urea {round(N_def * 2.17)} kg/ha this week. "
            f"Yield loss of 30–40% if not corrected."
        )
    elif irr['status'] == 'critical_deficit':
        priority = (
            f"Urgent irrigation needed — "
            f"{crop.title()} water deficit is critical. "
            f"Irrigate within 24 hours."
        )
    elif soil_score < 40:
        priority = (
            f"Very low soil health ({soil_score}/100). "
            f"Add organic matter — FYM 5–10 t/ha — before next crop."
        )
    elif season_warning:
        priority = (
            f"Wrong season for {crop.title()}. "
            f"Consider switching to: {', '.join(season_info['crops'][:2])}. "
            f"Or proceed with extra care."
        )
    else:
        priority = (
            f"Conditions moderate for {crop.title()} in {season.title()} season. "
            f"Follow fertilizer schedule and monitor weekly."
        )

    # ── Return JSON ───────────────────────────────────────────
    return jsonify({
        'yield':           yield_result['predicted_yield'],
        'yield_low':       yield_result['yield_range'][0],
        'yield_high':      yield_result['yield_range'][1],
        'grade':           yield_result['grade'],
        'soil_health':     sw_result['soil_health_score'],
        'overall_score':   recommendations['overall_score'],
        'disease':         disease_name if (disease_risk and image) else 'healthy',
        'confidence':      disease_conf,
        'is_healthy':      not (disease_risk and image),
        'priority':        priority,
        'fertilizer':      final_ferts,
        'irrigation':      irr,
        'improvement':     [{'action': tip} for tip in improve_tips],
        'crop':            crop,
        'season':          season,
        'season_months':   season_info['months'],
        'season_tip':      season_info['tip'],
        'season_warning':  season_warning,
        'fert_timing':     season_info['fert_timing'],
        'season_special':  season_info['special'],
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)