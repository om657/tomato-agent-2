import base64
import io
import json
import os
from datetime import datetime
import hashlib
import time

import requests
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from PIL import Image
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

# ================= CONFIG ================= #
# API key intentionally kept in-code per requirement.
NVIDIA_API_KEY = "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"
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL_NAME = "mistralai/mistral-large-3-675b-instruct-2512"
VISION_MODEL = "mistralai/mistral-large-3-675b-instruct-2512"
REASONING_MODEL = "deepseek-ai/deepseek-v4-flash"
SARVAM_API_KEY = "sk_w2ff5un8_NwCt6eC0gfgjKezrySpjUCKn"
SARVAM_TRANSLATE_URL = "https://api.sarvam.ai/translate"
USER_DB = "users.json"
EXPORT_DIR = "exports"

# ============== REAL-TIME NEARBY LOOKUPS (OSM) ============== #
_OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.nchc.org.tw/api/interpreter",
]


def _requests_session():
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": "tomato-agent/1.0 (streamlit; contact: local)",
            "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
        }
    )
    return s


@st.cache_data(show_spinner=False, ttl=60 * 30)
def geocode_location_osm(query: str):
    if not query or not query.strip():
        return None
    q = query.strip()
    try:
        geolocator = Nominatim(user_agent="tomato-agent-streamlit")
        loc = geolocator.geocode(q, addressdetails=True, timeout=10)
        if not loc:
            return None
        return float(loc.latitude), float(loc.longitude), getattr(loc, "address", q)
    except Exception:
        return None


def _overpass_request(query: str, timeout_s: int = 15):
    s = _requests_session()
    last_err = None
    for url in _OVERPASS_ENDPOINTS:
        try:
            r = s.post(url, data={"data": query}, timeout=timeout_s)
            if r.status_code == 200 and "application/json" in r.headers.get("Content-Type", ""):
                return r.json()
            last_err = f"HTTP {r.status_code}: {r.text[:200]}"
        except requests.exceptions.RequestException as e:
            last_err = str(e)
    return {"elements": []}


# ---- Lightweight exclusion lists (only remove truly irrelevant) ---- #
_SHOP_EXCLUDE = {
    "atm", "bank", "fuel", "petrol", "gas station",
}
_DOCTOR_EXCLUDE = {
    "atm", "bank", "fuel", "petrol", "gas station",
    "restaurant", "hotel", "cafe",
}


def _should_exclude(name: str, tags: dict, actor: str) -> bool:
    """Exclude only truly irrelevant places (ATMs, petrol pumps, etc.)."""
    combined = (name.strip().lower() + " " +
                " ".join(str(v).lower() for v in tags.values()))
    exclude_set = _SHOP_EXCLUDE if actor.lower().startswith("shop") else _DOCTOR_EXCLUDE
    return any(excl in combined for excl in exclude_set)


@st.cache_data(show_spinner=False, ttl=60 * 5)
def nearby_osm_places(lat: float, lon: float, actor: str, radius_m: int = 8000):
    """Search OSM for shops or doctors/clinics near the given location."""

    if actor.lower().startswith("shop"):
        # Search ALL shops + marketplaces + any node with agriculture-ish name
        selectors = [
            # ALL shops in the area (any type)
            'node["shop"](around:{r},{lat},{lon});',
            'way["shop"](around:{r},{lat},{lon});',
            # Marketplaces / mandis
            'node["amenity"="marketplace"](around:{r},{lat},{lon});',
            'way["amenity"="marketplace"](around:{r},{lat},{lon});',
            # Craft / trade
            'node["craft"](around:{r},{lat},{lon});',
            'way["craft"](around:{r},{lat},{lon});',
            # Landuse retail
            'way["landuse"="retail"](around:{r},{lat},{lon});',
        ]
    else:
        # Search ALL medical facilities
        selectors = [
            'node["amenity"="doctors"](around:{r},{lat},{lon});',
            'node["amenity"="clinic"](around:{r},{lat},{lon});',
            'node["amenity"="hospital"](around:{r},{lat},{lon});',
            'node["amenity"="pharmacy"](around:{r},{lat},{lon});',
            'node["amenity"="veterinary"](around:{r},{lat},{lon});',
            'node["healthcare"](around:{r},{lat},{lon});',
            'way["amenity"="doctors"](around:{r},{lat},{lon});',
            'way["amenity"="clinic"](around:{r},{lat},{lon});',
            'way["amenity"="hospital"](around:{r},{lat},{lon});',
            'way["amenity"="pharmacy"](around:{r},{lat},{lon});',
            'way["amenity"="veterinary"](around:{r},{lat},{lon});',
            'way["healthcare"](around:{r},{lat},{lon});',
        ]

    results = _run_osm_search(lat, lon, selectors, radius_m, actor)

    # Fallback: if too few results, auto-retry with bigger radius
    if len(results) < 3 and radius_m < 30000:
        results = _run_osm_search(lat, lon, selectors, min(radius_m * 2, 30000), actor)

    # If still no results, try maximum radius
    if not results and radius_m < 50000:
        results = _run_osm_search(lat, lon, selectors, 50000, actor)

    return results[:25]


def _run_osm_search(lat, lon, selectors, radius_m, actor):
    """Execute Overpass query and return filtered results."""
    q = (
        "[out:json][timeout:25];("
        + "".join(s.format(r=radius_m, lat=lat, lon=lon) for s in selectors)
        + ");out center tags;"
    )
    try:
        data = _overpass_request(q, timeout_s=25)
    except Exception:
        return []
    elements = data.get("elements", [])
    results = []
    origin = (lat, lon)
    seen_names = set()
    for el in elements:
        plat = el.get("lat") or (el.get("center") or {}).get("lat")
        plon = el.get("lon") or (el.get("center") or {}).get("lon")
        if plat is None or plon is None:
            continue
        tags = el.get("tags", {}) or {}
        name = tags.get("name") or tags.get("brand") or ""
        if not name or name.lower() == "unnamed":
            continue
        # Only exclude truly irrelevant (ATMs, petrol pumps)
        if _should_exclude(name, tags, actor):
            continue
        # Deduplicate by name (case-insensitive)
        name_key = name.strip().lower()
        if name_key in seen_names:
            continue
        seen_names.add(name_key)
        dist_km = float(geodesic(origin, (plat, plon)).km)
        results.append(
            {
                "name": name,
                "lat": float(plat),
                "lon": float(plon),
                "tags": tags,
                "distance_km": dist_km,
            }
        )
    results.sort(key=lambda x: x["distance_km"])
    return results


def tomato_fertilizer_plan_text():
    return """
**Tomato fertilizer plan (general guide, adjust to soil test):**

- **Basal (before transplanting, per acre)**
  - Well-decomposed FYM/compost: 4–6 tons
  - DAP: 45–55 kg (or SSP equivalent for phosphorus)
  - MOP (Potash): 35–45 kg
  - Neem cake (optional): 50–100 kg

- **Top dress / split N & K**
  - **20–25 DAT**: Urea 25–30 kg + MOP 15–20 kg
  - **40–45 DAT (flowering)**: Urea 25–30 kg + MOP 15–20 kg
  - **60–65 DAT (fruit set)**: Urea 15–20 kg + MOP 15–20 kg

- **Micronutrients**
  - **Calcium + Boron**: Ca(NO3)2 1% foliar + Boron 0.2%
  - **Magnesium**: MgSO4 1% foliar
"""


def _as_lines(value):
    """Normalize mixed AI outputs (list/dict/str) into bullet-friendly lines."""
    if value is None:
        return []
    if isinstance(value, list):
        out = []
        for x in value:
            if isinstance(x, str) and x.strip():
                out.append(x.strip())
            elif isinstance(x, dict):
                for k, v in x.items():
                    out.append(f"{k}: {v}")
        return out
    if isinstance(value, dict):
        return [f"{k}: {v}" for k, v in value.items()]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        # Split on newlines if it's a multi-line block
        if "\n" in s:
            return [ln.strip("-• \t") for ln in s.splitlines() if ln.strip()]
        return [s]
    return [str(value)]


def _extract_fertilizer_names(value):
    """
    Extract searchable fertilizer/product names from AI output.
    This is heuristic but works well for typical outputs.
    """
    lines = _as_lines(value)
    names = []
    for ln in lines:
        # Remove numbering like "1." or "0:"
        cleaned = ln.strip()
        cleaned = cleaned.lstrip("0123456789").lstrip(").:- ").strip()
        # Prefer short-ish “names”, but keep meaningful phrases
        if 2 <= len(cleaned) <= 80:
            names.append(cleaned)
    # de-dup preserving order
    seen = set()
    out = []
    for n in names:
        k = n.lower()
        if k not in seen:
            seen.add(k)
            out.append(n)
    return out[:12]


def render_analysis_report(res: dict, location: str):
    crop = res.get("crop_name", "Unknown")
    disease = res.get("disease_name", "Healthy")
    risk = res.get("risk_score", "None")

    st.markdown("---")
    st.markdown(
        f"""
        <div class="ta-card">
          <div style="display:flex; justify-content:space-between; gap:14px; flex-wrap:wrap;">
            <div>
              <div class="ta-title" style="font-size:18px;">{t("Analysis Report")}</div>
              <div class="ta-muted" style="font-size:13px;">{t("Fast summary and actionable steps")}</div>
            </div>
            <div style="display:flex; gap:10px; flex-wrap:wrap; align-items:center;">
              <span class="ta-badge">🌱 {t("Crop")}: {crop}</span>
              <span class="ta-badge">🦠 {t("Disease")}: {disease}</span>
              <span class="ta-badge">⚠️ {t("Risk")}: {risk}</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Quick actions
    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_a:
        st.download_button(
            t("Download JSON"),
            data=json.dumps(res, ensure_ascii=False, indent=2),
            file_name="analysis.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_b:
        st.download_button(
            t("Download TXT"),
            data="\n".join(
                [
                    f"Crop: {crop}",
                    f"Disease: {disease}",
                    f"Risk: {risk}",
                    "",
                    "Assessment:",
                    *( _as_lines(res.get("description")) ),
                    "",
                    "Prescription:",
                    *( _as_lines(res.get("solution")) ),
                    "",
                    "Soil:",
                    *( _as_lines(res.get("soil_insights")) ),
                    "",
                    "Water:",
                    *( _as_lines(res.get("water_forecast")) ),
                    "",
                    "Fertilizers:",
                    *( _as_lines(res.get("fertilizers")) ),
                ]
            ),
            file_name="analysis.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with col_c:
        if st.button(t("Clear result"), use_container_width=True):
            st.session_state.detection_result = None
            st.rerun()

    tab_assess, tab_actions, tab_soil, tab_water, tab_fert = st.tabs(
        [t("Assessment"), t("Prescription"), t("Soil"), t("Water"), t("Fertilizers & Suppliers")]
    )

    with tab_assess:
        st.markdown("<div class='ta-card'>", unsafe_allow_html=True)
        st.subheader(t("Condition Assessment"))
        st.write(res.get("description", t("No description available.")))
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_actions:
        st.markdown("<div class='ta-card'>", unsafe_allow_html=True)
        st.subheader(t("Actionable Prescription"))
        sol = res.get("solution", t("No solution provided."))
        lines = _as_lines(sol)
        if len(lines) > 1:
            for ln in lines:
                st.write(f"- {ln}")
        else:
            st.write(sol)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_soil:
        st.markdown("<div class='ta-card'>", unsafe_allow_html=True)
        st.subheader(t("Soil and Moisture Insights"))
        soil = res.get("soil_insights", "")
        if isinstance(soil, dict):
            st.write(f"**{t('Nutrients')}**: {soil.get('nutrients','')}")
            st.write(f"**{t('pH')}**: {soil.get('pH','')}")
            st.write(f"**{t('Moisture')}**: {soil.get('moisture','')}")
        else:
            st.write(soil or t("No soil insights available."))
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_water:
        st.markdown("<div class='ta-card'>", unsafe_allow_html=True)
        st.subheader(t("Water and Weather Outlook"))
        water = res.get("water_forecast", "")
        if isinstance(water, dict):
            st.write(f"**{t('Forecast')}**: {water.get('forecast_next_7_days','')}")
            st.write(f"**{t('Irrigation suggestions')}**: {water.get('irrigation_suggestions','')}")
        else:
            st.write(water or t("No water forecast available."))
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_fert:
        st.markdown("<div class='ta-card'>", unsafe_allow_html=True)
        st.subheader(t("Fertilizer Recommendations"))
        fert_val = res.get("fertilizers", "")
        fert_lines = _as_lines(fert_val)
        if fert_lines:
            for ln in fert_lines:
                st.write(f"- {ln}")
        else:
            st.write(t("No fertilizer recommendations available."))

        st.markdown("---")
        st.subheader(t("Search suppliers (one click)"))
        fert_names = _extract_fertilizer_names(fert_val)
        loc = (location or "").strip() or "near me"
        if fert_names:
            for name in fert_names:
                q = requests.utils.requote_uri(f"{name} supplier price {loc}")
                st.link_button(f"{t('Search')} — {name}", url=f"https://www.google.com/search?q={q}")
        else:
            q = requests.utils.requote_uri(f"fertilizer shop supplier {loc}")
            st.link_button(t("Search fertilizer suppliers"), url=f"https://www.google.com/search?q={q}")

        st.markdown("</div>", unsafe_allow_html=True)

# ================= LANGUAGE / FONT ================= #
TRANSLATIONS = {
    "English": {
        "home": "Home",
        "chat": "Chat",
        "shops": "Shop",
        "doctors": "Doctors",
        "contact": "Contact",
        "login": "Login",
        "username": "Username",
        "password": "Password",
        "upload": "Upload Leaf Image",
        "analyze": "Analyze",
        "btn_desc": "📄 Disease Description",
        "btn_sol": "💡 Get Solution",
        "btn_fert": "🧪 Get Fertilizers",
    },
    "Hindi": {
        "home": "होम",
        "chat": "चैट",
        "shops": "दुकान",
        "doctors": "डॉक्टर्स",
        "contact": "संपर्क",
        "login": "लॉगिन",
        "username": "यूज़रनेम",
        "password": "पासवर्ड",
        "upload": "पत्ता अपलोड करें",
        "analyze": "विश्लेषण",
        "btn_desc": "📄 बीमारी का विवरण",
        "btn_sol": "💡 समाधान प्राप्त करें",
        "btn_fert": "🧪 उर्वरक प्राप्त करें",
    },
    "Marathi": {
        "home": "मुख्यपृष्ठ",
        "chat": "चॅट",
        "shops": "दुकान",
        "doctors": "डॉक्टर्स",
        "contact": "संपर्क",
        "login": "लॉगिन",
        "username": "वापरकर्ता नाव",
        "password": "पासवर्ड",
        "upload": "पान अपलोड करा",
        "analyze": "विश्लेषण",
        "btn_desc": "📄 रोगाचे वर्णन",
        "btn_sol": "💡 उपाय मिळवा",
        "btn_fert": "🧪 खते मिळवा",
    },
}
FONT_MAP = {
    "English": "Arial, sans-serif",
    "Hindi": "'Nirmala UI', 'Mangal', sans-serif",
    "Marathi": "'Noto Sans Devanagari', 'Mangal', sans-serif",
}
LANGUAGE_CODE_MAP = {
    "English": "en-IN",
    "Hindi": "hi-IN",
    "Marathi": "mr-IN",
}

ACTION_MAP = {
    "Soil moisture modeling": "Analyze soil moisture modeling with sensor + weather assumptions and actionable irrigation guidance.",
    "Water requirement prediction": "Predict farm water requirement for next 14 days by crop stage and weather uncertainty.",
    "AI-driven irrigation schedule": "Create AI-driven irrigation schedule with time windows and liters/acre.",
    "Drought early warning": "Generate drought early warning indicators for 30 days.",
    "Water waste optimization %": "Estimate current water waste percentage and optimization opportunities.",
    "NPK prediction": "Predict nitrogen, phosphorus, potassium levels and corrective plan.",
    "pH imbalance detection": "Detect pH imbalance and recommend treatment protocol.",
    "Nutrient deficiency fusion": "Use leaf + soil fusion assumptions to identify nutrient deficiencies.",
    "Fertilizer recommendation": "Build fertilizer recommendation engine output for this farm.",
    "Long-term soil health score": "Estimate long-term soil health score and yearly action plan.",
    "Insect classification": "Classify likely insects and risk level by season.",
    "Pest density estimation": "Estimate pest density per acre with intervention threshold.",
    "Swarm detection": "Detect swarm risk and alert plan.",
    "Migration pattern prediction": "Predict wind-based pest migration pattern over 7 days.",
    "Smart pesticide timing": "Recommend ideal pesticide application timing.",
    "Satellite imagery integration": "Provide satellite imagery integration plan and inferred crop signals.",
    "Growth stage tracking": "Track crop growth stage and next milestones.",
    "Production estimate per acre": "Estimate production per acre with confidence range.",
    "Profit forecast": "Generate profit forecast using yield, costs, and market price assumptions.",
    "Market price integration": "Integrate market price trend and suggest sell timing.",
    "Camera→Analyze→Recommend→Auto-execute": "Design camera-to-execution pipeline with automation gates.",
    "Irrigation valve control": "Generate irrigation valve control logic and failsafe.",
    "Sprayer control": "Generate smart sprayer control strategy.",
    "Drone-based spraying": "Plan drone-based spraying route and timing.",
    "Automated farm reporting": "Create automated farm reporting template and KPI plan.",
    "Multi-modal fusion model": "Design fusion model: Vision + Weather + Soil + Time.",
    "Disease risk 7-30 days": "Predict disease risk for 7-30 days using humidity + temperature.",
    "Frost risk alerts": "Predict frost risk and preventive actions.",
    "Heat stress prediction": "Predict heat stress windows and protection actions.",
    "Crop growth stage mapping": "Generate crop growth stage map from multimodal data.",
    "Price prediction AI": "Calculate total crop production cost and expected local market gain/profit.",
    "Full Agent Pipeline": "Build one proper end-to-end AI agent pipeline using Vision/Climate/Soil/Water/Market/Execution layers.",
}


def call_openrouter(messages, model=REASONING_MODEL):
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"model": model, "messages": messages}
    try:
        response = requests.post(NVIDIA_API_URL, headers=headers, json=payload, timeout=6000)
        
        # 🔥 Check status first
        if response.status_code != 200:
            return f"HTTP Error {response.status_code}: {response.text}"

        # 🔥 Ensure JSON response
        if "application/json" not in response.headers.get("Content-Type", ""):
            return f"API returned non-JSON response:\n{response.text[:500]}"

        data = response.json()
        if "choices" in data:
            return data["choices"][0]["message"]["content"]
        if "error" in data:
            return f"API Error: {data['error'].get('message')}"
        return f"Unexpected format: {data}"
    except requests.exceptions.RequestException as e:
        return f"Network Error: {str(e)}"


def run_reasoning_model(image_bytes, species_info):
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    system_prompt = """You are a world-class plant pathologist and agronomist with 30 years of field experience.
Your PRIMARY task is ACCURATE crop/plant identification followed by disease diagnosis.

CRITICAL RULES FOR CROP IDENTIFICATION — follow these strictly:
1. FIRST analyze the leaf morphology before naming ANY crop:
   - Leaf shape (simple vs compound, lobed vs entire, serrated vs smooth margins)
   - Leaf arrangement (alternate, opposite, whorled)
   - Leaf texture (smooth, hairy/pubescent, waxy, sticky/glandular)
   - Leaf size and proportions
   - Venation pattern (parallel vs reticulate/net-veined)
   - Stem characteristics visible (woody, herbaceous, color)
   - Any flowers, fruits, or other distinguishing features visible

2. KEY DISTINGUISHING FEATURES — do NOT confuse these:
   - TOMATO: compound leaves with 5-9 irregularly lobed leaflets, strong tomato smell, hairy/glandular stems, serrated leaflet margins, reticulate venation
   - SUGARCANE: long narrow strap-like leaves (60-150cm), parallel venation, prominent midrib, smooth waxy surface, grows from thick jointed stalks/canes
   - POTATO: compound leaves similar to tomato but leaflets more oval/rounded
   - PEPPER: simple ovate leaves, smooth, alternate arrangement
   - CORN/MAIZE: long linear leaves, parallel venation, prominent midrib, alternate on thick stalk

3. If the leaf has RETICULATE (net-like) venation → it is a DICOT (tomato, potato, pepper, etc.), NOT a monocot (sugarcane, corn, rice, wheat)
4. If the leaf has PARALLEL venation → it is a MONOCOT (sugarcane, corn, rice, wheat), NOT a dicot
5. Tomato and sugarcane look NOTHING alike — tomato has small compound serrated leaflets; sugarcane has long strap-shaped parallel-veined blades.
6. NEVER guess. If uncertain about the crop, state your confidence level and the top 2-3 possibilities."""

    user_prompt = f"""Analyze this plant leaf image carefully using the morphological identification protocol.
Context/Metadata: {json.dumps(species_info)}

STEP 1 — CROP IDENTIFICATION (most important):
Examine the leaf morphology in detail (shape, margins, venation, texture, arrangement).
Based ONLY on the visual evidence in the image, identify the exact crop/plant species.

STEP 2 — DISEASE DIAGNOSIS:
Once the crop is correctly identified, diagnose any disease or health issue.
If healthy, state 'Healthy'.

STEP 3 — AGRONOMIC ADVICE:
Provide soil insights, water/irrigation guidance, and fertilizer recommendations specific to the CORRECTLY identified crop and its condition.

Return ONLY valid JSON (no extra text, no markdown) in this exact structure:
{{
    "crop_name": "Exact name of the crop based on leaf morphology analysis",
    "disease_name": "Specific disease name or 'Healthy'",
    "description": "Detailed description including leaf morphology observations that led to crop identification, and the disease condition",
    "solution": "Step-by-step treatment or care instructions specific to this crop and disease",
    "fertilizers": "Specific fertilizer recommendations for this crop and condition",
    "soil_insights": "Soil health insights (nutrients, pH, moisture) relevant to this crop",
    "water_forecast": "Water and irrigation recommendations for this crop",
    "risk_score": "Low/Medium/High"
}}"""

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0.1,
        "top_p": 0.9,
    }

    response = requests.post(NVIDIA_API_URL, headers=headers, json=payload, timeout=6000)
    
    if response.status_code != 200:
        return {"error": f"HTTP Error {response.status_code}: {response.text}"}

    if "application/json" not in response.headers.get("Content-Type", ""):
        return {"error": "API returned non-JSON response", "raw": response.text[:500]}

    try:
        result = response.json()
    except requests.exceptions.JSONDecodeError:
         return {"error": "Failed to decode JSON", "raw": response.text[:500]}

    if "choices" not in result:
        err_msg = "Unknown error"
        if "error" in result:
            err_msg = result["error"].get("message", "Unknown error")
        return {"error": f"API Error: {err_msg}", "raw_response": result}

    try:
        output_text = result["choices"][0]["message"]["content"]
        # Remove markdown code blocks if present
        if "```json" in output_text:
            output_text = output_text.split("```json")[1].split("```")[0].strip()
        elif "```" in output_text:
            output_text = output_text.split("```")[1].split("```")[0].strip()
            
        return json.loads(output_text)
    except Exception as e:
        return {"error": f"Reasoning model failed to parse output: {str(e)}", "raw_response": result}


def _preprocess_image_for_api(image: Image.Image, max_side_px: int = 1024, jpeg_quality: int = 90) -> bytes:
    """
    Speed optimization:
    - Convert to RGB
    - Resize to a sane max side (keeps accuracy for leaf disease while cutting upload size)
    - Encode to JPEG with moderate quality
    """
    if image.mode != "RGB":
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        image = background

    w, h = image.size
    scale = max(w, h) / float(max_side_px)
    if scale > 1.0:
        new_w = int(round(w / scale))
        new_h = int(round(h / scale))
        image = image.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=jpeg_quality, optimize=True, progressive=True)
    return buf.getvalue()


@st.cache_data(show_spinner=False, ttl=60 * 60)
def analyze_leaf_cached(image_jpeg_bytes: bytes, location: str):
    # Cache key includes location because it changes the recommendations.
    species_info = {"location": location or ""}
    return run_reasoning_model(image_jpeg_bytes, species_info)


def ensure_session_defaults():
    defaults = {
        "language": "English",
        "logged_in": False,
        "username": "",
        "photo_url": "https://api.dicebear.com/8.x/adventurer/png?seed=Farmer",
        "agent_status": "Idle",
        "task_queue": [],
        "reports": [],
        "chat_history": [],
        "detection_result": None,
        "menu_choice": "Home",
        "location": "",
        "cost_estimation": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def apply_local_font(language):
    font_family = FONT_MAP.get(language, FONT_MAP["English"])
    st.markdown(
        f"""
        <style>
            html, body, [class*="css"], .stApp {{
                font-family: {font_family};
            }}

            .stApp {{
                background: radial-gradient(1200px 700px at 10% -10%, rgba(34,197,94,0.14), transparent 60%),
                            radial-gradient(900px 600px at 90% 0%, rgba(16,185,129,0.10), transparent 55%),
                            linear-gradient(180deg, rgba(2,6,23,0.02), rgba(2,6,23,0.00));
            }}
            .block-container {{
                padding-top: 1.25rem !important;
                padding-bottom: 2.5rem !important;
                max-width: 1200px;
            }}

            footer {{ visibility: hidden; }}

            .ta-card {{
                border: 1px solid rgba(15, 23, 42, 0.08);
                background: rgba(255,255,255,0.72);
                backdrop-filter: blur(10px);
                border-radius: 16px;
                padding: 16px 16px;
                box-shadow: 0 12px 30px rgba(2, 6, 23, 0.08);
            }}
            .ta-muted {{ color: rgba(15, 23, 42, 0.65); }}
            .ta-title {{
                font-weight: 800;
                letter-spacing: -0.02em;
                margin: 0 0 2px 0;
            }}
            .ta-badge {{
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 6px 10px;
                border-radius: 999px;
                border: 1px solid rgba(34,197,94,0.25);
                background: rgba(34,197,94,0.10);
                font-size: 12px;
            }}

            .stButton > button {{
                border-radius: 12px !important;
                border: 1px solid rgba(15, 23, 42, 0.12) !important;
                padding: 0.60rem 0.9rem !important;
                box-shadow: 0 8px 18px rgba(2, 6, 23, 0.10) !important;
            }}
            .stButton > button:hover {{
                transform: translateY(-1px);
                transition: 120ms ease;
            }}
            .stTextInput input, .stTextArea textarea, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {{
                border-radius: 12px !important;
            }}

            section[data-testid="stSidebar"] > div {{
                padding-top: 1.0rem;
            }}
            section[data-testid="stSidebar"] .ta-sidebar-card {{
                border: 1px solid rgba(15, 23, 42, 0.10);
                background: rgba(255,255,255,0.70);
                border-radius: 16px;
                padding: 14px 14px;
                box-shadow: 0 10px 24px rgba(2, 6, 23, 0.08);
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def ui_header(lang_text):
    status = st.session_state.get("agent_status", "Idle")
    user = st.session_state.get("username") or "Farmer"
    st.markdown(
        f"""
        <div class="ta-card" style="padding: 18px 18px; margin-bottom: 14px;">
          <div style="display:flex; justify-content:space-between; align-items:center; gap:16px; flex-wrap:wrap;">
            <div>
              <div class="ta-title" style="font-size: 22px;">{t("Agricultural Super AI Agent")}</div>
              <div class="ta-muted" style="font-size: 13px; margin-top: 2px;">
                {t("Professional farm intelligence for crop health, inputs, and nearby services.")}
              </div>
            </div>
            <div style="display:flex; align-items:center; gap:10px;">
              <span class="ta-badge">🟢 {t("Status")}: {t(status)}</span>
              <span class="ta-badge">👤 {t("User")}: {user}</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False, ttl=360000)
def translate_text(text, language):
    if language == "English" or not isinstance(text, str):
        return text

    stripped = text.strip()
    if not stripped:
        return text

    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "content-type": "application/json",
    }
    payload = {
        "source_language_code": "en-IN",
        "target_language_code": LANGUAGE_CODE_MAP.get(language, "en-IN"),
        "speaker_gender": "Male",
        "mode": "formal",
        "model": "mayura:v1",
        "enable_preprocessing": False,
        "numerals_format": "native",
        "input": stripped,
    }
    try:
        response = requests.post(SARVAM_TRANSLATE_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return text

    translated = (
        data.get("translated_text")
        or data.get("translation")
        or data.get("output")
        or data.get("data", {}).get("translated_text")
    )
    return translated if isinstance(translated, str) and translated.strip() else text


def t(text):
    return translate_text(text, st.session_state.language)


def translate_result_data(data, language):
    if language == "English":
        return data
    if isinstance(data, dict):
        return {key: translate_result_data(value, language) for key, value in data.items()}
    if isinstance(data, list):
        return [translate_result_data(item, language) for item in data]
    if isinstance(data, str):
        return translate_text(data, language)
    return data


def queue_task(task_name, prompt, model=REASONING_MODEL):
    st.session_state.task_queue.append({"task": task_name, "prompt": prompt, "model": model})


def run_all_background_tasks():
    while st.session_state.task_queue:
        task = st.session_state.task_queue.pop(0)
        st.session_state.agent_status = f"Running: {task['task']}"
        
        report = call_openrouter(
            [
                {
                    "role": "system",
                    "content": (
                        "You are an agricultural super-agent. "
                        "Give structured operational report with metrics, risk score, timeline, ROI impact."
                    ),
                },
                {"role": "user", "content": task["prompt"]},
            ],
            task["model"],
        )

        st.session_state.reports.insert(
            0,
            {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "title": task["task"],
                "content": report,
            },
        )

    st.session_state.agent_status = "All tasks completed"


def export_chat_to_pdf():
    os.makedirs(EXPORT_DIR, exist_ok=True)
    path = os.path.join(EXPORT_DIR, f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")

    lines = ["AI Agent Chat Export", ""]
    for msg in st.session_state.chat_history:
        lines.append(f"[{msg['time']}] {msg['role'].upper()}: {msg['text']}")

    page_lines = 35
    with PdfPages(path) as pdf:
        for i in range(0, max(len(lines), 1), page_lines):
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.patch.set_facecolor('white')
            text_chunk = "\n".join(lines[i:i + page_lines]) or "No chat messages to export."
            fig.text(0.05, 0.95, text_chunk, va='top', fontsize=9, family='sans-serif', wrap=True)
            plt.axis('off')
            pdf.savefig(fig)
            plt.close(fig)

    return path


def login_block(lang_text):
    if not os.path.exists(USER_DB):
        with open(USER_DB, "w", encoding="utf-8") as file:
            json.dump({}, file)

    with open(USER_DB, "r", encoding="utf-8") as file:
        users = json.load(file)

    if st.session_state.logged_in:
        return

    st.title(lang_text["login"])
    username = st.text_input(lang_text["username"])
    password = st.text_input(lang_text["password"], type="password")

    if st.button("Continue"):
        if username in users and users[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Login successful")
            st.rerun()
        else:
            users[username] = password
            with open(USER_DB, "w", encoding="utf-8") as file:
                json.dump(users, file)
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Account created")
            st.rerun()
    st.stop()


def sidebar_controls(lang_text):
    with st.sidebar:
        st.markdown("<div class='ta-sidebar-card'>", unsafe_allow_html=True)
        st.markdown(f"### {t('Agent Control Panel')}")

        # Language selection with Apply button
        current_lang_idx = list(TRANSLATIONS.keys()).index(st.session_state.language)
        new_lang = st.selectbox(t("Select Language"), list(TRANSLATIONS.keys()), index=current_lang_idx)
        
        if st.button(t("Apply Language")):
            st.session_state.language = new_lang
            st.rerun()

        st.markdown("---")
        st.subheader(t("Cost Estimation"))

        # Inputs
        est_location = st.text_input(t("Location (city/region)"))
        est_crop = st.text_input(t("Crop name"))
        est_acres = st.number_input(t("Total acres"), min_value=0.0, step=0.1)
        est_invested = st.number_input(t("Total invested (₹ or $)"), min_value=0.0, step=100.0)

        if st.button(t("Estimate Cost & Profit")):
            if not (est_location and est_crop and est_acres > 0):
                st.error(t("Please fill all fields correctly."))
            else:
                # build prompt
                cost_prompt = f"""
                Location: {est_location}
                Crop: {est_crop}
                Acres: {est_acres}
                Investment: {est_invested}

                Provide a cost, revenue & profit analysis including:
                1) Current local market price per unit (use web inference)
                2) Expected monthly prices and best months to sell
                3) Estimate total cost, revenue, profit/loss
                4) Travel costs if selling outside local mandi/market
                5) Suggested sale timing and risk factors

                Format as JSON:
                {{
                  "market_price": "...",
                  "price_trend": "...",
                  "best_months": [...],
                  "total_cost": "...",
                  "expected_revenue": "...",
                  "profit_or_loss": "...",
                  "travel_costs": "...",
                  "recommendation": "..."
                }}
                """
                estimation = call_openrouter(
                    [
                        {"role": "system", "content": "You are an agricultural economic analyst."},
                        {"role": "user", "content": cost_prompt},
                    ],
                    REASONING_MODEL
                )
                st.session_state.cost_estimation = estimation

        st.markdown("---")
        st.subheader(t("Quick Agent Actions"))

        selected_action = st.selectbox(t("Select analysis"), list(ACTION_MAP.keys()))
        col1, col2 = st.columns(2)
        with col1:
            if st.button(t("Run analysis")):
                queue_task(selected_action, ACTION_MAP[selected_action])
                st.success(f"{t('Queued')}: {t(selected_action)}")
        
        with col2:
            if st.button(t("Do all analysis")):
                for action, prompt in ACTION_MAP.items():
                    queue_task(action, prompt)
                st.success(t("All analyses queued!"))

        if st.button(t("Run all core layers")):
            for layer_task in [
                "Vision Layer", "Climate Layer", "Soil Layer", "Water Layer", "Market Layer", "Execution Layer"
            ]:
                queue_task(layer_task, f"Generate operational report for {layer_task} with metrics and actions.")
            st.success(t("All layer analyses queued."))

        st.markdown("---")
        st.subheader(t("Chat Export"))
        if st.button(t("Export chat as PDF")):
            pdf_path = export_chat_to_pdf()
            st.success(f"{t('Saved')}: {pdf_path}")

        st.markdown("---")
        st.subheader(t("User"))
        st.image(st.session_state.photo_url, width=70)
        st.write(st.session_state.username)

        with st.expander(t("Profile menu")):
            if st.button(t("Settings")):
                st.info(t("Logout and more options are available below"))
            st.write(t("Logout"))
            st.write(t("More"))
            if st.button(t("Logout")):
                st.session_state.logged_in = False
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


def home_page(lang_text):
    st.title(t("Agricultural Super AI Agent"))
    st.session_state.location = st.text_input(t("Farm location"), value=st.session_state.location)
    location = st.session_state.location # Local reference
    uploaded_image = st.file_uploader(lang_text["upload"], type=["jpg", "jpeg", "png"])

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Leaf", use_container_width=True)
        
        if st.button(lang_text["analyze"]):
            # Fast path:
            # - preprocess image to reduce upload size (huge speedup)
            # - cache results for the same image+location
            status = st.status(t("Analyzing..."), expanded=False)
            status.update(label=t("Optimizing image for fast analysis..."), state="running")
            img_bytes = _preprocess_image_for_api(image)

            status.update(label=t("Running model..."), state="running")
            result = analyze_leaf_cached(img_bytes, location)
            status.update(label=t("Done"), state="complete")
            st.session_state.detection_result = result
            
            if "error" not in result:
                st.success(t("Analysis Complete!"))
            else:
                st.error(t(result["error"]))

    if st.session_state.detection_result and "error" not in st.session_state.detection_result:
        res = translate_result_data(st.session_state.detection_result, st.session_state.language)
        render_analysis_report(res, st.session_state.location)


def chat_page():
    st.title(t("Agent Chat"))
    
    # Chat container for scrollable messages
    chat_container = st.container()
    
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg['role']):
                st.markdown(f"*{msg['time']}*")
                st.write(t(msg['text']))

    query = st.chat_input(t("Ask about farming, costs, irrigation, market, disease..."))
    if query:
        st.session_state.chat_history.append(
            {"time": datetime.now().strftime("%H:%M:%S"), "role": "user", "text": query}
        )
        with st.spinner(t("Agent is thinking...")):
            answer = call_openrouter(
                [
                    {"role": "system", "content": "You are a practical agricultural AI agent."},
                    {"role": "user", "content": query},
                ]
            )
        st.session_state.chat_history.append(
            {"time": datetime.now().strftime("%H:%M:%S"), "role": "assistant", "text": answer}
        )
        st.rerun()


def _build_blinking_map(lat, lon, display_name, places, actor):
    """Build a folium map with animated blinking/pulsing markers."""
    is_shop = actor.lower().startswith("shop")
    accent = "#22c55e" if is_shop else "#ef4444"
    accent_glow = "rgba(34,197,94,0.5)" if is_shop else "rgba(239,68,68,0.5)"
    icon_emoji = "🛒" if is_shop else "🩺"

    m = folium.Map(
        location=[lat, lon],
        zoom_start=13,
        tiles="CartoDB positron",
        control_scale=True,
    )

    # Your location — large pulsing blue dot
    home_html = f'''
    <div style="position:relative;width:44px;height:44px;">
      <div style="position:absolute;top:0;left:0;width:44px;height:44px;
           border-radius:50%;background:rgba(59,130,246,0.18);
           animation:homePulse 2s ease-in-out infinite;"></div>
      <div style="position:absolute;top:12px;left:12px;width:20px;height:20px;
           border-radius:50%;background:#3b82f6;border:3px solid #fff;
           box-shadow:0 0 10px rgba(59,130,246,0.7);"></div>
    </div>
    <style>
    @keyframes homePulse {{
      0%,100% {{ transform:scale(1);opacity:0.7; }}
      50% {{ transform:scale(1.5);opacity:0; }}
    }}
    </style>
    '''
    folium.Marker(
        [lat, lon],
        tooltip=f"📍 Your Location",
        popup=f"<b>Your Farm</b><br>{display_name}",
        icon=folium.DivIcon(html=home_html, icon_size=(44, 44), icon_anchor=(22, 22)),
    ).add_to(m)

    # Place markers with blinking animation
    for idx, p in enumerate(places):
        gmaps = f"https://www.google.com/maps?q={p['lat']},{p['lon']}"
        dist = p['distance_km']
        name = p['name']
        addr = p['tags'].get('addr:full') or p['tags'].get('addr:street') or p['tags'].get('addr:city') or ''
        delay = round((idx * 0.15) % 2.0, 2)

        marker_html = f'''
        <div style="position:relative;width:36px;height:36px;">
          <div style="position:absolute;top:0;left:0;width:36px;height:36px;
               border-radius:50%;background:{accent_glow};
               animation:blink{idx} 1.4s ease-in-out infinite {delay}s;"></div>
          <div style="position:absolute;top:8px;left:8px;width:20px;height:20px;
               border-radius:50%;background:{accent};border:2.5px solid #fff;
               box-shadow:0 0 8px {accent_glow};
               display:flex;align-items:center;justify-content:center;
               font-size:11px;color:#fff;font-weight:700;">
          </div>
        </div>
        <style>
        @keyframes blink{idx} {{
          0%,100% {{ transform:scale(1);opacity:0.85; }}
          50% {{ transform:scale(1.8);opacity:0; }}
        }}
        </style>
        '''

        popup_html = f'''
        <div style="font-family:Inter,sans-serif;min-width:180px;">
          <b style="font-size:14px;">{icon_emoji} {name}</b><br>
          <span style="color:#64748b;font-size:12px;">📏 {dist:.1f} km away</span><br>
          {'<span style="color:#64748b;font-size:11px;">📍 ' + addr + '</span><br>' if addr else ''}
          <a href="{gmaps}" target="_blank"
             style="color:{accent};font-weight:600;font-size:12px;text-decoration:none;">
            🗺️ Open in Google Maps →
          </a>
        </div>
        '''
        folium.Marker(
            [p["lat"], p["lon"]],
            tooltip=f"{icon_emoji} {name} — {dist:.1f} km",
            popup=folium.Popup(popup_html, max_width=260),
            icon=folium.DivIcon(html=marker_html, icon_size=(36, 36), icon_anchor=(18, 18)),
        ).add_to(m)

    # Fit map bounds
    if places:
        all_coords = [[lat, lon]] + [[p["lat"], p["lon"]] for p in places]
        m.fit_bounds(all_coords, padding=(40, 40))

    return m


def _render_place_cards(places, actor):
    """Render result cards individually to avoid Streamlit HTML size limits."""
    is_shop = actor.lower().startswith("shop")
    accent = "#22c55e" if is_shop else "#ef4444"
    icon = "🛒" if is_shop else "🩺"

    # Render in rows of 3 columns
    for row_start in range(0, len(places), 3):
        row_places = places[row_start:row_start + 3]
        cols = st.columns(len(row_places))
        for col, p in zip(cols, row_places):
            dist = p["distance_km"]
            name = p["name"]
            addr = p["tags"].get("addr:full") or p["tags"].get("addr:street") or p["tags"].get("addr:city") or ""
            phone = p["tags"].get("phone") or p["tags"].get("contact:phone") or ""
            gmaps = f"https://www.google.com/maps?q={p['lat']},{p['lon']}"
            if dist < 2:
                dist_color = "#22c55e"
                dist_label = "Very Close"
            elif dist < 5:
                dist_color = "#f59e0b"
                dist_label = "Nearby"
            else:
                dist_color = "#ef4444"
                dist_label = "Far"

            card = f'''<div style="border:1px solid rgba(15,23,42,0.08);background:rgba(255,255,255,0.85);
backdrop-filter:blur(8px);border-radius:14px;padding:14px 16px;
box-shadow:0 4px 16px rgba(2,6,23,0.06);position:relative;overflow:hidden;margin-bottom:4px;">
<div style="position:absolute;top:0;left:0;width:4px;height:100%;background:{accent};border-radius:4px 0 0 4px;"></div>
<div style="display:flex;justify-content:space-between;align-items:start;gap:8px;">
<div style="font-weight:700;font-size:14px;color:#0f172a;">{icon} {name}</div>
<span style="display:inline-block;padding:3px 8px;border-radius:99px;font-size:11px;font-weight:600;color:#fff;background:{dist_color};">{dist:.1f} km</span>
</div>
{'<div style="color:#64748b;font-size:12px;margin:6px 0 2px 0;">📍 ' + addr + '</div>' if addr else ''}
{'<div style="color:#0f172a;font-size:12px;">📞 ' + phone + '</div>' if phone else ''}
<div style="margin-top:8px;display:flex;gap:6px;align-items:center;">
<span style="font-size:10px;color:{dist_color};font-weight:600;">{dist_label}</span>
<span style="flex:1;"></span>
<a href="{gmaps}" target="_blank" style="font-size:11px;color:{accent};font-weight:600;text-decoration:none;">Maps →</a>
</div></div>'''
            with col:
                st.markdown(card, unsafe_allow_html=True)


def shop_or_doctors_page(title, actor, lang_text):
    st.title(t(title))
    keyp = f"{actor.lower().replace(' ', '_')}_"
    is_shop = actor.lower().startswith("shop")

    col_in1, col_in2 = st.columns(2)
    with col_in1:
        crop = st.text_input(f"{t(actor)}: {t('Crop name')}", key=f"{keyp}crop")
    with col_in2:
        requirement = st.text_input(f"{t(actor)}: {t('Requirement')}", key=f"{keyp}requirement")

    # ── Nearby (real-time) with blinking map ──
    st.markdown("---")
    accent = "#22c55e" if is_shop else "#ef4444"
    st.markdown(
        f'''
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
          <div style="position:relative;width:14px;height:14px;">
            <div style="position:absolute;width:14px;height:14px;border-radius:50%;
                 background:{accent};animation:liveDot 1.2s ease-in-out infinite;"></div>
            <div style="position:absolute;top:3px;left:3px;width:8px;height:8px;
                 border-radius:50%;background:{accent};"></div>
          </div>
          <span style="font-size:20px;font-weight:800;letter-spacing:-0.02em;">
            {t("Nearby (real-time)")}
          </span>
          <span style="font-size:12px;color:#64748b;background:rgba(15,23,42,0.05);
               padding:3px 10px;border-radius:99px;">LIVE</span>
        </div>
        <style>
        @keyframes liveDot {{
          0%,100% {{ transform:scale(1);opacity:1; }}
          50% {{ transform:scale(2.2);opacity:0; }}
        }}
        </style>
        ''',
        unsafe_allow_html=True,
    )

    radius_km = st.slider(
        t("Search radius (km)"),
        min_value=1, max_value=50, value=15,
        key=f"{keyp}radius_km",
    )
    location = st.session_state.get("location", "").strip()
    if not location:
        st.info(t("Tip: set your Farm location on Home page for nearby results."))

    # Auto-search on button click (fast, cached)
    if st.button(f"🔍 {t('Search nearby on map')}", key=f"{keyp}show_nearby_map", type="primary", use_container_width=True):
        if not location:
            st.error(t("Farm location is empty. Please set it on Home page."))
        else:
            st.session_state[f"{keyp}nearby_search_trigger"] = True

    # Render map if triggered
    if st.session_state.get(f"{keyp}nearby_search_trigger") and location:
        with st.spinner(t("⚡ Locating nearby services...")):
            geo = geocode_location_osm(location)
        if not geo:
            st.error(t("Could not find this location. Try a more specific place."))
        else:
            lat, lon, display_name = geo
            with st.spinner(t("⚡ Fetching real-time data from OpenStreetMap...")):
                places = nearby_osm_places(lat, lon, actor, radius_m=int(radius_km * 1000))

            st.markdown(
                f'<div style="font-size:13px;color:#64748b;margin:4px 0 10px 0;">'
                f'📍 <b>{display_name}</b> — found <b>{len(places)}</b> results within {radius_km} km</div>',
                unsafe_allow_html=True,
            )

            if not places:
                st.warning(t("No nearby places found in OpenStreetMap for this category. Try increasing the radius."))
            else:
                # Render the blinking map
                m = _build_blinking_map(lat, lon, display_name, places, actor)
                st_folium(m, use_container_width=True, height=520, returned_objects=[])

                # Render place cards
                _render_place_cards(places, actor)

    # Tomato fertilizer hint
    if is_shop and crop.strip().lower() in {"tomato", "tomatoes"} and (
        "fert" in requirement.strip().lower() or requirement.strip() == ""
    ):
        with st.expander(t("🍅 Tomato fertilizer requirements"), expanded=False):
            st.markdown(tomato_fertilizer_plan_text())

    # AI Search buttons
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button(f"🤖 {t('AI Search')} {t(actor)}", key=f"{keyp}ai_search", use_container_width=True):
            with st.spinner(f"⚡ {t('Finding the best options for you...')}"):
                loc = st.session_state.get('location', 'unknown location')
                search_prompt = f"As an agricultural AI, find/recommend 5 {actor.lower()}s or services for {crop} with requirement: {requirement} near {loc}. Provide name, contact detail (simulated), and specialized service. Format as a clean list."
                response = call_openrouter([{"role": "user", "content": search_prompt}])
                if "error" not in response.lower() or "401" not in response:
                    st.success(f"{t('Found')} {t(actor)}!")
                    st.markdown(t(response))
                else:
                    st.error(f"{t('Search failed')}: {t(response)}")

    with col2:
        if st.button(f"📋 {t('Show all nearby')}", key=f"{keyp}ai_show_all", use_container_width=True):
            with st.spinner(t("Listing all major options...")):
                search_prompt = f"List all major {actor.lower()} options for {crop} farming. Include pricing estimates and usage summary."
                response = call_openrouter([{"role": "user", "content": search_prompt}])
                st.markdown(t(response))


def contact_page():
    st.title(t("Contact"))
    st.markdown(
        f"""
        **{t('AI Farm Agent Team')}**  
        {t('Email')}: hireom07@gmail.com  
        {t('Services')}: {t('Vision, Climate, Soil, Water, Market, Execution')}  
        """
    )


def show_reports_panel():
    st.markdown(f"## {t('Generated Reports')}")
    if not st.session_state.reports:
        st.info(t("No reports yet. Run analyses from the left panel."))
        return
    for report in st.session_state.reports[:12]:
        with st.expander(f"{report['time']} — {report['title']}"):
            st.write(t(report["content"]))


def main():
    st.set_page_config(
        page_title="Agri Super Agent",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    ensure_session_defaults()
    apply_local_font(st.session_state.language)

    lang_text = TRANSLATIONS[st.session_state.language]
    login_block(lang_text)

    sidebar_controls(lang_text)
    apply_local_font(st.session_state.language)

    run_all_background_tasks()
    ui_header(lang_text)

    tabs = st.tabs(
        [
            f"🏠 {lang_text['home']}",
            f"💬 {lang_text['chat']}",
            f"🛒 {lang_text['shops']}",
            f"🩺 {lang_text['doctors']}",
            f"☎️ {lang_text['contact']}",
        ]
    )

    with tabs[0]:
        home_page(lang_text)
    with tabs[1]:
        chat_page()
    with tabs[2]:
        shop_or_doctors_page("🛒 Fertilizer Shop", "Shop", lang_text)
    with tabs[3]:
        shop_or_doctors_page("🩺 Doctors", "Doctors", lang_text)
    with tabs[4]:
        contact_page()

    if st.session_state.cost_estimation:
        est = st.session_state.cost_estimation
        st.markdown("---")
        st.markdown(f"## {t('Cost and Profit Estimation Report')}")

        # Try JSON parse if AI returned text
        try:
            # Clean output in case of markdown blocks
            if "```json" in est:
                est = est.split("```json")[1].split("```")[0].strip()
            elif "```" in est:
                est = est.split("```")[1].split("```")[0].strip()
            
            est_json = json.loads(est)
        except:
            st.write(t("Could not parse estimation. Raw output:"))
            st.write(t(est))
            est_json = None

        if est_json:
            est_json = translate_result_data(est_json, st.session_state.language)
            st.write(f"### {t('Local Market Price')}")
            st.write(est_json.get("market_price","N/A"))

            st.write(f"### {t('Price Trend and Best Months')}")
            st.write(est_json.get("price_trend",""))
            st.write(f"{t('Best Months to Sell')}: {est_json.get('best_months', [])}")

            st.write(f"### {t('Cost and Revenue Breakdown')}")
            st.write(f"{t('Total Production Cost')}: {est_json.get('total_cost','')}")
            st.write(f"{t('Expected Revenue')}: {est_json.get('expected_revenue','')}")
            st.write(f"{t('Profit or Loss')}: {est_json.get('profit_or_loss','')}")

            st.write(f"### {t('Travel Costs')}")
            st.write(est_json.get("travel_costs",""))

            st.write(f"### {t('Recommendation')}")
            st.info(est_json.get("recommendation",""))

    show_reports_panel()


if __name__ == "__main__":
    main()
