# OSCE App — Single File (Bottom Buttons, No Sticky CSS)
# -----------------------------------------------------------------------------
# Menu → live SP chat (factsheet‑bound) → strict examiner → results.
# What this build changes:
#   • Removes sticky/fixed HTML bar (Streamlit components can’t live inside raw HTML)
#   • Adds a simple bottom control row with two **real Streamlit buttons** placed
#     at the very end of the script so they appear at the bottom and close together
#   • Working navigation helpers: 🏠 Menu → full reset; 🔁 Retry → restart station
#   • Station‑specific rubrics for 4 stations
#   • Programmatic intercept for tests (ECG/CXR/BNP/Trop/LFTs/Serology/CRP/WCC/Exam/Vitals)
#   • Natural SP opening line; never asks student for results
# -----------------------------------------------------------------------------

import os, json, time, re
from typing import Dict, Any, List
from pathlib import Path
from dotenv import load_dotenv
import requests
import streamlit as st

# 1) Load .env explicitly from the script’s directory (not cwd)
ENV_PATH = Path(__file__).resolve().parent / "/Users/waleedal-tahafi/PycharmProjects/PythonProject/.env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

# 2) Also support Streamlit secrets and (optionally) an inline fallback
API_KEY_INLINE = ""  # leave empty in prod
API_KEY = (
    os.getenv("OPENAI_API_KEY")
    or (st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None)
    or API_KEY_INLINE
)

MODEL = "gpt-4o-mini"
BASE_URL = "https://api.openai.com/v1"
STATION_DURATION_MIN = 8

if not API_KEY:
    st.error(
        "No API key found. Set OPENAI_API_KEY in .env (placed next to this .py), "
        "or add it to Streamlit Secrets, or temporarily paste into API_KEY_INLINE."
    )
    st.stop()
# ================== OpenAI-compatible helper ==================

def chat(messages: List[Dict[str, str]], temperature: float = 0.6, timeout: int = 40) -> str:
    url = BASE_URL.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {"model": MODEL, "temperature": temperature, "messages": messages}
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"LLM API error {r.status_code}: {r.text}")
    data = r.json()
    return data["choices"][0]["message"]["content"]

# ================== Scenarios ==================
STATIONS: Dict[str, Dict[str, Any]] = {
    # ---------------- Chest Pain ----------------
    "chest_pain": {
        "title": "Acute Chest Pain (55M)",
        "visible": (
            "A 55-year-old man presents with sudden, crushing central chest pain radiating to the left arm, "
            "with shortness of breath. Take a focused history and move towards initial management."
        ),
        "factsheet": {
            "opening": "The patient sits anxiously, looking at you, waiting for you to introduce yourself.",
            "demographics": {"name": "Mr Ahmed", "age": 55, "sex": "male"},
            "affect": "anxious, sweaty, mildly short of breath; worried about heart attack",
            "pain_history": {
                "site": "central chest",
                "onset": "45 minutes ago (sudden while watching TV)",
                "character": "crushing, heavy",
                "radiation": "left arm",
                "associated": ["short of breath", "sweaty"],
                "severity_0_10": 8,
                "time_course": "constant 45 min",
                "exacerbating": "movement",
                "relieving": "none tried"
            },
            "risk_factors": {"smoking": "20/day", "diabetes": "no", "hypertension": "yes", "family_history": "father MI at 58"},
            "meds_allergies": {"regular": ["none"], "allergies": ["aspirin"], "note": "hives with aspirin"},
            "vitals": {"HR": 104, "BP": "105/68", "RR": 22, "Temp": 37.2, "SpO2": "93% on air"},
            "exam": {"cv": "cool, clammy; heart sounds normal", "resp": "bibasal crackles", "jvp": "not raised"},
            "tests": {"ECG": "ST elevation in II, III, aVF; ST depression in I, aVL", "Troponin": "pending (early)"},
            "social": {"occupation": "taxi driver", "alcohol": "rare", "drugs": "no"}
        }
    },
    # ---------------- HFrEF ----------------
    "sob_hfref": {
        "title": "Shortness of Breath (70M)",
        "visible": (
            "A 70-year-old man presents with 3 days of worsening breathlessness, orthopnoea, and ankle swelling. "
            "Take a focused history and outline initial investigations and management."
        ),
        "factsheet": {
            "opening": "The patient looks tired and sits forward. He waits for you to introduce yourself.",
            "demographics": {"name": "Mr Jones", "age": 70, "sex": "male"},
            "affect": "tired, slightly embarrassed about swollen legs, cooperative",
            "symptoms": ["orthopnoea (3 pillows)", "paroxysmal nocturnal dyspnoea", "ankle swelling", "fatigue", "no chest pain"],
            "context": "Breathless on walking to bathroom; worse at night.",
            "pmh": ["MI 5 years ago", "hypertension"],
            "meds_allergies": {"regular": ["atorvastatin"], "allergies": ["none"]},
            "vitals": {"HR": 98, "BP": "150/90", "RR": 24, "Temp": 37.0, "SpO2": "94% on air"},
            "exam": {"lung": "bibasal crackles", "cv": "displaced apex beat", "oedema": "pitting to knees", "jvp": "raised"},
            "tests": {"CXR": "cardiomegaly with Kerley B lines", "BNP": "elevated"}
        }
    },
    # ---------------- Hepatitis A ----------------
    "hepatitis_a": {
        "title": "Jaundice after Holiday (22F)",
        "visible": "A 22-year-old student presents with jaundice and fatigue after recent travel to Mexico.",
        "factsheet": {
            "opening": "The patient looks tired and yellow-tinged, waiting for you to introduce yourself.",
            "demographics": {"name": "Sophie", "age": 22, "sex": "female"},
            "affect": "tired, mild RUQ discomfort",
            "symptoms": {"onset": "1 week", "fatigue": "severe", "jaundice": "yellow eyes/skin", "appetite": "poor", "abdominal_pain": "RUQ mild"},
            "risk": {"travel": "Mexico 3 weeks ago", "food": "ate street food", "vaccination": "no Hep A vaccine"},
            "vitals": {"HR": 90, "BP": "112/70", "RR": 16, "Temp": 37.8, "SpO2": "98%"},
            "exam": {"abdomen": "mild RUQ tenderness, no guarding, no peritonism"},
            "tests": {"LFTs": "ALT 950, AST 870, ALP 200, Bilirubin 120", "Hepatitis_serology": "Hep A IgM positive"}
        }
    },
    # ---------------- Pneumonia ----------------
    "pneumonia": {
        "title": "Cough (70M)",
        "visible": "A 70-year-old man presents with fever, productive cough, and breathlessness.",
        "factsheet": {
            "opening": "The patient is coughing and looks unwell, waiting for you to introduce yourself.",
            "demographics": {"name": "Mr Patel", "age": 70, "sex": "male"},
            "affect": "febrile, coughing, breathless",
            "symptoms": {"cough": "productive with green sputum", "breathlessness": "on exertion and now at rest", "chest_pain": "pleuritic left side", "fever": "39.2"},
            "risk": {"smoking": "ex-smoker 40 pack years", "COPD": "yes"},
            "vitals": {"HR": 110, "BP": "100/65", "RR": 28, "Temp": 39.2, "SpO2": "90% on air"},
            "exam": {"resp": "coarse crackles left base, bronchial breathing"},
            "tests": {"CXR": "Left lower lobe consolidation", "CRP": "180 mg/L", "WCC": "15 x10^9/L"}
        }
    }
}

# ================== Station‑Specific Rubrics ==================
RUBRICS: Dict[str, List[Dict[str, Any]]] = {
    "chest_pain": [
        {"id": "intro", "desc": "Introduces self, confirms identity (name/DOB), gains consent", "weight": 1, "critical": True},
        {"id": "open", "desc": "Begins with open question for chest pain", "weight": 1},
        {"id": "socrates", "desc": "SOCRATES chest pain features (site, onset, character, radiation, associated, timing, exacerbating/relieving, severity)", "weight": 2},
        {"id": "risk", "desc": "Cardiac risks (smoking, HTN, FHx, DM)", "weight": 1},
        {"id": "investigate", "desc": "Orders ECG, troponin, sats/monitoring; considers CXR if needed", "weight": 2},
        {"id": "diagnosis", "desc": "Identifies inferior STEMI from ECG leads II, III, aVF", "weight": 1},
        {"id": "management", "desc": "Time‑critical: PCI referral; DAPT with aspirin‑allergy handling; oxygen if hypoxic", "weight": 2},
        {"id": "consent", "desc": "Explains PCI risks/benefits and confirms consent", "weight": 1, "critical": True},
        {"id": "secondary", "desc": "Secondary prevention: statin, ACEi, beta‑blocker, rehab, smoking cessation", "weight": 1},
    ],
    "sob_hfref": [
        {"id": "intro", "desc": "Introduces self, confirms identity (name/DOB), gains consent", "weight": 1, "critical": True},
        {"id": "open", "desc": "Open question for breathlessness", "weight": 1},
        {"id": "severity", "desc": "Orthopnoea/PND, exercise tolerance; fluid overload signs (oedema, JVP)", "weight": 2},
        {"id": "risk", "desc": "Hx MI/HTN, meds, salt/fluid, adherence", "weight": 1},
        {"id": "investigate", "desc": "Orders CXR, BNP, sats/monitoring; considers ECG/echo pathway", "weight": 2},
        {"id": "diagnosis", "desc": "Likely decompensated HFrEF", "weight": 1},
        {"id": "management", "desc": "Oxygen if hypoxic; IV diuretics; optimise HF meds; fluid balance", "weight": 2},
        {"id": "consent", "desc": "Explains interventions/risks and confirms consent", "weight": 1, "critical": True},
        {"id": "safety_net", "desc": "Follow‑up with HF nurse; daily weights; salt restriction", "weight": 1},
    ],
    "hepatitis_a": [
        {"id": "intro", "desc": "Introduces self, confirms identity (name/DOB), gains consent", "weight": 1, "critical": True},
        {"id": "open", "desc": "Open question for jaundice/fatigue", "weight": 1},
        {"id": "exposures", "desc": "Travel/food/water, vaccination, contacts, onset of jaundice", "weight": 2},
        {"id": "investigate", "desc": "Orders LFTs, Hep A serology; baseline INR; trend ALT/AST", "weight": 2},
        {"id": "diagnosis", "desc": "Explains likely acute Hepatitis A (IgM+), red flags for liver failure", "weight": 1},
        {"id": "public_health", "desc": "Hygiene advice, isolation while infectious, public‑health notification", "weight": 2, "critical": True},
        {"id": "safety_net", "desc": "Avoid alcohol/paracetamol; return if confusion/bleeding; follow‑up LFTs", "weight": 1},
    ],
    "pneumonia": [
        {"id": "intro", "desc": "Introduces self, confirms identity (name/DOB), gains consent", "weight": 1, "critical": True},
        {"id": "open", "desc": "Open question for cough/breathlessness", "weight": 1},
        {"id": "severity", "desc": "CURB‑65 components; comorbid COPD; hydration/nutrition", "weight": 2},
        {"id": "investigate", "desc": "Orders CXR (critical for dx), CRP, WCC, sats/ABG if hypoxic", "weight": 2, "critical": True},
        {"id": "diagnosis", "desc": "States likely CAP with LLL consolidation", "weight": 1},
        {"id": "management", "desc": "Appropriate O2 target; empiric antibiotics per guideline; fluids/antipyretics", "weight": 2},
        {"id": "safety_net", "desc": "Review 48–72h or earlier if worse; smoking cessation", "weight": 1},
    ],
}

TOTAL_SCORES = {sid: sum(i["weight"] for i in rub) for sid, rub in RUBRICS.items()}
BORDERLINE_POINTS = 7

CRITICAL_FAIL_PHRASES = [
    "go home", "no tests needed", "probably heartburn", "wait and see", "review tomorrow", "repeat in an hour"
]

# ================== SP Roleplay Engine ==================
SP_POLICY = (
    "You are a standardized patient (SP) in an OSCE. Stay strictly in role.\n"
    "OPENING RULE: On the very first turn, output ONLY the factsheet['opening']. No symptoms yet.\n"
    "BEHAVIOUR: Keep replies natural and concise (1–3 sentences; up to 4 if asked 'tell me more').\n"
    "Only reveal information in the factsheet (or plain-language paraphrases).\n"
    "If the student requests vitals/exam/ECG/CXR/BNP/Troponin/LFTs/Hepatitis serology/CRP/WCC, provide the exact values immediately in one compact reply.\n"
    "If asked for something not present, say you don't know.\n"
    "Never give diagnoses or management advice—you are the patient.\n"
    "Mirror the 'affect' lightly. If the student uses unexplained jargon, ask briefly what it means."
)

SP_FEW_SHOTS = [
    {"role": "user", "content": "Hello, I’m a medical student. Can I confirm your name and date of birth?"},
    {"role": "assistant", "content": "I’m Mr Ahmed, 55. Yes, that’s right."},
    {"role": "user", "content": "What brought you in today?"},
    {"role": "assistant", "content": "There’s a heavy pain in the middle of my chest and it shoots down my left arm. It came on suddenly about 45 minutes ago."},
    {"role": "user", "content": "Okay, I’d like your observations and an ECG please."},
    {"role": "assistant", "content": "My observations are HR 104, BP 105/68, RR 22, Temp 37.2, SpO2 93%. The ECG shows ST elevation in II, III and aVF."}
]


def make_sp_messages(factsheet: Dict[str, Any], transcript: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SP_POLICY},
        {"role": "system", "content": "CASE FACTSHEET:\n" + json.dumps(factsheet, indent=2)},
        {"role": "system", "content": "For the very first assistant turn: output ONLY the opening from the factsheet."},
        *SP_FEW_SHOTS,
        *transcript,
    ]

# ============== Programmatic, guaranteed results on request ===============
VITALS_KEYS = ["HR", "BP", "RR", "Temp", "SpO2"]

REQUEST_PATTERNS = {
    "vitals": re.compile(r"\b(vital|obs|observations|bp|blood pressure|hr|heart rate|rr|resp(irat|)ory rate|sats?|oxygen sat|temperature|temp|monitor(ing)?)\b", re.I),
    "ecg": re.compile(r"\b(ecg|electro\w*gram)\b", re.I),
    "cxr": re.compile(r"\b(cxr|chest\s*x-?ray|x-?ray)\b", re.I),
    "bnp": re.compile(r"\b(bnp|nt\s*-?pro\s*bnp)\b", re.I),
    "troponin": re.compile(r"\b(trop(onin)?)\b", re.I),
    "lfts": re.compile(r"\b(lfts?|liver function tests?|transaminases|alt|ast|bilirubin|alkaline phosphatase|alp)\b", re.I),
    "hep_serol": re.compile(r"\b(hep(atitis)?\s*a\s*igm|hepatitis serology|viral\s*serolog(y|ies))\b", re.I),
    "crp": re.compile(r"\b(crp)\b", re.I),
    "wcc": re.compile(r"\b(wcc|white cell count|fbc|cbc)\b", re.I),
    "exam": re.compile(r"\b(exam(ination)?|examine|ausculta|palpate|have a look|physical)\b", re.I),
}


def detect_requests(text: str) -> Dict[str, bool]:
    t = text.lower()
    return {k: bool(p.search(t)) for k, p in REQUEST_PATTERNS.items()}


def compose_results_reply(station: Dict[str, Any], req: Dict[str, bool]) -> str:
    fs = station["factsheet"]
    parts = []
    if req.get("vitals") and fs.get("vitals"):
        vitals = ", ".join([f"{k} {fs['vitals'].get(k)}" for k in VITALS_KEYS if k in fs["vitals"]])
        parts.append(f"My observations are: {vitals}.")
    tests = fs.get("tests", {})
    if req.get("ecg") and tests.get("ECG"):
        parts.append(f"The ECG shows: {tests['ECG']}.")
    if req.get("cxr") and tests.get("CXR"):
        parts.append(f"The chest X-ray shows: {tests['CXR']}.")
    if req.get("bnp") and tests.get("BNP"):
        parts.append(f"The BNP is: {tests['BNP']}.")
    if req.get("troponin") and tests.get("Troponin"):
        parts.append(f"Troponin: {tests['Troponin']}.")
    if req.get("lfts") and tests.get("LFTs"):
        parts.append(f"LFTs: {tests['LFTs']}.")
    if req.get("hep_serol") and tests.get("Hepatitis_serology"):
        parts.append(f"Hepatitis serology: {tests['Hepatitis_serology']}.")
    if req.get("crp") and tests.get("CRP"):
        parts.append(f"CRP: {tests['CRP']}.")
    if req.get("wcc") and tests.get("WCC"):
        parts.append(f"WCC: {tests['WCC']}.")
    if req.get("exam") and fs.get("exam"):
        human = "; ".join([f"{k.upper()}: {v}" for k, v in fs["exam"].items()])
        parts.append(f"On examination you find: {human}.")
    return " ".join(parts)

# ================== Examiner ==================
EXAMINER_SYSTEM = (
    "You are an OSCE examiner. Grade STRICTLY using the station-specific rubric.\n"
    "Return JSON ONLY with keys: awarded:[{id,evidence}], missed:[{id,expected}], critical_fail:{triggered,reason}, rationale, next_steps.\n"
    "Award an item ONLY if you quote exact learner evidence (a verbatim substring from the transcript)."
)

EXAMINER_USER_TEMPLATE = (
    "STATION: {title}\nCASE (visible to candidate): {case}\n\nRUBRIC: {rubric}\n\nTRANSCRIPT (role‑tagged):\n{transcript}\n\nReturn JSON ONLY."
)


def role_tagged(messages: List[Dict[str, str]]) -> str:
    lines = []
    for m in messages:
        role = "STUDENT" if m["role"] == "user" else "PATIENT"
        lines.append(f"[{role}] {m['content']}")
    return "\n".join(lines)


def examiner_grade(station_id: str, station: Dict[str, Any], transcript: str) -> Dict[str, Any]:
    rubric = RUBRICS[station_id]
    total_score = TOTAL_SCORES[station_id]
    rubric_compact = [{k: r[k] for k in ("id", "desc", "weight")} for r in rubric]
    prompt = [
        {"role": "system", "content": EXAMINER_SYSTEM},
        {"role": "user", "content": EXAMINER_USER_TEMPLATE.format(title=station["title"], case=station["visible"], rubric=json.dumps(rubric_compact), transcript=transcript)},
        {"role": "system", "content": "Return ONLY JSON. No prose. No code fences."}
    ]
    raw = chat(prompt, temperature=0.1)
    s = raw.find("{"); e = raw.rfind("}")
    if s != -1 and e != -1:
        try:
            data = json.loads(raw[s:e+1])
        except Exception:
            data = {"awarded": [], "missed": [], "critical_fail": {"triggered": False, "reason": None}, "rationale": "", "next_steps": []}
    else:
        data = {"awarded": [], "missed": [], "critical_fail": {"triggered": False, "reason": None}, "rationale": "", "next_steps": []}

    # Local validation
    awarded = []
    awarded_ids = set()
    for a in data.get("awarded", []):
        iid = a.get("id"); ev = (a.get("evidence") or "").strip()
        if iid and ev and ev.lower() in transcript.lower() and any(r["id"] == iid for r in rubric):
            awarded.append({"id": iid, "evidence": ev}); awarded_ids.add(iid)

    # Critical fail text scan
    cf_triggered = data.get("critical_fail", {}).get("triggered", False)
    cf_reason = data.get("critical_fail", {}).get("reason")
    low_t = transcript.lower()
    if not cf_triggered:
        for p in CRITICAL_FAIL_PHRASES:
            if p in low_t:
                cf_triggered = True; cf_reason = "Unsafe reassurance/delay"; break

    # Compute points & band
    points = 0; critical_missed = False; missed_items = []
    for rbi in rubric:
        if rbi["id"] in awarded_ids:
            points += rbi["weight"]
        else:
            missed_items.append({"id": rbi["id"], "expected": rbi["desc"]})
            if rbi.get("critical"): critical_missed = True

    if cf_triggered or critical_missed:
        band = "Fail"
    elif points == total_score:
        band = "Pass"
    elif points >= BORDERLINE_POINTS:
        band = "Borderline"
    else:
        band = "Fail"

    return {
        "total_points": total_score,
        "points": points,
        "band": band,
        "awarded": awarded,
        "missed": missed_items,
        "critical_fail": {"triggered": cf_triggered, "reason": cf_reason},
        "rationale": data.get("rationale", ""),
        "next_steps": data.get("next_steps", [])
    }

# ================== Rerun + Navigation Helpers ==================

def _rerun():
    """Version‑safe rerun wrapper."""
    try:
        st.rerun()  # modern
    except Exception:
        st.experimental_rerun()  # legacy


def go_menu():
    """Hard‑reset session and boot into menu."""
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    # Seed minimal state
    st.session_state.phase = "menu"
    st.session_state.station_id = None
    st.session_state.messages = []
    st.session_state.start_time = None
    _rerun()


def retry_station():
    """Restart current station from its opening line."""
    sid = st.session_state.get("station_id")
    if not sid:
        go_menu()
        return
    S = STATIONS[sid]
    st.session_state.phase = "station"
    st.session_state.messages = [{"role": "assistant", "content": S["factsheet"]["opening"]}]
    st.session_state.start_time = time.time()
    _rerun()

# ================== App State (defaults) ==================
if "phase" not in st.session_state: st.session_state.phase = "menu"
if "messages" not in st.session_state: st.session_state.messages = []
if "station_id" not in st.session_state: st.session_state.station_id = None
if "start_time" not in st.session_state: st.session_state.start_time = None

# ================== UI: Menu ==================
if st.session_state.phase == "menu":
    st.title("🧪 Wmed OSCE — Choose a Scenario")
    st.caption("Click a station to begin. The patient will wait for your introduction.")

    cols = st.columns(2)
    for i, (sid, S) in enumerate(STATIONS.items()):
        with cols[i % 2]:
            st.subheader(S["title"])
            st.write(S["visible"])
            if st.button(f"Start: {S['title']}", key=f"start_{sid}"):
                st.session_state.station_id = sid
                st.session_state.phase = "station"
                st.session_state.messages = []
                st.session_state.start_time = time.time()
                st.session_state.messages.append({"role": "assistant", "content": S["factsheet"]["opening"]})
                _rerun()

# ================== UI: Station ==================
if st.session_state.phase == "station":
    S = STATIONS[st.session_state.station_id]
    sid = st.session_state.station_id
    st.title(S["title"])
    st.info(S["visible"])

    # Timer
    DURATION_SEC = STATION_DURATION_MIN * 60
    elapsed = int(time.time() - st.session_state.start_time)
    remaining = max(0, DURATION_SEC - elapsed)
    m, s = divmod(remaining, 60)
    st.metric("Time Remaining", f"{m:02d}:{s:02d}")

    # Transcript
    for mobj in st.session_state.messages:
        with st.chat_message("assistant" if mobj["role"] == "assistant" else "user"):
            st.markdown(mobj["content"])

    # Input
    if remaining > 0:
        user_input = st.chat_input("Speak to the patient… ask history, request vitals/ECG/CXR/LFTs/BNP, explain plan…")
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Guaranteed results intercept
            req = detect_requests(user_input)
            auto = compose_results_reply(S, req)
            if auto:
                st.session_state.messages.append({"role": "assistant", "content": auto})
            else:
                try:
                    sp_reply = chat(make_sp_messages(S["factsheet"], st.session_state.messages), temperature=0.7)
                except Exception as e:
                    sp_reply = f"[SP error: {e}]"
                st.session_state.messages.append({"role": "assistant", "content": sp_reply})
            _rerun()
    else:
        st.warning("⏰ Time is up. Please end the station and grade.")

    st.markdown("---")
    if st.button("End Station & Grade ▶", key="btn_end_grade"):
        tr = role_tagged(st.session_state.messages)
        try:
            result = examiner_grade(sid, S, tr)
        except Exception as e:
            st.error(f"Grading failed: {e}")
            st.stop()

        # Results UI
        st.success("Marking complete")
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Total Points", f"{result['total_points']}")
        with c2: st.metric("Awarded", f"{result['points']}")
        with c3: st.metric("Outcome", result['band'])

        if result["critical_fail"]["triggered"]:
            st.error(f"CRITICAL FAIL: {result['critical_fail']['reason']}")

        st.markdown("### ✅ Awarded (with evidence)")
        if result.get("awarded"):
            for it in result["awarded"]:
                st.write(f"**{it['id']}**")
                st.code(it["evidence"], language="text")
        else:
            st.caption("No awarded items.")

        st.markdown("### ❌ Missed Items")
        if result.get("missed"):
            for it in result["missed"]:
                st.write(f"**{it['id']}** — {it['expected']}")
        else:
            st.caption("None.")

        if result.get("rationale"):
            st.markdown("### 🧠 Examiner Rationale")
            st.write(result["rationale"])

        if result.get("next_steps"):
            st.markdown("### 📚 Next Steps")
            if isinstance(result["next_steps"], list):
                st.write("\n\n".join(result["next_steps"]))
            else:
                st.write(result["next_steps"])

# ================== Bottom Control Row (real Streamlit buttons) ==================
# Always render at the end so it sits visually at the bottom.
st.markdown("---")
ctrl_cols = st.columns([1, 1, 8])
with ctrl_cols[0]:
    if st.button("🏠 Menu", key="ctrl_menu_bottom"):
        go_menu()
with ctrl_cols[1]:
    if st.button("🔁 Retry", key="ctrl_retry_bottom"):
        retry_station()

st.caption("Educational use only. Natural SP follows factsheet; examiner grades via station‑specific rubrics. Bottom buttons are native Streamlit components (no HTML), so they always render and work.")
