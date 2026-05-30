import streamlit as st
import requests
import os
import base64
import datetime

BACKEND_URL = "https://ich-dx-deploy-1.onrender.com/predict"
LOGO_PATH = "logo.png"
st.set_page_config(page_title="ICH-DX · Classifier", layout="centered", page_icon="🧠")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Space+Mono:wght@400;700&display=swap');

/* ── FULL PAGE: pure black base + deep teal radial glow ── */
html, body { background-color: #000000 !important; }
.stApp     { background-color: #000000 !important; }

/* Radial glow on the page itself */
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 70% 55% at 50% 0%,   rgba(0,160,220,0.13) 0%, transparent 70%),
        radial-gradient(ellipse 50% 35% at 50% 100%, rgba(0,80,160,0.08)  0%, transparent 70%),
        #000000 !important;
}

/* Nuke every container background */
[data-testid="stHeader"],
[data-testid="stBottom"],
[data-testid="stDecoration"],
.main, .main > div,
.block-container,
[data-testid="stVerticalBlock"],
[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="column"],
.element-container,
div[class*="stMarkdown"] {
    background: transparent !important;
    background-color: transparent !important;
}

/* Hide chrome */
#MainMenu, footer,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display:none !important; }

.block-container {
    max-width: 720px !important;
    margin: 0 auto !important;
    padding: 0 2rem 5rem !important;
}

/* ── ANIMATED GRID — brighter lines ── */
.bg-grid {
    position: fixed; top:0; left:0; width:100%; height:100%;
    pointer-events: none; z-index: 0;
    background-image:
        linear-gradient(rgba(0,200,255,0.055) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,200,255,0.055) 1px, transparent 1px);
    background-size: 52px 52px;
    animation: grid-drift 25s linear infinite;
}
@keyframes grid-drift {
    from { background-position: 0 0; }
    to   { background-position: 52px 52px; }
}

/* ── HERO ── */
.hero-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    padding: 2.8rem 0 1.2rem;
    position: relative;
    z-index: 1;
    animation: fade-up 0.7s cubic-bezier(.22,1,.36,1) both;
}
@keyframes fade-up {
    from { opacity:0; transform:translateY(20px); }
    to   { opacity:1; transform:translateY(0); }
}

.logo-ring {
    width: 192px; height: 192px;
    border-radius: 50%;
    border: 1.5px solid rgba(0,200,255,0.28);
    box-shadow: 0 0 0 8px rgba(0,200,255,0.06), 0 0 55px rgba(0,130,255,0.22);
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 1.4rem;
    flex-shrink: 0;
    animation: pulse-ring 3.5s ease-in-out infinite;
}
@keyframes pulse-ring {
    0%,100% { box-shadow: 0 0 0 8px rgba(0,200,255,0.06), 0 0 55px rgba(0,130,255,0.22); }
    50%      { box-shadow: 0 0 0 14px rgba(0,200,255,0.11), 0 0 80px rgba(0,130,255,0.42); }
}
.logo-ring img {
    width: 160px !important; height: 160px !important;
    object-fit: contain; border-radius: 50%; display: block;
}

/* Badge */
.hero-badge {
    display: inline-flex;
    align-items: center;
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.22em;
    color: #00DDFF;
    background: rgba(0,200,255,0.09);
    border: 1px solid rgba(0,200,255,0.25);
    border-radius: 999px;
    padding: 5px 18px;
    margin-bottom: 0.9rem;
}

/* Title — full brightness white */
.hero-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 3.8rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    line-height: 1;
    margin: 0 0 0.55rem;
    color: #FFFFFF;
    text-shadow:
        0 0 40px rgba(0,180,255,0.5),
        0 0 80px rgba(0,100,220,0.25);
    white-space: nowrap;
}
.hero-title .accent { color: #00DDFF; }

/* Subtitle — clearly readable now */
.hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    color: #4A8FA8;
    text-transform: uppercase;
    white-space: nowrap;
    margin: 0;
}

/* ── SCAN DIVIDER ── */
.scan-divider {
    width: 100%;
    height: 1px;
    margin: 2rem 0 1.6rem;
    background: linear-gradient(90deg,
        transparent 0%, #0A4060 15%, #00CCFF 50%, #0A4060 85%, transparent 100%);
    position: relative; z-index: 1;
    animation: scan-pulse 3s ease-in-out infinite;
}
@keyframes scan-pulse {
    0%,100% { opacity: 0.5; }
    50%      { opacity: 1; }
}

/* ── SECTION LABEL ── */
.sec-label {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.63rem !important;
    letter-spacing: 0.2em !important;
    color: #3A7890 !important;
    text-transform: uppercase !important;
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    margin-bottom: 0.7rem !important;
    position: relative; z-index: 1;
}
.sec-label::before {
    content: '';
    display: inline-block;
    width: 18px; height: 1px;
    background: #00CCFF; flex-shrink: 0;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"],
[data-testid="stFileUploader"] > div,
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploaderDropzone"] > div,
section[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    background-color: transparent !important;
}
[data-testid="stFileUploader"] {
    border: 1px solid #0E3045 !important;
    border-radius: 12px !important;
    padding: 0.2rem 0.8rem !important;
    background: rgba(0,20,35,0.7) !important;
    position: relative; z-index: 1;
    transition: border-color 0.3s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(0,200,255,0.35) !important;
}
[data-testid="stFileUploaderDropzone"] {
    border: none !important;
    background: transparent !important;
    padding: 0.8rem !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] p,
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small {
    color: #3A7890 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.74rem !important;
}
[data-testid="stFileUploader"] button {
    background: rgba(0,200,255,0.07) !important;
    color: #00BBDD !important;
    border: 1px solid rgba(0,200,255,0.22) !important;
    border-radius: 6px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.08em !important;
}
[data-testid="stFileUploader"] button:hover {
    background: rgba(0,200,255,0.14) !important;
    color: #00DDFF !important;
}
/* Uploaded file name chip */
[data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] p {
    color: #60A8C0 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
}

/* ── BUTTONS ── */
[data-testid="stHorizontalBlock"] .stButton > button {
    width: 100% !important;
    padding: 0.75rem 1.2rem !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    transition: all 0.22s ease !important;
    white-space: nowrap !important;
}
/* Run Analysis */
[data-testid="stHorizontalBlock"] .stButton:nth-child(2) > button {
    background: linear-gradient(135deg, #002D6B 0%, #0050B8 100%) !important;
    border: 1px solid #0075E0 !important;
    color: #80C8FF !important;
    box-shadow: 0 2px 20px rgba(0,90,210,0.4) !important;
}
[data-testid="stHorizontalBlock"] .stButton:nth-child(2) > button:hover {
    background: linear-gradient(135deg, #004898 0%, #0080FF 100%) !important;
    box-shadow: 0 5px 30px rgba(0,130,255,0.6) !important;
    transform: translateY(-2px) !important;
    color: #FFFFFF !important;
}
/* Cancel */
[data-testid="stHorizontalBlock"] .stButton:nth-child(3) > button {
    background: transparent !important;
    border: 1px solid #0E3045 !important;
    color: #3A7890 !important;
}
[data-testid="stHorizontalBlock"] .stButton:nth-child(3) > button:hover {
    border-color: rgba(255,60,80,0.45) !important;
    color: #FF6080 !important;
    background: rgba(255,60,80,0.06) !important;
    transform: translateY(-1px) !important;
}

/* ── ALERTS ── */
[data-testid="stAlert"] {
    background: rgba(0,15,28,0.85) !important;
    border-radius: 8px !important;
    position: relative; z-index: 1;
}
[data-testid="stAlert"] p {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.74rem !important;
    color: #90B8C8 !important;
}

/* ── RESULT CARD ── */
.result-card {
    background: rgba(0,12,22,0.9);
    border: 1px solid #0E3045;
    border-radius: 14px;
    overflow: hidden;
    margin-top: 1.6rem;
    position: relative; z-index: 1;
    animation: fade-up 0.5s cubic-bezier(.22,1,.36,1) both;
}
.result-card::before {
    content: '';
    position: absolute;
    top:0; left:0; right:0; height:2px;
    background: linear-gradient(90deg, #003890, #00CCFF, #00E5A0);
}
.rc-header {
    padding: 0.9rem 1.6rem;
    border-bottom: 1px solid #0E3045;
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: rgba(0,8,16,0.6);
}
.rc-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.95rem; font-weight: 700;
    letter-spacing: 0.15em;
    color: #00DDFF;
    text-transform: uppercase;
}
.rc-id {
    font-family: 'Space Mono', monospace;
    font-size: 0.58rem; color: #1A4A60;
    letter-spacing: 0.08em;
}
.rc-row {
    display: flex;
    padding: 0.9rem 1.6rem;
    border-bottom: 1px solid rgba(14,48,69,0.8);
    gap: 1.2rem;
    align-items: flex-start;
}
.rc-row:last-child { border-bottom: none; }
.rc-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem; letter-spacing: 0.14em;
    color: #2A6070;
    text-transform: uppercase;
    min-width: 160px; padding-top: 4px; flex-shrink: 0;
}
.rc-value {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.05rem; font-weight: 600;
    color: #D8EEFF;
    letter-spacing: 0.02em;
    flex: 1; line-height: 1.4;
}
.sev-badge {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem; letter-spacing: 0.12em;
    padding: 4px 14px; border-radius: 999px;
    text-transform: uppercase;
}
.sev-high   { background:rgba(255,64,96,0.1);  color:#FF4060; border:1px solid rgba(255,64,96,0.3); }
.sev-medium { background:rgba(255,193,64,0.1); color:#FFC140; border:1px solid rgba(255,193,64,0.3); }
.sev-low    { background:rgba(0,229,160,0.1);  color:#00E5A0; border:1px solid rgba(0,229,160,0.3); }

[data-testid="stImage"] img {
    border-radius: 10px !important;
    border: 1px solid #0E3045 !important;
    display: block !important; margin: 0 auto !important;
}
[data-testid="stSpinner"] > div { border-top-color: #00DDFF !important; }

/* Body text bump — readable on black */
p, span, li { color: #7AAABB !important; }
</style>
<div class="bg-grid"></div>
""", unsafe_allow_html=True)


# ── HERO ─────────────────────────────────────────────────────────────────────
if os.path.exists(LOGO_PATH):
    with open(LOGO_PATH, "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode()
    ext  = LOGO_PATH.rsplit(".", 1)[-1].lower()
    mime = "image/png" if ext == "png" else "image/jpeg"
    logo_tag = f'<img src="data:{mime};base64,{logo_b64}" alt="ICH-DX Logo"/>'
else:
    logo_tag = '<div style="width:160px;height:160px;border-radius:50%;background:linear-gradient(135deg,#041828,#082840);display:flex;align-items:center;justify-content:center;font-size:72px;">🧠</div>'

st.markdown(f"""
<div class="hero-wrap">
  <div class="logo-ring">{logo_tag}</div>
  <div class="hero-badge">NEURO &nbsp;·&nbsp; AI &nbsp;·&nbsp; v2.0</div>
  <h1 class="hero-title">ICH<span class="accent">-DX</span></h1>
  <p class="hero-sub">Intracranial&nbsp;Hemorrhage &nbsp;·&nbsp; DICOM&nbsp;Analysis &nbsp;·&nbsp; AI&nbsp;Detection</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="scan-divider"></div>', unsafe_allow_html=True)

# ── UPLOAD ───────────────────────────────────────────────────────────────────
st.markdown('<p class="sec-label">Upload DICOM Scan</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["dcm"], label_visibility="collapsed")
st.markdown("<br/>", unsafe_allow_html=True)

# ── BUTTONS ──────────────────────────────────────────────────────────────────
_, col1, col2, _ = st.columns([1, 2.2, 1.4, 1])
with col1:
    classify_btn = st.button("▶  Run Analysis", use_container_width=True)
with col2:
    cancel_btn   = st.button("✕  Cancel", use_container_width=True)
st.markdown("<br/>", unsafe_allow_html=True)

# ── LOGIC ────────────────────────────────────────────────────────────────────
if classify_btn and uploaded_file is not None:
    with st.spinner("Analysing DICOM scan…"):
        with open("temp.dcm", "wb") as f:
            f.write(uploaded_file.read())
        with open("temp.dcm", "rb") as f:
            response = requests.post(BACKEND_URL, files={"file": f}, timeout=120)

    if response.status_code == 200:
        result = response.json()
        st.success("✓  Classification complete")
        st.markdown('<br/><p class="sec-label">Processed Scan</p>', unsafe_allow_html=True)
        st.image(result["image_url"], caption="AI-annotated output", width=720)


        hem  = result.get("Predicted Hemorrhage Type", "—")
        desc = result.get("Description", "—")
        sev  = result.get("Severity Level", "—")
        sugg = result.get("Medical Suggestions", "—")
        sl   = str(sev).lower()
        sev_cls = ("sev-high"   if any(x in sl for x in ["high","severe","critical"]) else
                   "sev-low"    if any(x in sl for x in ["low","mild","minor"])        else
                   "sev-medium")
        ts = datetime.datetime.now().strftime("SCAN-%Y%m%d-%H%M%S")

        st.markdown(f"""<br/>
        <div class="result-card">
          <div class="rc-header">
            <span class="rc-title">Diagnostic Report</span>
            <span class="rc-id">{ts}</span>
          </div>
          <div class="rc-row">
            <span class="rc-label">Hemorrhage Type</span>
            <span class="rc-value">{hem}</span>
          </div>
          <div class="rc-row">
            <span class="rc-label">Description</span>
            <span class="rc-value">{desc}</span>
          </div>
          <div class="rc-row">
            <span class="rc-label">Severity Level</span>
            <span class="rc-value"><span class="sev-badge {sev_cls}">{sev}</span></span>
          </div>
          <div class="rc-row">
            <span class="rc-label">Medical Suggestions</span>
            <span class="rc-value">{sugg}</span>
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.error(f"Backend error: {response.json().get('error', 'Unknown error')}")

elif classify_btn and uploaded_file is None:
    st.error("⚠  Please upload a DICOM (.dcm) file before running analysis.")

if cancel_btn:
    st.warning("Upload cleared — select a new file to begin.")
