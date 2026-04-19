"""
Cotton Guard - AI-Powered Cotton Leaf Disease Detection
Streamlit App with Chatbot Sidebar
=====================================================
Models trained: ResNet50, DenseNet121, EfficientNetB7, ViT-B/16, Swin-T, ConvNeXt-T
Datasets: SAR-CLD 2024 (7 classes), Cotton Leaf Disease (4 classes)

HOW TO RUN:
    pip install streamlit torch torchvision Pillow numpy
    streamlit run app.py

HOW TO ADD YOUR TRAINED MODELS:
    1. Create a folder called 'saved_models/' in the same directory as this file
    2. Place your .pt files inside, e.g.:
       saved_models/
         ├── SAR-CLD_2024/
         │     ├── ResNet50_best.pt
         │     ├── DenseNet121_best.pt
         │     ├── EfficientNetB7_best.pt
         │     ├── ViT_B16_best.pt
         │     ├── Swin_T_best.pt
         │     └── ConvNeXt_T_best.pt
         └── Cotton_Leaf_Disease/
               ├── ResNet50_best.pt
               ├── DenseNet121_best.pt
               └── ... (same structure)
    3. Set USE_REAL_MODEL = True below
    4. Restart the app
"""

import streamlit as st
import numpy as np
import time
from PIL import Image

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
USE_REAL_MODEL = False  # Set True once you place .pt files in saved_models/

SAR_CLD_CLASSES = [
    "Bacterial Blight", "Curl Virus", "Healthy",
    "Herbicide Growth Damage", "Leaf Hopper Jassids",
    "Leaf Reddening", "Leaf Variegation"
]

COTTON_LEAF_CLASSES = [
    "Bacterial Blight", "Curl Virus", "Fusarium Wilt", "Healthy"
]

MODELS_LIST = [
    "ResNet50", "DenseNet121", "EfficientNetB7",
    "ViT-B/16", "Swin-T", "ConvNeXt-T"
]

MODEL_STATS = {
    "ResNet50":       {"params": "23.5M", "flops": "4.1G",  "acc": "97.90%", "color": "#00d2d3"},
    "DenseNet121":    {"params": "7.0M",  "flops": "2.9G",  "acc": "98.36%", "color": "#2ed573"},
    "EfficientNetB7": {"params": "64.1M", "flops": "37.8G", "acc": "96.20%", "color": "#ffa502"},
    "ViT-B/16":       {"params": "85.8M", "flops": "17.6G", "acc": "95.79%", "color": "#a29bfe"},
    "Swin-T":         {"params": "27.5M", "flops": "4.5G",  "acc": "98.36%", "color": "#ff6b81"},
    "ConvNeXt-T":     {"params": "27.8M", "flops": "4.5G",  "acc": "97.66%", "color": "#48dbfb"},
}

DISEASE_INFO = {
    "Bacterial Blight": {
        "severity": "🔴 High",
        "emoji": "🦠",
        "symptoms": "Angular water-soaked lesions on leaves that turn brown/black, vein necrosis, premature defoliation.",
        "cause": "Xanthomonas citri pv. malvacearum — spread by rain splash, wind, contaminated seeds.",
        "treatment": [
            "Spray Copper oxychloride (3g/L) or Streptocycline (0.5g/L)",
            "Remove and destroy infected plant debris",
            "Use certified disease-free seeds",
            "Avoid overhead irrigation during humid weather",
        ],
        "prevention": "Use resistant varieties (e.g., FH-142, MNH-886). Rotate crops every 2–3 years. Treat seeds with Carboxin + Thiram.",
    },
    "Curl Virus": {
        "severity": "🔴 Critical",
        "emoji": "🦟",
        "symptoms": "Upward/downward leaf curling, stunted growth, thickened veins, small & deformed bolls, severe yield loss (up to 80%).",
        "cause": "Cotton Leaf Curl Virus (CLCuV) — transmitted by whitefly (Bemisia tabaci).",
        "treatment": [
            "Spray Imidacloprid (0.5ml/L) or Thiamethoxam against whiteflies",
            "Apply Neem oil (5ml/L) as a repellent",
            "Remove severely infected plants immediately",
            "Install yellow sticky traps (20–25 per acre)",
        ],
        "prevention": "Plant resistant/tolerant varieties. Early sowing (April–May). Avoid ratoon cropping. Monitor whitefly populations weekly.",
    },
    "Healthy": {
        "severity": "🟢 None",
        "emoji": "✅",
        "symptoms": "No disease symptoms. Leaves are green, turgid, and normal-sized.",
        "cause": "N/A — Plant is healthy.",
        "treatment": ["Continue regular crop management practices."],
        "prevention": "Maintain proper spacing, balanced fertilization, and regular scouting.",
    },
    "Herbicide Growth Damage": {
        "severity": "🟡 Medium",
        "emoji": "⚗️",
        "symptoms": "Cupped or strap-shaped leaves, abnormal growth patterns, epinasty (downward bending).",
        "cause": "Herbicide drift or misapplication — commonly from 2,4-D or Dicamba on nearby fields.",
        "treatment": [
            "Apply foliar nutrients (Zinc sulfate 2g/L + Urea 1%)",
            "Irrigate to help plant recover",
            "Avoid further herbicide application near cotton",
            "Plants usually recover if damage is mild",
        ],
        "prevention": "Use herbicide-tolerant varieties if available. Apply herbicides on calm days. Maintain buffer zones.",
    },
    "Leaf Hopper Jassids": {
        "severity": "🟡 Medium",
        "emoji": "🐛",
        "symptoms": "Yellowing leaf margins (hopper burn), downward curling of leaf edges, reduced photosynthesis.",
        "cause": "Amrasca biguttula biguttula — sucking pest that feeds on leaf undersides.",
        "treatment": [
            "Spray Acetamiprid (0.2g/L) or Dimethoate (2ml/L)",
            "Apply Neem-based insecticide for organic control",
            "Ensure good field sanitation",
            "Economic threshold: 1–2 jassids per leaf",
        ],
        "prevention": "Grow hairy-leaf varieties. Avoid excess nitrogen. Encourage natural predators (ladybugs, lacewings).",
    },
    "Leaf Reddening": {
        "severity": "🟠 Low–Medium",
        "emoji": "🍁",
        "symptoms": "Premature reddening/purpling of leaves, often starting from lower canopy, early senescence.",
        "cause": "Nutrient deficiency (Mg, K), waterlogging, heavy boll load, or mite damage.",
        "treatment": [
            "Foliar spray of MgSO₄ (10g/L) + KNO₃ (5g/L)",
            "Improve drainage if waterlogged",
            "Check for mite infestation underneath leaves",
            "Balanced NPK fertilization",
        ],
        "prevention": "Soil test before sowing. Ensure adequate potassium and magnesium. Avoid water stress.",
    },
    "Leaf Variegation": {
        "severity": "🟢 Low",
        "emoji": "🎨",
        "symptoms": "Irregular light and dark green patches on leaves, mosaic-like patterns, mild chlorosis.",
        "cause": "Genetic factors, mild viral infection, or nutrient imbalance.",
        "treatment": [
            "If viral: control whitefly vectors",
            "Foliar application of micronutrients (Zn, Fe, Mn)",
            "Monitor for progression to more severe symptoms",
            "Usually does not significantly affect yield",
        ],
        "prevention": "Use quality seeds from certified sources. Maintain balanced soil fertility.",
    },
    "Fusarium Wilt": {
        "severity": "🔴 High",
        "emoji": "🍂",
        "symptoms": "Wilting of branches (often one-sided), browning of vascular tissue, yellowing & drooping of leaves, plant death.",
        "cause": "Fusarium oxysporum f. sp. vasinfectum — soil-borne fungus persisting for years.",
        "treatment": [
            "No effective chemical cure once infected",
            "Uproot and burn infected plants",
            "Apply Trichoderma viride (5g/kg seed) as biocontrol",
            "Soil drenching with Carbendazim (1g/L) around healthy plants",
        ],
        "prevention": "Grow resistant varieties. Long crop rotation (3+ years). Avoid wounding roots during cultivation. Improve soil drainage.",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING & INFERENCE (Real)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model(model_name, dataset_name):
    """
    Load a trained PyTorch model from saved_models/ directory.
    Only called when USE_REAL_MODEL = True.

    Expected path format:
        saved_models/SAR-CLD_2024/DenseNet121_best.pt
        saved_models/Cotton_Leaf_Disease/ResNet50_best.pt
    """
    import torch
    import torch.nn as nn
    from torchvision import models

    dataset_folder = dataset_name.replace(" ", "_").replace("-", "-")
    if dataset_name == "SAR-CLD 2024":
        dataset_folder = "SAR-CLD_2024"
        num_classes = 7
    else:
        dataset_folder = "Cotton_Leaf_Disease"
        num_classes = 4

    # Map display names to filename-safe names
    name_map = {
        "ResNet50": "ResNet50",
        "DenseNet121": "DenseNet121",
        "EfficientNetB7": "EfficientNetB7",
        "ViT-B/16": "ViT_B16",
        "Swin-T": "Swin_T",
        "ConvNeXt-T": "ConvNeXt_T",
    }
    safe_name = name_map.get(model_name, model_name)
    model_path = f"saved_models/{dataset_folder}/{safe_name}_best.pt"

    # Build model architecture (same as your training code)
    if model_name == "ResNet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif model_name == "DenseNet121":
        m = models.densenet121(weights=None)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    elif model_name == "EfficientNetB7":
        m = models.efficientnet_b7(weights=None)
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_f, num_classes),
        )
    elif model_name == "ViT-B/16":
        m = models.vit_b_16(weights=None)
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
    elif model_name == "Swin-T":
        m = models.swin_t(weights=None)
        m.head = nn.Linear(m.head.in_features, num_classes)
    elif model_name == "ConvNeXt-T":
        m = models.convnext_tiny(weights=None)
        in_f = m.classifier[2].in_features
        m.classifier[2] = nn.Linear(in_f, num_classes)
    else:
        st.error(f"Unknown model: {model_name}")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m.load_state_dict(torch.load(model_path, map_location=device))
    m.to(device)
    m.eval()
    return m


def real_predict(model, image_pil, model_name, class_names):
    """Run real inference using a loaded PyTorch model."""
    import torch
    import torch.nn.functional as F
    from torchvision import transforms

    img_size = 600 if model_name == "EfficientNetB7" else 224

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_tensor = transform(image_pil.convert("RGB")).unsqueeze(0).to(device)

    start = time.time()
    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    inference_ms = (time.time() - start) * 1000

    pred_idx = int(np.argmax(probs))
    return {
        "predicted": class_names[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {cn: float(probs[i]) for i, cn in enumerate(class_names)},
        "inference_time": round(inference_ms, 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATED INFERENCE (Demo mode when no .pt files available)
# ══════════════════════════════════════════════════════════════════════════════
def simulated_predict(class_names):
    """Generate realistic-looking fake predictions for demo/prototype."""
    n = len(class_names)
    winner = np.random.randint(n)
    probs = np.array([
        (0.7 + np.random.random() * 0.25) if i == winner else np.random.random() * 0.05
        for i in range(n)
    ])
    probs /= probs.sum()
    return {
        "predicted": class_names[winner],
        "confidence": float(probs[winner]),
        "probabilities": {cn: float(probs[i]) for i, cn in enumerate(class_names)},
        "inference_time": round(80 + np.random.random() * 120, 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CHATBOT RESPONSES
# ══════════════════════════════════════════════════════════════════════════════
def get_chat_response(msg: str) -> str:
    lower = msg.lower()

    if any(w in lower for w in ["curl", "clcuv"]) and any(w in lower for w in ["treat", "cure", "solution", "fix", "spray"]):
        return (
            "🦟 **Cotton Leaf Curl Virus (CLCuV) Management:**\n\n"
            "1. **Spray Imidacloprid** (0.5ml/L) or **Thiamethoxam** to kill whiteflies\n"
            "2. **Neem oil** (5ml/L) as organic repellent\n"
            "3. **Remove severely infected plants** and burn them\n"
            "4. **Yellow sticky traps** — 20–25 per acre\n\n"
            "🛡️ **Prevention:** Plant resistant varieties, sow early (April–May), monitor whitefly weekly."
        )
    if any(w in lower for w in ["curl", "clcuv"]):
        return (
            "🍃 **Cotton Leaf Curl Virus (CLCuV)** is one of the most devastating diseases for cotton in Pakistan. "
            "It causes upward/downward curling, stunted growth, and up to **80% yield loss**. "
            "Transmitted by **whiteflies** (Bemisia tabaci). Would you like treatment or prevention tips?"
        )
    if "blight" in lower and any(w in lower for w in ["treat", "solution", "fix", "spray"]):
        return (
            "💊 **Bacterial Blight Treatment:**\n\n"
            "1. Spray **Copper oxychloride** (3g/L) or **Streptocycline** (0.5g/L)\n"
            "2. Remove and destroy infected debris\n"
            "3. Use disease-free certified seeds\n"
            "4. Avoid overhead irrigation\n\n"
            "💡 **Tip:** Treat seeds with Carboxin + Thiram before sowing."
        )
    if "blight" in lower:
        return (
            "🔬 **Bacterial Blight** is caused by *Xanthomonas citri pv. malvacearum*. "
            "Look for angular water-soaked lesions turning brown/black. "
            "Spreads through rain and contaminated seeds. Want treatment advice?"
        )
    if "whitefl" in lower:
        return (
            "🦟 **Whitefly Control:**\n\n"
            "1. 💊 Imidacloprid, Thiamethoxam, or Spiromesifen\n"
            "2. 🌿 Neem oil (5ml/L), yellow sticky traps\n"
            "3. 🐞 Encourage Encarsia formosa parasitoid\n"
            "4. 🌾 Remove weeds, avoid excess nitrogen\n\n"
            "Monitor weekly — ETL: 5–8 adults per leaf."
        )
    if any(w in lower for w in ["wilt", "fusarium"]):
        return (
            "🍂 **Fusarium Wilt** — soil-borne, **no chemical cure** once infected.\n\n"
            "1. Uproot & burn infected plants\n"
            "2. Apply **Trichoderma viride** (5g/kg seed)\n"
            "3. Long crop rotation (3+ years)\n"
            "4. Grow **resistant varieties**\n\n"
            "⚠️ Prevention is the only real defense!"
        )
    if "healthy" in lower:
        return (
            "✅ A healthy cotton plant shows green, turgid leaves.\n\n"
            "Keep up: 🔍 Regular scouting (2x/week) • 🧪 Balanced NPK • 💧 Proper irrigation • 🧹 Field hygiene"
        )
    if any(w in lower for w in ["model", "accuracy", "which model", "best model"]):
        return (
            "📊 **Model Performance:**\n\n"
            "🥇 **DenseNet121** — 98.36% (7M params)\n"
            "🥈 **Swin-T** — 98.36% (27.5M params)\n"
            "🥉 **ResNet50** — 97.90% (23.5M params)\n"
            "4️⃣ ConvNeXt-T — 97.66%\n"
            "5️⃣ ViT-B/16 — 95.79%\n\n"
            "DenseNet121 offers the best accuracy-to-efficiency ratio!"
        )
    if any(w in lower for w in ["dataset", "sar-cld", "classes"]):
        return (
            "📁 **Datasets Used:**\n\n"
            "🔵 **SAR-CLD 2024** — 7 classes: Bacterial Blight, Curl Virus, Healthy, "
            "Herbicide Damage, Jassids, Leaf Reddening, Leaf Variegation\n\n"
            "🟢 **Cotton Leaf Disease** — 4 classes: Bacterial Blight, Curl Virus, Fusarium Wilt, Healthy"
        )
    if any(w in lower for w in ["jassid", "hopper"]):
        return (
            "🐛 **Leaf Hopper Jassids** cause 'hopper burn' — yellowing margins.\n\n"
            "1. Spray Acetamiprid (0.2g/L) or Dimethoate\n"
            "2. Neem-based organic spray\n"
            "3. ETL: 1–2 jassids per leaf\n\n"
            "🛡️ Grow hairy-leaf varieties, avoid excess nitrogen."
        )
    if any(w in lower for w in ["redden", "purple", "red leaf"]):
        return (
            "🍁 **Leaf Reddening** — premature purpling from lower canopy.\n\n"
            "**Causes:** Mg/K deficiency, waterlogging, mites\n"
            "**Fix:** MgSO₄ (10g/L) + KNO₃ (5g/L) spray. Improve drainage."
        )
    if any(w in lower for w in ["hello", "hi", "hey", "salam", "assalam"]):
        return (
            "Wa Alaikum Assalam! 🌿✨\n\n"
            "Welcome to Cotton Guard! I can help with:\n\n"
            "🔍 Disease identification\n"
            "💊 Treatment & prevention\n"
            "🦟 Pest management\n"
            "📊 Model & dataset info\n\n"
            "Just ask away!"
        )
    if "thank" in lower:
        return "You're welcome! 🌱💚 Wishing you a healthy and productive cotton season!"
    if any(w in lower for w in ["help", "what can you"]):
        return (
            "🌿 **I can help with:**\n\n"
            "🔍 Disease identification\n"
            "💊 Treatment plans (chemical & organic)\n"
            "🛡️ Prevention strategies\n"
            "🦟 Whitefly & pest management\n"
            "📊 Model accuracy & comparison\n"
            "📁 Dataset details\n\n"
            "Describe your problem or ask a question!"
        )
    return (
        "🌿 I'm your cotton crop assistant! Try asking:\n\n"
        "• \"How to treat curl virus?\"\n"
        "• \"What causes bacterial blight?\"\n"
        "• \"Whitefly control methods\"\n"
        "• \"Which model is best?\""
    )


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT PAGE CONFIG & STYLING
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Cotton Guard — Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown("""
<style>
    /* ── Global ── */
    .stApp {
        background: linear-gradient(135deg, #0a0f1c 0%, #0d1525 40%, #0f1a2e 100%);
    }
    
    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c1222 0%, #111827 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.15);
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li {
        color: #cbd5e1;
    }
    
    /* ── Headers ── */
    h1, h2, h3 { color: #f1f5f9 !important; }
    
    /* ── Cards ── */
    .css-card {
        background: linear-gradient(135deg, rgba(15,23,42,0.7), rgba(20,30,50,0.6));
        border: 1px solid rgba(51,65,85,0.4);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 16px;
        backdrop-filter: blur(20px);
    }
    
    /* ── Metric Cards ── */
    .metric-card {
        border-radius: 14px;
        padding: 16px;
        text-align: center;
    }
    .metric-label {
        font-size: 11px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 20px;
        font-weight: 800;
    }
    
    /* ── Disease Cards ── */
    .symptom-card {
        background: rgba(255,107,107,0.06);
        border: 1px solid rgba(255,107,107,0.15);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 10px;
    }
    .cause-card {
        background: rgba(255,165,2,0.06);
        border: 1px solid rgba(255,165,2,0.15);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 10px;
    }
    .treatment-card {
        background: rgba(46,213,115,0.06);
        border: 1px solid rgba(46,213,115,0.15);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 10px;
    }
    .prevention-card {
        background: rgba(99,102,241,0.08);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 10px;
    }
    
    /* ── Probability Bar ── */
    .prob-bar-bg {
        background: rgba(30,41,59,0.6);
        border-radius: 6px;
        height: 10px;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 1s ease;
    }
    
    /* ── Chat ── */
    .chat-msg-bot {
        background: linear-gradient(135deg, rgba(30,41,59,0.8), rgba(40,50,70,0.6));
        border: 1px solid rgba(99,102,241,0.15);
        border-radius: 16px 16px 16px 4px;
        padding: 12px 16px;
        color: #e2e8f0;
        font-size: 14px;
        line-height: 1.65;
        margin-bottom: 12px;
    }
    .chat-msg-user {
        background: linear-gradient(135deg, #0ea5e9, #2563eb);
        border-radius: 16px 16px 4px 16px;
        padding: 12px 16px;
        color: #ffffff;
        font-size: 14px;
        line-height: 1.65;
        margin-bottom: 12px;
        text-align: right;
    }
    
    /* ── Buttons ── */
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    /* ── File Uploader ── */
    .stFileUploader {
        border-radius: 16px;
    }
    
    /* ── Selectbox ── */
    .stSelectbox label { color: #94a3b8 !important; font-weight: 600 !important; }
    
    /* ── Dividers ── */
    hr { border-color: rgba(51,65,85,0.3) !important; }
    
    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
    
    /* ── Hide default Streamlit elements ── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    
    /* ── Tag badges ── */
    .tag-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 6px;
    }
    
    /* ── Section Label ── */
    .section-label {
        font-size: 11px;
        color: #94a3b8;
        letter-spacing: 2px;
        text-transform: uppercase;
        font-weight: 700;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Assalam o Alaikum! 🌿✨ I'm your Cotton Guard Assistant. Ask me about diseases, treatments, prevention, or anything about your cotton crop!"}
    ]
if "prediction" not in st.session_state:
    st.session_state.prediction = None


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — CHATBOT
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Chat Header
    st.markdown("""
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;">
        <div style="width:42px; height:42px; border-radius:12px;
                    background:linear-gradient(135deg, #6366f1, #8b5cf6, #a855f7);
                    display:flex; align-items:center; justify-content:center;
                    box-shadow: 0 0 20px rgba(99,102,241,0.4);">
            <span style="font-size:20px;">🤖</span>
        </div>
        <div>
            <div style="font-size:16px; font-weight:700; color:#e2e8f0;">Crop Assistant</div>
            <div style="font-size:11px; color:#22c55e;">● Online — Ask me anything</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Chat Messages
    chat_container = st.container(height=420)
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "assistant":
                st.markdown(f'<div class="chat-msg-bot">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-msg-user">{msg["content"]}</div>', unsafe_allow_html=True)

    # Quick Action Buttons
    st.markdown('<p class="section-label">⚡ Quick Questions</p>', unsafe_allow_html=True)
    qcols = st.columns(2)
    quick_questions = [
        ("🦟 Curl Virus", "How to treat curl virus?"),
        ("🐛 Whitefly", "Whitefly control methods"),
        ("📊 Models", "Which model is best?"),
        ("💡 Help", "What can you help with?"),
    ]
    for i, (label, question) in enumerate(quick_questions):
        with qcols[i % 2]:
            if st.button(label, key=f"quick_{i}", use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": question})
                response = get_chat_response(question)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()

    st.markdown("---")

    # Chat Input
    user_input = st.chat_input("Ask about cotton diseases...", key="chat_input")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        response = get_chat_response(user_input)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PANEL
# ══════════════════════════════════════════════════════════════════════════════

# ── Header ──
st.markdown("""
<div style="display:flex; align-items:center; gap:16px; margin-bottom:8px;">
    <div style="width:54px; height:54px; border-radius:14px;
                background:linear-gradient(135deg, #10b981, #059669, #047857);
                display:flex; align-items:center; justify-content:center;
                box-shadow: 0 0 30px rgba(16,185,129,0.4);
                font-size: 28px;">
        🌿
    </div>
    <div>
        <h1 style="margin:0; font-size:32px; font-weight:800;
                   background:linear-gradient(135deg, #10b981, #34d399, #6ee7b7);
                   -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            Cotton Guard
        </h1>
        <p style="margin:0; font-size:13px; color:#64748b; letter-spacing:2px; text-transform:uppercase;">
            AI-Powered Disease Detection for Cotton Farmers 🇵🇰
        </p>
    </div>
    <div style="margin-left:auto; display:flex; gap:6px;">
        <span class="tag-badge" style="background:rgba(16,185,129,0.15); color:#34d399; border:1px solid rgba(16,185,129,0.3);">v1.0</span>
        <span class="tag-badge" style="background:rgba(99,102,241,0.15); color:#818cf8; border:1px solid rgba(99,102,241,0.3);">
            {'🟢 Real Model' if USE_REAL_MODEL else '🟡 Demo Mode'}
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Upload & Config ──
col_upload, col_config = st.columns([3, 2])

with col_upload:
    st.markdown('<p class="section-label">📸 Upload Cotton Leaf Image</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop your cotton leaf image here",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"📎 {uploaded_file.name} ({uploaded_file.size // 1024} KB)", use_container_width=True)

with col_config:
    st.markdown('<p class="section-label">🗂️ Dataset</p>', unsafe_allow_html=True)
    dataset = st.selectbox(
        "Dataset",
        ["SAR-CLD 2024", "Cotton Leaf Disease"],
        label_visibility="collapsed",
    )
    class_names = SAR_CLD_CLASSES if dataset == "SAR-CLD 2024" else COTTON_LEAF_CLASSES

    ds_info = "7 classes • SAR imagery" if dataset == "SAR-CLD 2024" else "4 classes • RGB imagery"
    st.caption(f"📋 {ds_info}")

    st.markdown('<p class="section-label">🧠 Model</p>', unsafe_allow_html=True)
    model_name = st.selectbox(
        "Model",
        MODELS_LIST,
        index=1,  # Default: DenseNet121
        label_visibility="collapsed",
    )
    stats = MODEL_STATS.get(model_name, {})
    st.caption(f"📐 {stats.get('params', '—')} params  •  ⚡ {stats.get('flops', '—')} FLOPs  •  🎯 {stats.get('acc', '—')}")

    st.markdown("")

    # Analyze Button
    analyze_btn = st.button(
        "✨ Analyze Leaf",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None,
    )

# ── Run Prediction ──
if analyze_btn and uploaded_file:
    with st.spinner("🔍 Analyzing leaf image..."):
        time.sleep(1.5)  # Simulated delay for UX

        if USE_REAL_MODEL:
            try:
                model = load_model(model_name, dataset)
                if model is not None:
                    image_pil = Image.open(uploaded_file)
                    result = real_predict(model, image_pil, model_name, class_names)
                    st.session_state.prediction = result
                else:
                    st.error("❌ Failed to load model.")
            except FileNotFoundError:
                st.error(f"❌ Model file not found! Please place your `.pt` file in `saved_models/` directory.")
                st.info("👆 See instructions at the top of `app.py` for the expected file structure.")
            except Exception as e:
                st.error(f"❌ Error during inference: {e}")
        else:
            result = simulated_predict(class_names)
            st.session_state.prediction = result

# ── Display Results ──
if st.session_state.prediction:
    pred = st.session_state.prediction
    info = DISEASE_INFO.get(pred["predicted"], {})
    is_healthy = pred["predicted"] == "Healthy"

    st.markdown("---")

    # ── Prediction Banner ──
    if is_healthy:
        banner_bg = "rgba(46,213,115,0.1)"
        banner_border = "rgba(46,213,115,0.3)"
        banner_color = "#2ed573"
    else:
        disease_color = info.get("emoji", "⚠️")
        banner_bg = "rgba(255,75,87,0.08)"
        banner_border = "rgba(255,75,87,0.25)"
        banner_color = "#ff4757"

    st.markdown(f"""
    <div style="background:{banner_bg}; border:1px solid {banner_border};
                border-radius:20px; padding:24px; margin-bottom:20px;">
        <div style="font-size:11px; letter-spacing:2px; text-transform:uppercase; 
                    font-weight:700; color:{banner_color}; margin-bottom:8px;">
            {'✅' if is_healthy else '⚠️'} Prediction Result
        </div>
        <div style="font-size:32px; font-weight:800; color:{banner_color}; margin-bottom:6px;">
            {info.get('emoji', '🔍')} {pred['predicted']}
        </div>
        <div style="font-size:14px; color:#94a3b8;">
            🎯 Confidence: <strong style="color:{banner_color}">{pred['confidence']*100:.1f}%</strong> &nbsp;•&nbsp;
            ⚡ Inference: {pred['inference_time']}ms &nbsp;•&nbsp;
            🧠 {model_name} &nbsp;•&nbsp;
            Severity: <strong>{info.get('severity', 'N/A')}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not USE_REAL_MODEL:
        st.warning("⚠️ **Demo Mode** — Predictions are simulated. Place your trained `.pt` model files in `saved_models/` and set `USE_REAL_MODEL = True` for real inference.")

    # ── Confidence + Probabilities ──
    col_conf, col_probs = st.columns([1, 2])

    with col_conf:
        st.markdown('<p class="section-label">🎯 Confidence Score</p>', unsafe_allow_html=True)
        conf_pct = pred["confidence"] * 100
        conf_color = "#2ed573" if conf_pct > 80 else "#ffa502" if conf_pct > 50 else "#ff4757"
        st.markdown(f"""
        <div class="css-card" style="text-align:center;">
            <div style="font-size:56px; font-weight:800; color:{conf_color}; margin:10px 0;">
                {conf_pct:.0f}%
            </div>
            <div style="background:rgba(30,41,59,0.6); border-radius:8px; height:12px; overflow:hidden; margin:12px 0;">
                <div style="width:{conf_pct}%; height:100%; border-radius:8px;
                            background:linear-gradient(90deg, {conf_color}, {conf_color}aa);
                            box-shadow: 0 0 12px {conf_color}50;"></div>
            </div>
            <div style="font-size:12px; color:#64748b;">Model: {model_name}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_probs:
        st.markdown('<p class="section-label">📊 Class Probabilities</p>', unsafe_allow_html=True)
        sorted_probs = sorted(pred["probabilities"].items(), key=lambda x: -x[1])
        prob_colors = ["#6c5ce7", "#00cec9", "#fdcb6e", "#e17055", "#00b894", "#e84393", "#0984e3", "#ff7675"]
        
        prob_html = '<div class="css-card">'
        for i, (cls, prob) in enumerate(sorted_probs):
            color = prob_colors[i % len(prob_colors)]
            bar_width = max(prob * 100, 0.8)
            is_top = i == 0
            prob_html += f"""
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">
                <span style="width:22px; height:22px; border-radius:6px; font-size:11px; font-weight:700;
                             background:{color}25; color:{color};
                             display:inline-flex; align-items:center; justify-content:center; flex-shrink:0;">
                    {i+1}
                </span>
                <span style="width:160px; font-size:13px; font-weight:{'700' if is_top else '400'};
                             color:{'#f1f5f9' if is_top else '#64748b'}; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">
                    {cls}
                </span>
                <div style="flex:1;">
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill" style="width:{bar_width}%;
                             background:{'linear-gradient(90deg, '+color+', '+color+'cc)' if is_top else 'rgba(71,85,105,0.6)'};
                             {'box-shadow: 0 0 10px '+color+'50;' if is_top else ''}"></div>
                    </div>
                </div>
                <span style="width:48px; text-align:right; font-size:13px; font-weight:700;
                             color:{color if is_top else '#475569'};">
                    {prob*100:.1f}%
                </span>
            </div>
            """
        prob_html += "</div>"
        st.markdown(prob_html, unsafe_allow_html=True)

    # ── Disease Details ──
    if not is_healthy and info:
        st.markdown("---")
        st.markdown('<p class="section-label">🔬 Disease Information</p>', unsafe_allow_html=True)

        # Symptoms
        st.markdown(f"""
        <div class="symptom-card">
            <h4 style="color:#ff6b6b; margin:0 0 8px; font-size:14px;">🔍 Symptoms</h4>
            <p style="color:#cbd5e1; margin:0; line-height:1.7; font-size:14px;">{info['symptoms']}</p>
        </div>
        """, unsafe_allow_html=True)

        # Cause
        st.markdown(f"""
        <div class="cause-card">
            <h4 style="color:#ffa502; margin:0 0 8px; font-size:14px;">🧬 Cause</h4>
            <p style="color:#cbd5e1; margin:0; line-height:1.7; font-size:14px;">{info['cause']}</p>
        </div>
        """, unsafe_allow_html=True)

        # Treatment
        treatment_items = ""
        for i, t in enumerate(info["treatment"]):
            treatment_items += f"""
            <div style="display:flex; gap:10px; margin-bottom:8px; align-items:flex-start;">
                <span style="width:22px; height:22px; border-radius:6px; font-size:11px; font-weight:700;
                             background:rgba(46,213,115,0.15); color:#2ed573;
                             display:inline-flex; align-items:center; justify-content:center; flex-shrink:0;">{i+1}</span>
                <span style="font-size:14px; color:#e2e8f0; line-height:1.6;">{t}</span>
            </div>
            """
        st.markdown(f"""
        <div class="treatment-card">
            <h4 style="color:#2ed573; margin:0 0 12px; font-size:14px;">💊 Recommended Treatment</h4>
            {treatment_items}
        </div>
        """, unsafe_allow_html=True)

        # Prevention
        st.markdown(f"""
        <div class="prevention-card">
            <h4 style="color:#818cf8; margin:0 0 8px; font-size:14px;">🛡️ Prevention</h4>
            <p style="color:#cbd5e1; margin:0; line-height:1.7; font-size:14px;">{info['prevention']}</p>
        </div>
        """, unsafe_allow_html=True)

    elif is_healthy:
        st.markdown("---")
        st.markdown("""
        <div style="background:rgba(46,213,115,0.08); border:1px solid rgba(46,213,115,0.25);
                    border-radius:20px; padding:30px; text-align:center;">
            <div style="font-size:56px; margin-bottom:12px;">🌿</div>
            <h3 style="color:#2ed573; margin:0 0 10px;">Your Cotton Leaf is Healthy!</h3>
            <p style="color:#94a3b8; font-size:14px; line-height:1.6; max-width:500px; margin:0 auto;">
                No disease detected. Keep up your good crop management — regular scouting, 
                balanced fertilization, and proper irrigation.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── Technical Specifications ──
    st.markdown("---")
    st.markdown('<p class="section-label">⚙️ Technical Specifications</p>', unsafe_allow_html=True)

    spec_cols = st.columns(4)
    specs = [
        ("🧠", "Architecture", model_name, stats.get("color", "#818cf8")),
        ("📐", "Parameters", stats.get("params", "—"), "#00d2d3"),
        ("⚡", "FLOPs", stats.get("flops", "—"), "#ffa502"),
        ("🎯", "Accuracy", stats.get("acc", "—"), "#2ed573"),
    ]
    for col, (icon, label, value, color) in zip(spec_cols, specs):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="background:{color}08; border:1px solid {color}25;">
                <div style="font-size:22px; margin-bottom:4px;">{icon}</div>
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="color:{color};">{value}</div>
            </div>
            """, unsafe_allow_html=True)

# ── Footer ──
st.markdown("---")
st.markdown("""
<div style="text-align:center; padding:12px 0; color:#1e293b; font-size:12px;">
    Cotton Guard v1.0 — Built with ❤️ for Pakistani Cotton Farmers 🇵🇰
</div>
""", unsafe_allow_html=True)
