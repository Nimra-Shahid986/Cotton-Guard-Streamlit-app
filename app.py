"""
Cotton Guard — Cotton Leaf Disease Detection System
Exact LDASN architecture from training code + ConvNeXt-T
"""
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import time

st.set_page_config(page_title="Cotton Guard — Leaf Disease Detection", page_icon="🍃", layout="wide", initial_sidebar_state="expanded")

# ─── CSS ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=Nunito:wght@300;400;500;600;700;800&family=Fira+Code:wght@400;500&display=swap');
.stApp { background: #e8efe2; font-family: 'Nunito', sans-serif; }
.app-header { text-align:center; padding:1.5rem 0 0.5rem; }
.app-header h1 { font-family:'DM Serif Display',serif; font-size:2.6rem; color:#2d5016; margin:0; }
.app-header .subtitle { color:#7a8b6e; font-size:0.95rem; margin-top:0.2rem; }
.header-divider { width:80px; height:3px; background:linear-gradient(90deg,#8b6914,#2d5016,#8b6914); margin:0.8rem auto 1.5rem; border-radius:2px; }
.earth-card { background:#d9e4cf; border:1px solid #b8c9a8; border-radius:14px; padding:1.4rem; margin-bottom:1rem; box-shadow:0 2px 12px rgba(45,80,22,0.08); }
.earth-card-header { font-family:'Fira Code',monospace; font-size:0.68rem; font-weight:500; color:#8b6914; text-transform:uppercase; letter-spacing:2.5px; margin-bottom:0.8rem; padding-bottom:0.5rem; border-bottom:1px solid #c5d4b5; }
.prediction-box { background:linear-gradient(135deg,#c8ddb8,#b8d0a5); border:1px solid #8baf72; border-left:4px solid #2d5016; border-radius:12px; padding:1.2rem 1.5rem; margin:0.8rem 0; }
.prediction-label { font-size:0.68rem; font-weight:700; color:#2d5016; text-transform:uppercase; letter-spacing:2.5px; }
.prediction-name { font-family:'DM Serif Display',serif; font-size:1.8rem; color:#1a3a0a; margin:0.3rem 0 0.1rem; }
.prediction-index { font-size:0.82rem; color:#5a6650; }
.disease-box { background:linear-gradient(135deg,#efe0c0,#e8d5a8); border:1px solid #d4a843; border-left:4px solid #8b6914; border-radius:12px; padding:1.2rem 1.5rem; margin:0.8rem 0; }
.disease-label { font-size:0.68rem; font-weight:700; color:#8b6914; text-transform:uppercase; letter-spacing:2.5px; }
.confidence-section { text-align:center; padding:0.8rem 0; }
.confidence-pct { font-family:'DM Serif Display',serif; font-size:2.8rem; line-height:1; }
.metric-row { display:flex; gap:0.7rem; flex-wrap:wrap; margin:0.8rem 0; }
.metric-card { flex:1; min-width:110px; background:#cddabe; border:1px solid #b8c9a8; border-radius:10px; padding:0.7rem 0.5rem; text-align:center; }
.metric-card .metric-label { font-size:0.6rem; color:#8b6914; text-transform:uppercase; letter-spacing:1.5px; font-weight:600; }
.metric-card .metric-value { font-size:0.95rem; font-weight:700; color:#2d3a1e; margin-top:0.15rem; font-family:'Fira Code',monospace; }
.prob-item { display:flex; align-items:center; margin:0.45rem 0; gap:0.7rem; }
.prob-name { width:170px; font-size:0.8rem; color:#3a4a30; font-weight:600; text-align:right; flex-shrink:0; }
.prob-bar-bg { flex:1; height:10px; background:#c5d4b5; border-radius:5px; overflow:hidden; }
.prob-bar-fill { height:100%; border-radius:5px; transition:width 0.5s ease; }
.prob-pct { width:48px; font-size:0.75rem; color:#5a6650; font-family:'Fira Code',monospace; text-align:right; flex-shrink:0; }
.info-card { background:#cddabe; border:1px solid #b8c9a8; border-radius:10px; padding:1rem 1.2rem; margin:0.5rem 0; }
.info-card h4 { color:#2d5016; font-size:0.88rem; margin:0 0 0.35rem 0; font-weight:700; }
.info-card p { color:#3a4a30; font-size:0.82rem; margin:0; line-height:1.6; }
.info-card-warn h4 { color:#8b6914; }
div[data-testid="stFileUploader"] { background:#d9e4cf; border:2px dashed #a8b898; border-radius:14px; padding:1rem; }
div[data-testid="stFileUploader"]:hover { border-color:#8b6914; }
section[data-testid="stSidebar"] { background:#2d3a1e !important; }
section[data-testid="stSidebar"] .stMarkdown p, section[data-testid="stSidebar"] .stMarkdown li { color:#d4dccb !important; }
div[data-testid="stSelectbox"] > div > div { background:#d9e4cf !important; border-color:#b8c9a8 !important; }
.stSelectbox label { color:#5a6650 !important; font-family:'Fira Code',monospace !important; font-size:0.68rem !important; text-transform:uppercase; letter-spacing:2px; }
.stButton > button[kind="primary"] { background:linear-gradient(135deg,#2d5016,#3d6b20) !important; color:white !important; border:none !important; border-radius:10px !important; font-weight:700 !important; font-size:1rem !important; }
.stButton > button[kind="primary"]:hover { background:linear-gradient(135deg,#3d6b20,#4a8028) !important; }
.stProgress > div > div { background-color:#2d5016 !important; }
#MainMenu {visibility:hidden;} footer {visibility:hidden;} .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)

# ─── Constants ─────────────────────────────────────────────────────────────
SAR_CLD_CLASSES = ["Bacterial Blight","Curl Virus","Healthy Leaf","Herbicide Growth Damage","Leaf Hopper Jassids","Leaf Redding","Leaf Variegation"]
COTTON_LEAF_CLASSES = ["Bacterial Blight","Curl Virus","Fussarium Wilt","Healthy"]
DATASET_INFO = {
    "SAR-CLD 2024 — 7 Classes": {"classes":SAR_CLD_CLASSES,"model_file":"models/swin_t_best.pt","architecture":"LDASN (Lightweight Dual-Attention)","arch_key":"LDASN","img_size":224},
    "Cotton Leaf Disease — 4 Classes": {"classes":COTTON_LEAF_CLASSES,"model_file":"models/convnext_t_best.pt","architecture":"ConvNeXt Tiny (ConvNeXt-T)","arch_key":"ConvNeXt_T","img_size":224},
}
DISEASE_INFO = {
    "Bacterial Blight": {"severity":"High","icon":"🔴","description":"Angular water-soaked lesions on leaves that turn brown. Causes defoliation and boll rot.","symptoms":"Water-soaked angular spots, blackening of veins, premature defoliation.","treatment":"Use copper-based bactericides. Plant resistant varieties. Remove and destroy infected debris.","prevention":"Use certified disease-free seeds, crop rotation with non-host crops, avoid overhead irrigation."},
    "Curl Virus": {"severity":"Very High","icon":"🔴","description":"Transmitted by whiteflies, causes upward or downward curling of leaves, stunted growth, and severe yield loss.","symptoms":"Leaf curling, thickened veins, enation (leaf-like outgrowths), stunted plants, reduced boll formation.","treatment":"Control whitefly vectors with insecticides (imidacloprid, acetamiprid). Remove infected plants early. Use sticky traps.","prevention":"Plant resistant varieties (BT cotton with CLCuV tolerance), early sowing, maintain field hygiene."},
    "Healthy Leaf": {"severity":"None","icon":"🟢","description":"The leaf appears healthy with no visible signs of disease or pest damage.","symptoms":"No symptoms — uniform green color, normal leaf shape and size.","treatment":"No treatment needed. Continue regular crop management.","prevention":"Maintain balanced nutrition, proper irrigation scheduling, and regular scouting."},
    "Healthy": {"severity":"None","icon":"🟢","description":"The leaf appears healthy with no visible signs of disease or pest damage.","symptoms":"No symptoms.","treatment":"No treatment needed.","prevention":"Maintain balanced nutrition, proper irrigation, and regular scouting."},
    "Herbicide Growth Damage": {"severity":"Medium","icon":"🟡","description":"Damage from herbicide drift or misapplication, resulting in abnormal leaf growth.","symptoms":"Cupped or strapped leaves, abnormal growth, epinasty, chlorosis.","treatment":"Foliar application of growth regulators. Provide adequate irrigation and nutrition.","prevention":"Proper herbicide application techniques, avoid spraying on windy days, calibrate sprayers."},
    "Leaf Hopper Jassids": {"severity":"Medium-High","icon":"🟠","description":"Jassids suck cell sap from leaves causing yellowing and curling of leaf margins.","symptoms":"Yellowing of leaf margins, downward curling, hopper burn in severe cases.","treatment":"Apply systemic insecticides (thiamethoxam, imidacloprid). Use neem-based sprays.","prevention":"Use resistant varieties, intercropping, maintain natural predators."},
    "Leaf Redding": {"severity":"Medium","icon":"🟡","description":"Reddening of leaves due to nutrient deficiency (often magnesium) or physiological stress.","symptoms":"Reddish-purple discoloration, starting from lower leaves and moving upward.","treatment":"Foliar application of magnesium sulphate. Correct nutrient imbalances.","prevention":"Regular soil testing, balanced NPK application."},
    "Leaf Variegation": {"severity":"Medium","icon":"🟡","description":"Irregular patches of different colors on leaves, often caused by viral infections.","symptoms":"Mosaic patterns, irregular light and dark green patches, sometimes yellow streaks.","treatment":"Remove severely affected plants. Control insect vectors.","prevention":"Use virus-free planting material, control aphid and whitefly vectors."},
    "Fussarium Wilt": {"severity":"High","icon":"🔴","description":"Soil-borne fungal disease that blocks water-conducting vessels, causing wilting and death.","symptoms":"Yellowing on one side, wilting despite adequate moisture, brown vascular tissue.","treatment":"Remove and destroy infected plants. Soil solarization. Trichoderma biocontrol.","prevention":"Use resistant varieties, long crop rotation (3+ years), avoid waterlogging."},
}

# ─── EXACT LDASN Architecture (from training code) ────────────────────────
LDASN_IMG_SIZE = 224
LDASN_D_MODEL = 256
LDASN_N_HEADS = 4
LDASN_N_LAYERS = 4
LDASN_TOP_K = 50
LDASN_PATCH_SIZE = 16

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size, padding=padding, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))

class SEBlock(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, max(ch // r, 4)), nn.ReLU(),
            nn.Linear(max(ch // r, 4), ch), nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(x).view(x.size(0), -1, 1, 1)

class MultiScaleExtractor(nn.Module):
    def __init__(self, in_ch=3, out_ch=128):
        super().__init__()
        mid = out_ch // 2
        self.stem = nn.Sequential(nn.Conv2d(in_ch, 32, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32), nn.GELU())
        self.scale1 = nn.Sequential(DepthwiseSeparableConv(32, 64, 3, 1), DepthwiseSeparableConv(64, mid, 3, 1))
        self.scale2 = nn.Sequential(DepthwiseSeparableConv(32, 64, 5, 2), DepthwiseSeparableConv(64, mid, 5, 2))
        self.merge_se = SEBlock(out_ch)
        self.proj = nn.Sequential(nn.Conv2d(out_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.GELU())
        self.shallow = nn.Sequential(DepthwiseSeparableConv(32, out_ch, 1, 0))
    def forward(self, x):
        s = self.stem(x)
        s1 = self.scale1(s); s2 = self.scale2(s)
        merged = self.merge_se(torch.cat([s1, s2], dim=1))
        return self.proj(merged), self.shallow(s)

class SparsePatchSelector(nn.Module):
    def __init__(self, in_ch, d_model, patch_size, top_k):
        super().__init__()
        self.patch_size = patch_size; self.top_k = top_k
        self.saliency = nn.Conv2d(in_ch, 1, 1)
        self.proj = nn.Linear(in_ch * patch_size * patch_size, d_model)
        n_patches = (LDASN_IMG_SIZE // 2 // patch_size) ** 2
        self.pos_emb = nn.Embedding(n_patches, d_model)
        self.register_buffer('pos_ids', torch.arange(n_patches))
    def forward(self, deep, shallow):
        B, C, H, W = deep.shape; P = self.patch_size
        sal = self.saliency(deep); n_h, n_w = H // P, W // P
        sal_flat = F.avg_pool2d(sal, P).view(B, -1)
        k = min(self.top_k, sal_flat.shape[1])
        _, top_idx = sal_flat.topk(k, dim=1)
        feat = deep + shallow
        feat_unf = feat.unfold(2, P, P).unfold(3, P, P).contiguous().view(B, C, n_h*n_w, P*P)
        feat_unf = feat_unf.permute(0, 2, 1, 3).contiguous().view(B, n_h*n_w, C*P*P)
        idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, feat_unf.shape[-1])
        tokens = self.proj(torch.gather(feat_unf, 1, idx_exp))
        pos_e = self.pos_emb(self.pos_ids).unsqueeze(0).expand(B, -1, -1)
        idx_pos = top_idx.unsqueeze(-1).expand(-1, -1, LDASN_D_MODEL)
        tokens = tokens + torch.gather(pos_e, 1, idx_pos)
        ch_attn = self.saliency(deep).sigmoid().mean(dim=[2, 3])
        return tokens, top_idx, ch_attn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        mlp_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d_model, mlp_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(mlp_dim, d_model), nn.Dropout(dropout))
    def forward(self, x, ch_attn_mask=None):
        norm_x = self.norm1(x); attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        if ch_attn_mask is not None: attn_out = attn_out * ch_attn_mask.unsqueeze(-1)
        x = x + attn_out; return x + self.mlp(self.norm2(x))

class TinyTransformer(nn.Module):
    def __init__(self, d_model, n_heads, n_layers):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
    def forward(self, tokens, ch_attn):
        B = tokens.shape[0]; x = torch.cat([self.cls_token.expand(B, -1, -1), tokens], dim=1)
        for blk in self.blocks: x = blk(x, ch_attn)
        return self.norm(x[:, 0])

class TemperatureScaledHead(nn.Module):
    def __init__(self, d_model, num_classes):
        super().__init__()
        self.fc = nn.Linear(d_model, num_classes)
        self.temperature = nn.Parameter(torch.ones(1))
    def forward(self, x): return self.fc(x) / self.temperature.clamp(min=0.1)

class LDASN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        C = 128
        self.extractor = MultiScaleExtractor(in_ch=3, out_ch=C)
        self.selector = SparsePatchSelector(in_ch=C, d_model=LDASN_D_MODEL, patch_size=LDASN_PATCH_SIZE, top_k=LDASN_TOP_K)
        self.transformer = TinyTransformer(LDASN_D_MODEL, LDASN_N_HEADS, LDASN_N_LAYERS)
        self.head = TemperatureScaledHead(LDASN_D_MODEL, num_classes)
    def forward(self, x):
        deep, shallow = self.extractor(x)
        tokens, top_idx, ch_attn = self.selector(deep, shallow)
        cls_feat = self.transformer(tokens, ch_attn)
        return self.head(cls_feat)

# ─── Model Loading ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model(arch_key, model_path, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if arch_key == "LDASN":
        model = LDASN(num_classes)
    elif arch_key == "ConvNeXt_T":
        model = models.convnext_tiny(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    else:
        raise ValueError(f"Unknown: {arch_key}")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.to(device).eval()
    return model, device

def predict(model, image, device, class_names, img_size=224):
    tf = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    start = time.time()
    with torch.no_grad(): probs = F.softmax(model(tf(image).unsqueeze(0).to(device)), 1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    return {"class":class_names[idx],"index":idx,"confidence":float(probs[idx]),"probabilities":{cn:float(probs[i]) for i,cn in enumerate(class_names)},"inference_time_ms":(time.time()-start)*1000}

# ─── AI Chatbot (Groq) ────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Cotton Guard Assistant — an AI expert on cotton leaf diseases for Pakistani cotton farmers.
APP: Cotton Guard detects cotton leaf diseases via deep learning. Users upload leaf photos for instant diagnosis.
MODELS: SAR-CLD 2024 (7 classes, LDASN, 224x224, ~98.4% acc) | Cotton Leaf Disease (4 classes, ConvNeXt-T, 224x224, ~97.7% acc).
TRAINING: 80/20 split, Focal Loss, AdamW, Cosine LR, augmentation, early stopping.
DISEASES: Bacterial Blight (High), Curl Virus/CLCuV (Very High), Fussarium Wilt (High), Herbicide Damage (Medium), Jassids (Medium-High), Leaf Redding (Medium), Leaf Variegation (Medium).
RULES: Only answer about cotton diseases/farming/this app. Redirect unrelated questions politely. Be concise, farmer-friendly. ALWAYS respond in English by default. Only respond in Urdu/Roman Urdu if the user explicitly asks you to write in Urdu or writes their message in Urdu. Never make up info."""

def get_ai_response(user_msg, chat_history):
    import requests
    api_key = st.secrets.get("GROQ_API_KEY", "")
    if not api_key: return "Please add GROQ_API_KEY to Streamlit secrets."
    messages = [{"role":"system","content":SYSTEM_PROMPT}] + [{"role":m["role"],"content":m["content"]} for m in chat_history[-10:]] + [{"role":"user","content":user_msg}]
    try:
        r = requests.post("https://api.groq.com/openai/v1/chat/completions", json={"model":"llama-3.3-70b-versatile","messages":messages,"temperature":0.7,"max_tokens":500},
            headers={"Authorization":f"Bearer {api_key}","Content-Type":"application/json"}, timeout=15).json()
        return r["choices"][0]["message"]["content"] if "choices" in r else f"Error: {r.get('error',{}).get('message','Unknown')}"
    except: return "Connection error. Please try again."

# ╔═══════════════════════ SIDEBAR — CHATBOT ═══════════════════════════════╗
with st.sidebar:
    st.markdown("""<div style="text-align:center;padding:0.5rem 0 0.8rem;"><span style="font-size:2rem;">🍃</span>
    <h2 style="font-family:'DM Serif Display',serif;color:#e8f0e0;margin:0.2rem 0 0;">Cotton Guard</h2>
    <p style="color:#a8b89e;font-size:0.78rem;margin:0;">AI Assistant</p></div>
    <hr style="border:none;border-top:1px solid #3d4e2e;margin:0 0 1rem;">""", unsafe_allow_html=True)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role":"assistant","content":"Assalam o Alaikum! 👋\n\nI'm your Cotton Guard Assistant. I can help with:\n\n🌿 Disease identification\n💊 Treatment advice\n🔬 How this app works\n\nAsk me anything about cotton crops!"}]
    chat_box = st.container(height=400)
    with chat_box:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"], avatar="🌿" if msg["role"]=="assistant" else "👤"):
                st.markdown(msg["content"])
    user_input = st.chat_input("Ask about cotton diseases...", key="chat_input")
    if user_input:
        st.session_state.chat_history.append({"role":"user","content":user_input})
        with st.spinner("Thinking..."): response = get_ai_response(user_input, st.session_state.chat_history)
        st.session_state.chat_history.append({"role":"assistant","content":response}); st.rerun()

# ╔═══════════════════════ MAIN AREA ═══════════════════════════════════════╗
st.markdown("""<div class="app-header"><h1>🍃 Cotton Guard</h1><p class="subtitle">Deep Learning Cotton Leaf Disease Detection for Farmers</p></div><div class="header-divider"></div>""", unsafe_allow_html=True)

st.markdown('<div class="earth-card"><div class="earth-card-header">📋 Select Dataset & Model</div>', unsafe_allow_html=True)
dataset_choice = st.selectbox("Dataset", list(DATASET_INFO.keys()), label_visibility="collapsed")
ds = DATASET_INFO[dataset_choice]
st.markdown(f"""<div class="metric-row">
<div class="metric-card"><div class="metric-label">Architecture</div><div class="metric-value">{ds['architecture'].split('(')[0].strip()}</div></div>
<div class="metric-card"><div class="metric-label">Classes</div><div class="metric-value">{len(ds['classes'])}</div></div>
<div class="metric-card"><div class="metric-label">Input Size</div><div class="metric-value">{ds['img_size']}×{ds['img_size']}</div></div>
<div class="metric-card"><div class="metric-label">Normalization</div><div class="metric-value">ImageNet</div></div>
</div></div>""", unsafe_allow_html=True)

col_up, col_prev = st.columns([1,1])
with col_up:
    st.markdown('<div class="earth-card"><div class="earth-card-header">📷 Upload Cotton Leaf</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload", type=["jpg","jpeg","png","bmp","webp"], label_visibility="collapsed")
    st.markdown('<p style="color:#7a8b6e;font-size:0.75rem;text-align:center;margin-top:0.5rem;">Upload a clear close-up photo of a single cotton leaf</p></div>', unsafe_allow_html=True)
with col_prev:
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.markdown('<div class="earth-card"><div class="earth-card-header">🖼️ Preview</div>', unsafe_allow_html=True)
        st.image(image, caption=uploaded_file.name, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

analyze = st.button("🔬  Analyze Leaf", use_container_width=True, type="primary")

if analyze and uploaded_file:
    try:
        with st.spinner("🌿 Analyzing your cotton leaf..."):
            model, device = load_model(ds["arch_key"], ds["model_file"], len(ds["classes"]))
            result = predict(model, image, device, ds["classes"], ds["img_size"])
        pc, conf = result["class"], result["confidence"]
        healthy = pc in ["Healthy","Healthy Leaf"]
        bc = "prediction-box" if healthy else "disease-box"
        lc = "prediction-label" if healthy else "disease-label"
        ic = "✅" if healthy else "⚠️"
        st.markdown(f'<div class="{bc}"><div class="{lc}">{ic} Prediction Result</div><div class="prediction-name">{pc}</div><div class="prediction-index">Class Index: {result["index"]} · Confidence: {conf*100:.1f}%</div></div>', unsafe_allow_html=True)
        c1, c2 = st.columns([1,2])
        with c1:
            cc = "#2d5016" if conf>0.8 else "#8b6914" if conf>0.5 else "#a83232"
            st.markdown(f'<div class="earth-card"><div class="earth-card-header">Confidence</div><div class="confidence-section"><div class="confidence-pct" style="color:{cc}">{conf*100:.1f}%</div></div>', unsafe_allow_html=True)
            st.progress(conf); st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="earth-card"><div class="earth-card-header">Class Probabilities</div>', unsafe_allow_html=True)
            for cn,p in sorted(result["probabilities"].items(), key=lambda x:-x[1]):
                bc2 = "#2d5016" if cn==pc else "#8b9e78"
                st.markdown(f'<div class="prob-item"><div class="prob-name">{cn}</div><div class="prob-bar-bg"><div class="prob-bar-fill" style="width:{p*100:.1f}%;background:{bc2}"></div></div><div class="prob-pct">{p*100:.1f}%</div></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="earth-card"><div class="earth-card-header">⚡ Performance</div><div class="metric-row"><div class="metric-card"><div class="metric-label">Inference</div><div class="metric-value">{result["inference_time_ms"]:.1f}ms</div></div><div class="metric-card"><div class="metric-label">Model</div><div class="metric-value">{ds["arch_key"]}</div></div><div class="metric-card"><div class="metric-label">Original</div><div class="metric-value">{image.size[0]}×{image.size[1]}</div></div><div class="metric-card"><div class="metric-label">Dataset</div><div class="metric-value">{dataset_choice.split("—")[0].strip()}</div></div></div></div>', unsafe_allow_html=True)
        if pc in DISEASE_INFO:
            info = DISEASE_INFO[pc]
            if not healthy:
                st.markdown(f'<div class="earth-card"><div class="earth-card-header">🔬 Disease Information & Treatment</div></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="info-card info-card-warn"><h4>{info.get("icon","")} Severity: {info["severity"]}</h4><p>{info["description"]}</p></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="info-card"><h4>🔍 Symptoms</h4><p>{info["symptoms"]}</p></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="info-card"><h4>💊 Treatment</h4><p>{info["treatment"]}</p></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="info-card"><h4>🛡️ Prevention</h4><p>{info["prevention"]}</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="info-card"><h4>✅ Your cotton leaf looks healthy!</h4><p>{info["description"]}</p></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="info-card"><h4>🌱 Maintenance Tips</h4><p>{info["prevention"]}</p></div>', unsafe_allow_html=True)
    except FileNotFoundError: st.error(f"Model not found: `{ds['model_file']}`")
    except Exception as e: st.error(f"Error: {str(e)}")
elif analyze and not uploaded_file:
    st.warning("Please upload a cotton leaf image first.")

# ╔═══════════════════════ XAI SECTION ═════════════════════════════════════╗

def get_xai_transform(img_size=224):
    return transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

def get_raw_image(image, img_size=224):
    """Get normalized numpy image for overlay."""
    tf = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    return tf(image).permute(1, 2, 0).numpy()

def compute_saliency(model, image, device, img_size=224):
    """Compute vanilla gradient saliency map."""
    tf = get_xai_transform(img_size)
    inp = tf(image).unsqueeze(0).to(device)
    inp.requires_grad_(True)
    model.zero_grad()
    logits = model(inp)
    pred_class = logits.argmax(1).item()
    logits[0, pred_class].backward()
    saliency = inp.grad.data.abs().squeeze().cpu()
    saliency = saliency.max(dim=0)[0]  # max across channels
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency.numpy()

def compute_gradcam(model, image, device, img_size=224, target_layer=None):
    """Compute GradCAM heatmap."""
    activations = {}
    gradients = {}
    def fwd_hook(m, i, o): activations['val'] = o
    def bwd_hook(m, i, o): gradients['val'] = o[0]

    handle_f = target_layer.register_forward_hook(fwd_hook)
    handle_b = target_layer.register_full_backward_hook(bwd_hook)

    tf = get_xai_transform(img_size)
    inp = tf(image).unsqueeze(0).to(device)
    inp.requires_grad_(True)
    model.zero_grad()
    logits = model(inp)
    pred_class = logits.argmax(1).item()
    logits[0, pred_class].backward()

    act = activations['val'].detach()
    grad = gradients['val'].detach()
    weights = grad.mean(dim=[2, 3], keepdim=True)
    cam = (weights * act).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=(img_size, img_size), mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    handle_f.remove()
    handle_b.remove()
    return cam

def compute_gradcam_pp(model, image, device, img_size=224, target_layer=None):
    """Compute GradCAM++ heatmap."""
    activations = {}
    gradients = {}
    def fwd_hook(m, i, o): activations['val'] = o
    def bwd_hook(m, i, o): gradients['val'] = o[0]

    handle_f = target_layer.register_forward_hook(fwd_hook)
    handle_b = target_layer.register_full_backward_hook(bwd_hook)

    tf = get_xai_transform(img_size)
    inp = tf(image).unsqueeze(0).to(device)
    inp.requires_grad_(True)
    model.zero_grad()
    logits = model(inp)
    pred_class = logits.argmax(1).item()
    logits[0, pred_class].backward()

    act = activations['val'].detach()
    grad = gradients['val'].detach()

    # GradCAM++ weighting
    grad_pow2 = grad ** 2
    grad_pow3 = grad ** 3
    sum_act = act.sum(dim=[2, 3], keepdim=True)
    alpha = grad_pow2 / (2 * grad_pow2 + sum_act * grad_pow3 + 1e-8)
    alpha = alpha * F.relu(grad)
    weights = alpha.sum(dim=[2, 3], keepdim=True)

    cam = (weights * act).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=(img_size, img_size), mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    handle_f.remove()
    handle_b.remove()
    return cam

def compute_lime(model, image, device, class_names, img_size=224, num_samples=100):
    """Compute LIME explanation."""
    from skimage.segmentation import quickshift
    
    tf = get_xai_transform(img_size)
    raw_tf = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    img_np = raw_tf(image).permute(1, 2, 0).numpy()
    
    # Segment image into superpixels
    segments = quickshift(img_np, kernel_size=4, max_dist=200, ratio=0.2)
    n_segments = len(np.unique(segments))
    
    # Get original prediction
    inp = tf(image).unsqueeze(0).to(device)
    with torch.no_grad():
        orig_probs = F.softmax(model(inp), dim=1).cpu().numpy()[0]
    pred_class = orig_probs.argmax()
    
    # Generate perturbed samples
    np.random.seed(42)
    perturbations = np.random.binomial(1, 0.5, size=(num_samples, n_segments))
    perturbations[0] = np.ones(n_segments)  # include original
    
    predictions = []
    for pert in perturbations:
        masked = img_np.copy()
        for seg_id in np.unique(segments):
            if pert[seg_id] == 0:
                masked[segments == seg_id] = 0.5  # gray out
        masked_tensor = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])(
            torch.tensor(masked).permute(2, 0, 1).float()
        ).unsqueeze(0).to(device)
        with torch.no_grad():
            p = F.softmax(model(masked_tensor), dim=1).cpu().numpy()[0]
        predictions.append(p[pred_class])
    
    predictions = np.array(predictions)
    
    # Compute importance of each segment using correlation
    importance = np.zeros(n_segments)
    for seg_id in range(n_segments):
        mask_col = perturbations[:, seg_id]
        if mask_col.std() > 0:
            importance[seg_id] = np.corrcoef(mask_col, predictions)[0, 1]
    
    # Create heatmap
    lime_map = np.zeros((img_size, img_size))
    for seg_id in np.unique(segments):
        lime_map[segments == seg_id] = importance[seg_id]
    
    # Normalize to 0-1
    lime_map = (lime_map - lime_map.min()) / (lime_map.max() - lime_map.min() + 1e-8)
    return lime_map

def overlay_heatmap(raw_img, heatmap, colormap='jet', alpha=0.5):
    """Overlay heatmap on raw image."""
    import matplotlib.cm as cm
    cmap = cm.get_cmap(colormap)
    heatmap_colored = cmap(heatmap)[:, :, :3]
    overlay = (1 - alpha) * raw_img + alpha * heatmap_colored
    return np.clip(overlay, 0, 1)

def get_target_layer(model, arch_key):
    """Get the appropriate target layer for GradCAM."""
    if arch_key == "LDASN":
        return model.extractor.proj[0]  # last conv before BN in proj
    else:  # ConvNeXt
        return model.features[-1][-1].block[-1]  # last layer

# ─── Show XAI Results ─────────────────────────────────────────────────────
if analyze and uploaded_file and 'result' in dir():
    try:
        st.markdown('<div class="earth-card"><div class="earth-card-header">🧠 Explainable AI — Visual Explanations</div>', unsafe_allow_html=True)
        st.markdown('<p style="color:#5a6650;font-size:0.82rem;margin-bottom:1rem;">These maps show which regions of the leaf the model focuses on to make its prediction.</p>', unsafe_allow_html=True)

        model_xai, device_xai = load_model(ds["arch_key"], ds["model_file"], len(ds["classes"]))
        model_xai.train()  # enable gradients for hooks
        
        raw_img = get_raw_image(image, ds["img_size"])
        target_layer = get_target_layer(model_xai, ds["arch_key"])

        def get_focus_region(heatmap):
            h, w = heatmap.shape
            top = heatmap[:h//2, :].mean(); bottom = heatmap[h//2:, :].mean()
            left = heatmap[:, :w//2].mean(); right = heatmap[:, w//2:].mean()
            center = heatmap[h//4:3*h//4, w//4:3*w//4].mean()
            edges = (top + bottom + left + right) / 4
            regions = []
            if center > edges * 1.2: regions.append("center")
            if top > bottom * 1.3: regions.append("upper")
            if bottom > top * 1.3: regions.append("lower")
            if left > right * 1.3: regions.append("left")
            if right > left * 1.3: regions.append("right")
            if not regions: regions.append("spread across the entire leaf")
            return ", ".join(regions)

        def get_focus_intensity(heatmap):
            high = (heatmap > 0.7).mean() * 100
            if high > 30: return "strongly concentrated"
            elif high > 15: return "moderately focused"
            elif high > 5: return "lightly distributed"
            else: return "diffusely spread"

        # Generate all XAI maps
        with st.spinner("Generating Saliency Map..."):
            sal = compute_saliency(model_xai, image, device_xai, ds["img_size"])
            sal_overlay = overlay_heatmap(raw_img, sal, colormap='hot')
        with st.spinner("Generating GradCAM..."):
            gcam = compute_gradcam(model_xai, image, device_xai, ds["img_size"], target_layer)
            gcam_overlay = overlay_heatmap(raw_img, gcam)
        with st.spinner("Generating GradCAM++..."):
            gcpp = compute_gradcam_pp(model_xai, image, device_xai, ds["img_size"], target_layer)
            gcpp_overlay = overlay_heatmap(raw_img, gcpp, colormap='inferno')
        with st.spinner("Generating LIME..."):
            lime_map = compute_lime(model_xai, image, device_xai, ds["classes"], ds["img_size"])
            lime_overlay = overlay_heatmap(raw_img, lime_map, colormap='RdYlGn')

        # Row 1: All 4 images
        ic1, ic2, ic3, ic4 = st.columns(4)
        with ic1: st.image(sal_overlay, use_container_width=True, clamp=True)
        with ic2: st.image(gcam_overlay, use_container_width=True, clamp=True)
        with ic3: st.image(gcpp_overlay, use_container_width=True, clamp=True)
        with ic4: st.image(lime_overlay, use_container_width=True, clamp=True)

        # Row 2: All 4 titles
        tc1, tc2, tc3, tc4 = st.columns(4)
        with tc1: st.markdown('<p style="color:#2d5016;font-size:1rem;font-weight:700;text-align:center;">🔥 Saliency Map</p>', unsafe_allow_html=True)
        with tc2: st.markdown('<p style="color:#2d5016;font-size:1rem;font-weight:700;text-align:center;">🎯 GradCAM</p>', unsafe_allow_html=True)
        with tc3: st.markdown('<p style="color:#2d5016;font-size:1rem;font-weight:700;text-align:center;">🔬 GradCAM++</p>', unsafe_allow_html=True)
        with tc4: st.markdown('<p style="color:#2d5016;font-size:1rem;font-weight:700;text-align:center;">🧩 LIME</p>', unsafe_allow_html=True)

        # Compute stats
        sal_region = get_focus_region(sal); sal_intensity = get_focus_intensity(sal); sal_coverage = (sal > 0.5).mean() * 100
        gcam_region = get_focus_region(gcam); gcam_intensity = get_focus_intensity(gcam); gcam_coverage = (gcam > 0.5).mean() * 100
        gcpp_region = get_focus_region(gcpp); gcpp_intensity = get_focus_intensity(gcpp); gcpp_coverage = (gcpp > 0.5).mean() * 100
        lime_region = get_focus_region(lime_map); lime_positive = (lime_map > 0.6).mean() * 100; lime_negative = (lime_map < 0.3).mean() * 100

        # Row 3: All 4 descriptions
        dc1, dc2, dc3, dc4 = st.columns(4)
        with dc1:
            st.markdown(f'<div class="info-card"><p>Computes raw input gradients to show which pixels most influence the prediction of <b>{pc}</b>. Activation is <b>{sal_intensity}</b> in the <b>{sal_region}</b> region, covering <b>{sal_coverage:.1f}%</b> of the image. Bright spots indicate high-impact pixels.</p></div>', unsafe_allow_html=True)
        with dc2:
            st.markdown(f'<div class="info-card"><p>Highlights which leaf regions contribute most to detecting <b>{pc}</b> using gradient-weighted activations. Attention is <b>{gcam_intensity}</b> on the <b>{gcam_region}</b> area, covering <b>{gcam_coverage:.1f}%</b>. Warm colors show disease-relevant features.</p></div>', unsafe_allow_html=True)
        with dc3:
            st.markdown(f'<div class="info-card"><p>Enhanced GradCAM with pixel-wise weighting for capturing <b>multiple disease instances</b>. For <b>{pc}</b>, focus is <b>{gcpp_intensity}</b> in the <b>{gcpp_region}</b> portion, covering <b>{gcpp_coverage:.1f}%</b>. Detects scattered symptoms better.</p></div>', unsafe_allow_html=True)
        with dc4:
            st.markdown(f'<div class="info-card"><p>Tests which superpixel regions are essential for predicting <b>{pc}</b>. Green areas (<b>{lime_positive:.1f}%</b>) support the diagnosis, red areas (<b>{lime_negative:.1f}%</b>) oppose it. Model relies on the <b>{lime_region}</b> area.</p></div>', unsafe_allow_html=True)

        model_xai.eval()
        st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"XAI visualization unavailable: {str(e)}")
