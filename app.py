"""
app.py
======
Streamlit App — Indonesian Food Calorie Estimator
Model di-load otomatis dari Hugging Face Hub.

Run: streamlit run app.py
"""

import os
import json
import datetime
import numpy as np
import streamlit as st
from PIL import Image
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🍽️ Indonesian Food Calorie Estimator",
    page_icon="🍽️",
    layout="wide"
)

# ── CONFIG ────────────────────────────────────────────────────────────────────
HF_REPO_ID = "friscaocta/indonesian-food-calorie-estimator"
MODEL_FILE = "food_classifier.h5"
CLASS_FILE = "class_indices.json"
IMG_SIZE   = (224, 224)

# ── Nutrition Database ─────────────────────────────────────────────────────────
NUTRITION_DB = {
    "Nasi Goreng"   : {"kalori": 260, "protein": 8.0,  "karbohidrat": 35.0, "lemak": 10.0},
    "Gado-Gado"     : {"kalori": 180, "protein": 9.0,  "karbohidrat": 18.0, "lemak": 8.0},
    "Rendang"       : {"kalori": 320, "protein": 25.0, "karbohidrat": 5.0,  "lemak": 22.0},
    "Soto Ayam"     : {"kalori": 150, "protein": 12.0, "karbohidrat": 12.0, "lemak": 5.0},
    "Mie Goreng"    : {"kalori": 300, "protein": 9.0,  "karbohidrat": 42.0, "lemak": 11.0},
    "Bakso"         : {"kalori": 210, "protein": 14.0, "karbohidrat": 20.0, "lemak": 8.0},
    "Ayam Goreng"   : {"kalori": 290, "protein": 27.0, "karbohidrat": 8.0,  "lemak": 17.0},
    "Lontong Sayur" : {"kalori": 200, "protein": 6.0,  "karbohidrat": 32.0, "lemak": 6.0},
    "Pempek"        : {"kalori": 195, "protein": 10.0, "karbohidrat": 28.0, "lemak": 5.0},
    "Ketoprak"      : {"kalori": 220, "protein": 9.0,  "karbohidrat": 30.0, "lemak": 8.0},
}

# ── Load Model dari Hugging Face Hub ──────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Loading model dari Hugging Face...")
def load_resources():
    import tensorflow as tf
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILE)
    class_path = hf_hub_download(repo_id=HF_REPO_ID, filename=CLASS_FILE)

    model = tf.keras.models.load_model(model_path)
    with open(class_path, "r") as f:
        class_indices = json.load(f)  # {"0": "Ayam Goreng", "1": "Bakso", ...}

    return model, class_indices

# ── Predict ────────────────────────────────────────────────────────────────────
def predict(image: Image.Image, model, class_indices: dict, top_k=3):
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds       = model.predict(arr, verbose=0)[0]
    top_indices = np.argsort(preds)[::-1][:top_k]

    best_label = class_indices.get(str(int(top_indices[0])), "Unknown")
    confidence = float(preds[top_indices[0]])
    nutrition  = NUTRITION_DB.get(best_label, {})

    top_predictions = [
        {"label": class_indices.get(str(int(i)), "?"), "confidence": float(preds[i])}
        for i in top_indices
    ]

    return {
        "predicted_class" : best_label,
        "confidence"      : confidence,
        "nutrition"       : nutrition,
        "top_predictions" : top_predictions,
    }

# ── Session State ──────────────────────────────────────────────────────────────
if "food_log" not in st.session_state:
    st.session_state.food_log = []

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Flag_of_Indonesia.svg/320px-Flag_of_Indonesia.svg.png", width=80)
    st.title("🍽️ Calorie Estimator")
    st.markdown("Deteksi makanan Indonesia dari foto dan estimasi kalorinya.")
    st.divider()

    daily_target = st.number_input("🎯 Target Kalori Harian (kcal)", min_value=1000, max_value=5000, value=2000, step=50)

    st.divider()
    st.markdown("**📋 Food Log Hari Ini**")
    total_cal = sum(item["kalori"] for item in st.session_state.food_log)

    if st.session_state.food_log:
        for i, item in enumerate(st.session_state.food_log):
            col1, col2 = st.columns([3, 1])
            col1.markdown(f"• {item['label']} — **{item['kalori']} kcal**")
            if col2.button("✕", key=f"del_{i}"):
                st.session_state.food_log.pop(i)
                st.rerun()
        st.divider()
        st.metric("Total Kalori", f"{total_cal} kcal", f"{total_cal - daily_target} dari target")
    else:
        st.info("Belum ada makanan yang dicatat.")

    if st.button("🗑️ Reset Log", use_container_width=True):
        st.session_state.food_log = []
        st.rerun()

# ── Main ───────────────────────────────────────────────────────────────────────
st.title("🍽️ Indonesian Food Calorie Estimator")
st.markdown("Upload foto makanan Indonesia — sistem akan mendeteksi jenis makanan dan menampilkan informasi nutrisinya secara otomatis.")

tab1, tab2, tab3 = st.tabs(["📷 Deteksi Makanan", "📊 Daily Tracker", "ℹ️ Tentang Model"])

# TAB 1 ────────────────────────────────────────────────────────────────────────
with tab1:
    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.subheader("📤 Upload Foto Makanan")
        uploaded = st.file_uploader("Pilih gambar (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if uploaded:
            image = Image.open(uploaded)
            st.image(image, caption="Foto yang diupload", use_column_width=True)

    with col_result:
        st.subheader("🔍 Hasil Deteksi")
        if uploaded:
            with st.spinner("Menganalisis gambar..."):
                try:
                    model, class_indices = load_resources()
                    result = predict(image, model, class_indices)

                    food_name  = result["predicted_class"]
                    confidence = result["confidence"]
                    nutrition  = result["nutrition"]
                    conf_pct   = confidence * 100

                    badge = "🟢" if conf_pct >= 70 else "🟡" if conf_pct >= 50 else "🔴"
                    st.markdown(f"### {badge} {food_name}")
                    st.progress(confidence, text=f"Kepercayaan: {conf_pct:.1f}%")
                    st.divider()

                    if nutrition:
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("🔥 Kalori",      f"{nutrition.get('kalori', '-')} kcal")
                        c2.metric("💪 Protein",     f"{nutrition.get('protein', '-')} g")
                        c3.metric("🌾 Karbohidrat", f"{nutrition.get('karbohidrat', '-')} g")
                        c4.metric("🫙 Lemak",       f"{nutrition.get('lemak', '-')} g")

                        fig = go.Figure(go.Pie(
                            labels=["Protein", "Karbohidrat", "Lemak"],
                            values=[nutrition.get("protein", 0), nutrition.get("karbohidrat", 0), nutrition.get("lemak", 0)],
                            hole=0.4,
                            marker_colors=["#4CAF50", "#2196F3", "#FF9800"]
                        ))
                        fig.update_layout(title="Komposisi Makro Nutrisi", height=280, margin=dict(t=40, b=0))
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("⚠️ Data nutrisi untuk makanan ini belum tersedia.")

                    with st.expander("🔎 Lihat Prediksi Lainnya"):
                        for p in result["top_predictions"]:
                            st.markdown(f"- **{p['label']}** — {p['confidence']*100:.1f}%")

                    st.divider()
                    if st.button("➕ Tambahkan ke Food Log", use_container_width=True, type="primary"):
                        st.session_state.food_log.append({
                            "label"       : food_name,
                            "kalori"      : nutrition.get("kalori", 0),
                            "protein"     : nutrition.get("protein", 0),
                            "karbohidrat" : nutrition.get("karbohidrat", 0),
                            "lemak"       : nutrition.get("lemak", 0),
                            "waktu"       : datetime.datetime.now().strftime("%H:%M")
                        })
                        st.success(f"✅ {food_name} ditambahkan ke food log!")
                        st.rerun()

                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    st.info("Pastikan HF_REPO_ID di baris 20 sudah diubah ke username Hugging Face kamu!")
        else:
            st.info("⬆️ Upload foto makanan untuk memulai deteksi.")

# TAB 2 ────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("📊 Daily Calorie Tracker")

    total_cal = sum(item["kalori"] for item in st.session_state.food_log)
    remaining = daily_target - total_cal

    col_gauge, col_info = st.columns([1, 1])
    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=total_cal,
            delta={"reference": daily_target},
            title={"text": "Kalori Hari Ini (kcal)"},
            gauge={
                "axis": {"range": [0, daily_target * 1.3]},
                "bar":  {"color": "#4CAF50" if total_cal <= daily_target else "#f44336"},
                "steps": [
                    {"range": [0, daily_target * 0.5],         "color": "#e8f5e9"},
                    {"range": [daily_target * 0.5, daily_target], "color": "#fff9c4"},
                    {"range": [daily_target, daily_target * 1.3], "color": "#ffebee"}
                ],
                "threshold": {"line": {"color": "red", "width": 3}, "value": daily_target}
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(t=40, b=0))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_info:
        st.metric("🎯 Target",     f"{daily_target} kcal")
        st.metric("🔥 Dikonsumsi", f"{total_cal} kcal")
        st.metric("⚖️ Sisa",       f"{remaining} kcal")

        if total_cal == 0:
            st.info("Belum ada makanan dicatat hari ini.")
        elif total_cal < daily_target * 0.5:
            st.warning("⚠️ Asupan kalori masih sangat kurang!")
        elif total_cal <= daily_target:
            st.success(f"✅ Bagus! Kamu masih bisa mengonsumsi {remaining} kcal lagi.")
        else:
            st.error(f"🚨 Kalori melebihi target sebesar {total_cal - daily_target} kcal!")

    if st.session_state.food_log:
        st.divider()
        st.markdown("#### 📋 Rincian Makanan")
        df = pd.DataFrame(st.session_state.food_log)
        st.dataframe(df[["waktu", "label", "kalori", "protein", "karbohidrat", "lemak"]].rename(columns={
            "waktu": "Waktu", "label": "Makanan", "kalori": "Kalori (kcal)",
            "protein": "Protein (g)", "karbohidrat": "Karbo (g)", "lemak": "Lemak (g)"
        }), use_container_width=True)

        fig_bar = px.bar(df, x="label", y="kalori", color="kalori",
                         color_continuous_scale="RdYlGn_r",
                         labels={"label": "Makanan", "kalori": "Kalori (kcal)"},
                         title="Kalori per Makanan")
        st.plotly_chart(fig_bar, use_container_width=True)

# TAB 3 ────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("ℹ️ Tentang Model")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
**🏗️ Arsitektur**
- Base Model: **MobileNetV2** (pretrained ImageNet)
- Teknik: **Transfer Learning** + Fine-tuning
- Input: 224 × 224 px RGB

**📦 Tech Stack**
- Python, TensorFlow/Keras
- Streamlit (UI)
- Plotly (visualisasi)
- Hugging Face Hub (model hosting)
        """)
    with col_b:
        st.markdown("""
**🍛 Kelas Makanan**
- Ayam Goreng, Bakso, Gado-Gado
- Ketoprak, Lontong Sayur, Mie Goreng
- Nasi Goreng, Pempek, Rendang, Soto Ayam

**📊 Metrik Evaluasi**
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
        """)
    st.divider()
    st.markdown("""
**🔬 Metodologi**
1. **Dataset**: rzyuanda/indonesian_food_v2 dari Hugging Face (350 gambar, 10 kelas)
2. **Preprocessing**: Resize 224×224, normalisasi pixel, augmentasi data
3. **Training Phase 1**: Freeze base model, train classification head (15 epoch)
4. **Training Phase 2**: Fine-tune 50 layer terakhir MobileNetV2 (25 epoch)
5. **Evaluasi**: Classification report + confusion matrix pada test set
    """)
