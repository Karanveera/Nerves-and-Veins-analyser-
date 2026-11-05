import cv2, numpy as np, streamlit as st
from PIL import Image
from vein_detector import process

st.set_page_config(page_title="Vein Detector", layout="wide")
st.title("💪 Arm / Hand Vein Detector")
st.caption("Upload an arm or hand photo – veins are traced and marked with coloured, numbered lines.")

f = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp"])
sens = st.sidebar.slider("Sensitivity (more veins →)", 0.0, 1.0, 0.7, 0.05)
minlen = st.sidebar.slider("Minimum vein length (px)", 10, 150, 35)
thick = st.sidebar.slider("Line thickness", 1, 6, 2)
tags = st.sidebar.checkbox("Show vein labels", True)
hand = st.sidebar.selectbox("Where are the fingers / fist?", ["auto", "top", "bottom", "left", "right", "none"], index=0)
hand = None if hand == "none" else hand
hfrac = st.sidebar.slider("Fingers area size (%)", 10, 50, 28) / 100
use_ign = st.sidebar.checkbox("Ignore a region (e.g. fingers/knuckles)")
ign = None
if use_ign:
    x = st.sidebar.slider("X range (%)", 0, 100, (50, 100))
    y = st.sidebar.slider("Y range (%)", 0, 100, (10, 40))
    ign = (x[0]/100, y[0]/100, x[1]/100, y[1]/100)

if f:
    img = cv2.cvtColor(np.array(Image.open(f).convert("RGB")), cv2.COLOR_RGB2BGR)
    orig, marked, n = process(img, sens, minlen, thick, tags, ign, hand, hfrac)
    a, b = st.columns(2)
    a.image(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
    b.image(cv2.cvtColor(marked, cv2.COLOR_BGR2RGB), caption=f"Predicted veins: {n}", use_container_width=True)
    ok, buf = cv2.imencode(".png", marked)
    st.download_button("Download marked image", buf.tobytes(), "veins_marked.png", "image/png")
