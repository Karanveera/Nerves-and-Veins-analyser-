import cv2, numpy as np, streamlit as st
from PIL import Image
from vein_detector import process

st.set_page_config(page_title="Vein Detector", layout="wide")
st.title("💪 Arm / Hand Vein Detector")
st.caption("Upload an arm or hand photo – veins are traced and marked with coloured, numbered lines.")

import glob, os, time, io
mode = st.sidebar.radio("Image source", ["Webcam (live camera + capture)", "Upload", "Cloud (latest photo from camera)", "Camera IP address (live)", "Camera folder (same computer)"])
f = None
auto = False
if mode.startswith("Webcam"):
    f = st.camera_input("Allow the camera, then click \"Take Photo\"")
elif mode == "Upload":
    f = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp"])
elif mode.startswith("Camera IP"):
    from camera_ip import grab_frame, is_private
    cam_url = st.sidebar.text_input("Camera photo/stream URL", "http://192.168.1.50/capture")
    auto = st.sidebar.checkbox("Auto-refresh every 3 s", True)
    if is_private(cam_url) and os.environ.get("HOME", "") == "/home/appuser":
        st.error("This is a private (local) IP like 192.168.x.x. A hosted website cannot reach it. "
                 "Run this app on the same Wi-Fi as the camera, or make the camera public (see README).")
    else:
        try:
            f = io.BytesIO(grab_frame(cam_url))
        except Exception as e:
            st.error(f"Cannot reach camera: {e}")
elif mode.startswith("Cloud"):
    import requests
    auto = st.sidebar.checkbox("Auto-refresh every 3 s", True)
    try:
        U, K, B = st.secrets["SUPABASE_URL"].rstrip("/"), st.secrets["SUPABASE_KEY"], st.secrets.get("SUPABASE_BUCKET", "photos")
        H = {"apikey": K, "Authorization": f"Bearer {K}"}
        r = requests.post(f"{U}/storage/v1/object/list/{B}", headers=H, timeout=20,
                          json={"prefix": "", "limit": 1, "sortBy": {"column": "name", "order": "desc"}})
        items = [i for i in r.json() if i.get("name")]
        if items:
            name = items[0]["name"]
            st.info(f"Latest photo: {name}")
            f = io.BytesIO(requests.get(f"{U}/storage/v1/object/{B}/{name}", headers=H, timeout=30).content)
        else:
            st.warning("No photos uploaded yet.")
    except Exception as e:
        st.error(f"Cloud storage not reachable / not configured: {e}")
else:
    folder = st.sidebar.text_input("Folder where the camera saves photos", "photos")
    auto = st.sidebar.checkbox("Auto-refresh every 3 s", True)
    files = [p for e in ("jpg", "jpeg", "png", "webp", "bmp") for p in glob.glob(os.path.join(folder, f"*.{e}"))]
    if files:
        latest = max(files, key=os.path.getmtime)
        st.info(f"Latest photo: {os.path.basename(latest)}")
        f = io.BytesIO(open(latest, "rb").read())
    else:
        st.warning("No photos found in that folder yet.")
sens = st.sidebar.slider("Sensitivity (more veins →)", 0.0, 1.0, 0.8, 0.05)
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

if auto:
    time.sleep(3)
    st.rerun()
