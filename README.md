# Vein Detector
Upload an arm/hand image → veins are detected and drawn as coloured, numbered lines.

## Run
    pip install -r requirements.txt
    streamlit run app.py          # web UI with upload + sliders
    python vein_detector.py arm.jpg -o out.png -s 0.5 --ignore 0.5 0.1 1 0.4   # CLI

## How it works
1. Skin mask (YCrCb) removes background.
2. Green channel + CLAHE, subtract blurred copy → thin dark structures.
3. Multi-scale Frangi ridge filter → tubular (vein-like) response.
4. Hysteresis threshold, skeletonize, drop short/blobby pieces.
5. Each vein is drawn in its own colour with a label (V1, V2 …).

Tips: raise Sensitivity for faint veins; use "Ignore region" to skip knuckle/finger creases.
Classical CV, not a trained model – for best accuracy use well-lit photos of veiny arms.

## Using photos from a camera / ESP32-CAM
**A) Photos are saved to a folder** (SD card copy, shared folder, etc.):

    python watch_folder.py --input "C:/esp_photos" --output "C:/esp_photos/marked"

Every new photo is marked automatically and saved as `<name>_marked.png` in the output folder.

**B) Pull photos straight from the ESP32-CAM over Wi-Fi** (needs the camera web server's capture URL):

    python watch_folder.py --input photos --esp-url http://<esp-ip>/capture --every 10

**C) Live view in the web app:** run `streamlit run app.py`, choose *Camera folder (latest photo)*,
and type the folder path. It always shows the newest photo with veins marked.

## Hosted on Streamlit Cloud + camera on another computer
A hosted app cannot read a folder on your PC. Use cloud storage as the bridge:
1. supabase.com → new project → Storage → create bucket `photos` (private).
2. On the camera PC: set SUPABASE_URL / SUPABASE_KEY (service_role key) and run
   `python cloud_uploader.py --input "C:/esp_photos"`.
3. In Streamlit Cloud → App settings → Secrets add:
       SUPABASE_URL = "https://xxxx.supabase.co"
       SUPABASE_KEY = "<service_role key>"
       SUPABASE_BUCKET = "photos"
4. In the app choose *Cloud (latest photo from camera)*.

## Camera with an IP address (live)
In the app choose *Camera IP address (live)* and enter the camera URL, e.g.
`http://192.168.1.50/capture` (single photo) or `http://192.168.1.50:81/stream` (video).
- Works if the app runs on a computer **on the same Wi-Fi** as the camera (`streamlit run app.py`).
- A website hosted on Streamlit Cloud can only reach a **public** address. Private IPs (192.168.x.x,
  10.x.x.x) are invisible to it. Either expose the camera with a tunnel (Cloudflare Tunnel / ngrok /
  port-forwarding + public IP or DDNS) and use that public URL, or use the cloud uploader route above.

## One-click start (no commands)
- Windows: double-click `start.bat`
- Mac: double-click `start.command` (first time: right-click → Open)
- Linux: run `./start.sh`
First launch installs everything automatically (internet needed once); later launches open instantly.
Choose *Webcam (live camera + capture)* in the app, allow the camera, click Take Photo.

## HTML page + hosted Python API (no Streamlit)
- `api.py` is the hosted Python part (FastAPI). `index.html` is the page you open on the laptop.
- Deploy `api.py`: push this folder to GitHub → Render.com → New Web Service → "Docker" (uses `Dockerfile`).
  You get an address like `https://your-api.onrender.com`. Test: open it in a browser, you should see {"status":"ok"}.
- Edit the top of `index.html`: set `CAMERA_IP` and `API_URL`, then double-click `index.html`.
- Test API locally: `uvicorn api:app --port 8000` and set API_URL to `http://localhost:8000`.
