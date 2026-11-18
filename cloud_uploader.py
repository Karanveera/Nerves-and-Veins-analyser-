"""Run this on the computer that has the camera-photo folder.
It uploads every new photo to a Supabase Storage bucket so the hosted Streamlit app can show it.

    set SUPABASE_URL=https://xxxx.supabase.co
    set SUPABASE_KEY=<service_role key>
    python cloud_uploader.py --input "C:/esp_photos"
(Linux/Mac: use `export` instead of `set`.)
"""
import argparse, os, time, urllib.request, urllib.error

EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
KEY = os.environ.get("SUPABASE_KEY", "")
BUCKET = os.environ.get("SUPABASE_BUCKET", "photos")


def stable(path):
    last = -1
    for _ in range(20):
        size = os.path.getsize(path)
        if size > 0 and size == last:
            return True
        last = size; time.sleep(0.5)
    return False


def upload(path):
    name = f"{int(time.time())}_{os.path.basename(path)}"   # timestamp prefix keeps order
    req = urllib.request.Request(
        f"{URL}/storage/v1/object/{BUCKET}/{name}", data=open(path, "rb").read(), method="POST",
        headers={"apikey": KEY, "Authorization": f"Bearer {KEY}", "Content-Type": "application/octet-stream"})
    urllib.request.urlopen(req, timeout=60).read()
    print("[uploaded]", name, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--interval", type=float, default=2)
    a = ap.parse_args()
    if not URL or not KEY:
        raise SystemExit("Set SUPABASE_URL and SUPABASE_KEY first.")
    done = set(os.listdir(a.input))        # ignore photos already there; upload only NEW ones
    print("Watching", a.input, "- Ctrl+C to stop")
    while True:
        for f in sorted(os.listdir(a.input)):
            p = os.path.join(a.input, f)
            if f.lower().endswith(EXT) and f not in done and os.path.isfile(p):
                done.add(f)
                try:
                    if stable(p): upload(p)
                except Exception as e:
                    print("[error]", f, e)
        time.sleep(a.interval)
