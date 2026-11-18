"""Watch a folder for new photos (e.g. saved by an ESP32-CAM) and mark the veins automatically.

Usage:
    python watch_folder.py --input "C:/esp_photos" --output "C:/esp_photos/marked"
Optional (grab photos straight from the ESP32-CAM over Wi-Fi):
    python watch_folder.py --input photos --esp-url http://192.168.1.50/capture --every 10
"""
import argparse, os, time, urllib.request
import cv2, numpy as np
from PIL import Image
from vein_detector import process

EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def wait_until_written(path, tries=20):
    """Camera may still be writing the file: wait until its size stops changing."""
    last = -1
    for _ in range(tries):
        size = os.path.getsize(path)
        if size > 0 and size == last:
            return True
        last = size
        time.sleep(0.5)
    return False


def mark_file(path, out_dir, args):
    img = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
    _, marked, n = process(img, args.sensitivity, args.min_length, hand=args.hand)
    name = os.path.splitext(os.path.basename(path))[0] + "_marked.png"
    cv2.imwrite(os.path.join(out_dir, name), marked)
    print(f"[done] {os.path.basename(path)} -> {name}  ({n} veins)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="folder where the camera saves photos")
    ap.add_argument("--output", default=None, help="folder for marked images (default: <input>/marked)")
    ap.add_argument("--hand", default="auto", choices=["auto", "top", "bottom", "left", "right", "none"])
    ap.add_argument("--sensitivity", type=float, default=0.8)
    ap.add_argument("--min-length", type=int, default=45)
    ap.add_argument("--interval", type=float, default=2.0, help="seconds between folder checks")
    ap.add_argument("--esp-url", default=None, help="ESP32-CAM capture URL, e.g. http://192.168.1.50/capture")
    ap.add_argument("--every", type=float, default=10, help="seconds between ESP captures")
    ap.add_argument("--once", action="store_true", help="process existing photos and exit")
    args = ap.parse_args()
    if args.hand == "none":
        args.hand = None

    out_dir = args.output or os.path.join(args.input, "marked")
    os.makedirs(args.input, exist_ok=True); os.makedirs(out_dir, exist_ok=True)
    done = set()
    next_cap = 0
    print(f"Watching {args.input}  ->  {out_dir}   (Ctrl+C to stop)")

    while True:
        if args.esp_url and time.time() >= next_cap:
            try:
                data = urllib.request.urlopen(args.esp_url, timeout=10).read()
                fn = os.path.join(args.input, time.strftime("esp_%Y%m%d_%H%M%S.jpg"))
                open(fn, "wb").write(data)
            except Exception as e:
                print("[esp] capture failed:", e)
            next_cap = time.time() + args.every

        for f in sorted(os.listdir(args.input)):
            p = os.path.join(args.input, f)
            if f.lower().endswith(EXT) and os.path.isfile(p) and p not in done:
                done.add(p)
                try:
                    if wait_until_written(p):
                        mark_file(p, out_dir, args)
                except Exception as e:
                    print(f"[error] {f}: {e}")
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
