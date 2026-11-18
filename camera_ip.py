"""Grab one frame from an IP camera (ESP32-CAM etc.) by URL."""
import ipaddress, socket
from urllib.parse import urlparse
import cv2, numpy as np, requests


def is_private(url):
    """True if the camera address is only reachable inside a home/office network."""
    host = urlparse(url if "://" in url else "http://" + url).hostname or ""
    try:
        return ipaddress.ip_address(socket.gethostbyname(host)).is_private
    except Exception:
        return False


def grab_frame(url, timeout=10):
    """Return JPEG bytes of one frame. Works with /capture (a JPEG) or /stream (MJPEG video)."""
    if "://" not in url:
        url = "http://" + url
    try:
        r = requests.get(url, timeout=timeout, stream=True)
        ctype = r.headers.get("Content-Type", "")
        if "image/" in ctype:
            return r.content
        r.close()
    except requests.RequestException:
        pass
    cap = cv2.VideoCapture(url)          # video stream fallback: take a single frame
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Could not read a frame from the camera URL")
    return cv2.imencode(".jpg", frame)[1].tobytes()
