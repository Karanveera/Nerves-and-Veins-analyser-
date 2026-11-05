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
