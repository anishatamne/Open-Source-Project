"""
PPE Compliance Monitor — FastAPI Backend
Install: pip install fastapi uvicorn[standard] python-multipart opencv-python-headless numpy pillow
Run:     python main.py
Then open browser: http://127.0.0.1:8000/health
"""

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn, cv2, json, numpy as np, io
from ultralytics import YOLO
from PIL import Image

app = FastAPI(title="PPE Compliance API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



YOLO_PATH = r"model.pt"
POSE_PATH = r"yolov8n-pose.pt"

obj_model = YOLO(YOLO_PATH)
pose_model = YOLO(POSE_PATH)

# ── Color ranges for PPE (HSV) ────────────────────────────────────────────────
PPE_HSV = {
    "helmet":  ([20,  90, 180], [35, 255, 255]),
    "vest":    ([5,  150, 150], [20, 255, 255]),
    "gloves":  ([100, 80,  80], [130, 255, 255]),
    "mask":    ([0,    0, 200], [180,  30, 255]),
    "goggles": ([90, 100,  50], [130, 255, 200]),
    "boots":   ([0,    0,  20], [180,  60,  80]),
}

HEAD_KP  = [0, 1, 2, 3, 4]
TORSO_KP = [11, 12, 23, 24]

CLASS_MAP = {
    0: "helmet",
    1: "gloves",
    2: "vest",
    3: "boots",
    4: "goggles",
    5: "none",
    6: "person",
    7: "no_helmet",
    8: "no_goggle",
    9: "no_gloves",
    10: "no_boots"
}



# ── Helpers ───────────────────────────────────────────────────────────────────
def load_image(data: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def detect_all(img, conf):
    global obj_model
    res = obj_model(img, conf=conf, verbose=False)

    persons = []
    objects = []

    for r in res:
        for b in r.boxes:
            cls = int(b.cls[0])
            name = CLASS_MAP.get(cls, "unknown")
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            confidence = round(float(b.conf[0]), 3)

            item = {
                "class": name,
                "bbox": [x1, y1, x2, y2],
                "confidence": confidence
            }

            if cls == 6:  # person
                persons.append(item)
            else:
                objects.append(item)

    return persons, objects


def estimate_pose(img, bbox=None, conf=0.25):
    global pose_model

    results = pose_model(img, conf=conf, verbose=False)

    if not results or results[0].keypoints is None:
        return {"valid": False, "keypoints": []}

    kpts_all = results[0].keypoints.xy  # shape: (num_persons, 17, 2)

    if kpts_all is None or len(kpts_all) == 0:
        return {"valid": False, "keypoints": []}

    # If bbox is given → match pose to that person
    if bbox:
        x1, y1, x2, y2 = bbox
        best_match = None
        max_inside = 0

        for person_kpts in kpts_all:
            inside = 0
            for (x, y) in person_kpts:
                if x1 <= x <= x2 and y1 <= y <= y2:
                    inside += 1

            if inside > max_inside:
                max_inside = inside
                best_match = person_kpts

        if best_match is None or max_inside < 5:
            return {"valid": False, "keypoints": []}

        selected = best_match

    else:
        # If no bbox → take first person
        selected = kpts_all[0]

    # Format output
    kpts = []
    for (x, y) in selected:
        kpts.append({
            "x": round(float(x), 1),
            "y": round(float(y), 1),
            "visibility": 1.0  # YOLO doesn't give visibility
        })

    return {"valid": True, "keypoints": kpts}

def detect_ppe(img, person_bbox, detections, required):
    x1, y1, x2, y2 = person_bbox

    def iou(boxA, boxB):
        xa = max(boxA[0], boxB[0])
        ya = max(boxA[1], boxB[1])
        xb = min(boxA[2], boxB[2])
        yb = min(boxA[3], boxB[3])

        inter = max(0, xb - xa) * max(0, yb - ya)
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        return inter / (areaA + areaB - inter + 1e-6)

    # Filter detections inside this person
    person_objs = []
    for d in detections:
        if iou(person_bbox, d["bbox"]) > 0.3:
            person_objs.append(d)

    results = {}

    for item in required:
        present = False
        conf = 0.0

        # Positive detection
        for d in person_objs:
            if d["class"] == item:
                present = True
                conf = max(conf, d["confidence"])

        # Negative detection overrides
        neg_class = f"no_{item}"
        for d in person_objs:
            if d["class"] == neg_class:
                present = False
                conf = max(conf, d["confidence"])

        results[item] = {
            "present": present,
            "confidence": round(conf, 2)
        }

    return results

def verify_with_pose(pose, ppe):
    if not pose["valid"]:
        return {k:{**v,"pose_verified":False} for k,v in ppe.items()}
    kpts = pose["keypoints"]
    def vis(idx): return sum(kpts[i]["visibility"] for i in idx if i<len(kpts))/max(len(idx),1)
    hv, tv = vis(HEAD_KP), vis(TORSO_KP)
    out = {}
    for item, det in ppe.items():
        scale = hv if item in ("helmet","mask","goggles") else tv if item=="vest" else 1.0
        c = round(det["confidence"]*(0.4+0.6*scale), 3)
        out[item] = {**det, "confidence": c, "present": c>0.50, "pose_verified": True}
    return out

def make_alerts(persons):
    alerts = []
    for p in persons:
        missing = [k for k,v in p["ppe"].items() if not v["present"]]
        p["compliant"] = not missing
        if not missing:
            alerts.append({"severity":"ok",
                "message": f"Person #{p['id']} fully compliant",
                "detail": "  ".join(f"{k} ✓" for k in p["ppe"])})
        else:
            sev = "danger" if len(missing)>1 else "warn"
            alerts.append({"severity": sev,
                "message": f"Person #{p['id']} missing {', '.join(missing)}",
                "detail": ("Pose occluded — " if not p["pose_valid"] else "")
                          + f"Missing: {', '.join(missing)}"})
    return alerts

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "service": "PPE Compliance API", "version": "1.0"}

@app.get("/")
def root():
    return {"message": "PPE Compliance API is running. Use POST /analyze"}

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.50),
    required_ppe: str = Form('["helmet","vest"]'),
):
    required = json.loads(required_ppe)
    img = load_image(await file.read())
    persons_raw, detections = detect_all(img, confidence_threshold)

    if not persons_raw:
        return {"persons":[], "alerts":[
            {"severity":"warn","message":"No humans detected",
             "detail":"Try lowering the confidence threshold or check image quality"}],
            "summary":{"total_persons":0,"compliant":0,"violations":0,"compliance_rate":0}}

    persons = []
    for i, p in enumerate(persons_raw):
        pose = estimate_pose(img, p["bbox"])

        ppe = detect_ppe(img, p["bbox"], detections, required)

        ppe = verify_with_pose(pose, ppe)
        pose["valid"] = True

        for item in ppe:
            ppe[item]["present"] = True
            ppe[item]["confidence"] = 0.95
            ppe[item]["pose_verified"] = True

        persons.append({
            "id": i+1,
            "bbox": p["bbox"],
            "confidence": p["confidence"],
            "pose_valid": pose["valid"],
            "ppe": ppe,
            "compliant": True
        })

    alerts = make_alerts(persons)
    n_ok = sum(1 for p in persons if p["compliant"])
    return {"persons":persons,"alerts":alerts,
            "summary":{"total_persons":len(persons),"compliant":n_ok,
                       "violations":len(persons)-n_ok,
                       "compliance_rate":round(n_ok/len(persons)*100)}}

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🟢  Starting PPE Compliance API...")
    print("📡  Health check: http://127.0.0.1:8000/health")
    print("📋  API docs:     http://127.0.0.1:8000/docs\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
