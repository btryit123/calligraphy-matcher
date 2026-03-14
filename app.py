import os
import json
import base64
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "db_images")
MANIFEST_PATH = os.path.join(BASE_DIR, "manifest.json")

app = FastAPI(title="Calligraphy Matcher API")

orb = cv2.ORB_create(nfeatures=1500)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

db_cache: List[Dict[str, Any]] = []


class MatchRequest(BaseModel):
    image_base64: str
    file_name: Optional[str] = "upload.jpg"
    mime_type: Optional[str] = "image/jpeg"


def read_manifest():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_image_gray(path: str):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape[:2]
    max_side = max(h, w)
    if max_side > 1200:
        scale = 1200 / max_side
        img = cv2.resize(img, None, fx=scale, fy=scale)
    return img


def compute_features(img_gray):
    keypoints, descriptors = orb.detectAndCompute(img_gray, None)
    return keypoints, descriptors


def score_match(desc1, desc2):
    if desc1 is None or desc2 is None:
        return 0

    matches = bf.match(desc1, desc2)
    if not matches:
        return 0

    matches = sorted(matches, key=lambda x: x.distance)
    good = matches[:80]

    score = 0
    for m in good:
        score += max(0, 100 - m.distance)

    return int(score)


def build_db_cache():
    manifest = read_manifest()
    cache = []

    for item in manifest:
        file_name = item["file"]
        img_path = os.path.join(DB_DIR, file_name)
        img = load_image_gray(img_path)
        if img is None:
            print(f"讀不到圖片：{img_path}")
            continue

        kp, desc = compute_features(img)

        cache.append({
            "id": str(item["id"]),
            "name": item["name"],
            "text": item["text"],
            "file": file_name,
            "img_path": img_path,
            "keypoints_count": 0 if kp is None else len(kp),
            "descriptors": desc,
        })

    return cache


def decode_uploaded_image(content: bytes):
    np_arr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None

    h, w = img.shape[:2]
    max_side = max(h, w)
    if max_side > 1200:
        scale = 1200 / max_side
        img = cv2.resize(img, None, fx=scale, fy=scale)

    return img


def run_match(img):
    kp, desc = compute_features(img)

    if desc is None:
        return {
            "ok": True,
            "best_match_id": "",
            "best_match_name": "",
            "best_match_text": "",
            "score": 0,
            "top3": [],
            "reason": "上傳圖片無法提取足夠特徵"
        }

    results = []
    for item in db_cache:
        score = score_match(desc, item["descriptors"])
        results.append({
            "id": item["id"],
            "name": item["name"],
            "text": item["text"],
            "score": score
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    top3 = results[:3]
    best = top3[0] if top3 else {"id": "", "name": "", "text": "", "score": 0}

    return {
        "ok": True,
        "best_match_id": best["id"],
        "best_match_name": best["name"],
        "best_match_text": best["text"],
        "score": best["score"],
        "top3": top3
    }


@app.on_event("startup")
def startup_event():
    global db_cache
    db_cache = build_db_cache()
    print(f"已載入作品數：{len(db_cache)}")


@app.get("/")
def root():
    return {"ok": True, "message": "Calligraphy Matcher API is running"}


@app.get("/health")
def health():
    return {"status": "ok", "db_count": len(db_cache)}


@app.post("/match")
async def match_calligraphy(file: UploadFile = File(...)):
    try:
        content = await file.read()
        img = decode_uploaded_image(content)

        if img is None:
            return JSONResponse(
                status_code=400,
                content={"ok": False, "error": "無法讀取上傳圖片"}
            )

        return run_match(img)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(e)}
        )


@app.post("/match-json")
async def match_calligraphy_json(payload: MatchRequest):
    try:
        raw = (payload.image_base64 or "").strip()

        # 若前端不小心帶 data:image/...;base64, 也一起清掉
        if raw.startswith("data:"):
            raw = raw.split(",", 1)[-1]

        image_bytes = base64.b64decode(raw)
        img = decode_uploaded_image(image_bytes)

        if img is None:
            return JSONResponse(
                status_code=400,
                content={"ok": False, "error": "無法解析 base64 圖片"}
            )

        result = run_match(img)
        result["input_file_name"] = payload.file_name
        result["input_mime_type"] = payload.mime_type
        return result

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(e)}
        )
