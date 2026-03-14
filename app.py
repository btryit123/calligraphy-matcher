import os
import json
from typing import List, Dict, Any

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "db_images")
MANIFEST_PATH = os.path.join(BASE_DIR, "manifest.json")

app = FastAPI(title="Calligraphy Matcher API")

orb = cv2.ORB_create(nfeatures=1500)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

db_cache: List[Dict[str, Any]] = []


def read_manifest():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_image_gray(path: str):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    # 統一縮放，讓比對更穩
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

    # 取前 80 個最好 match
    good = matches[:80]

    # distance 越小越好，轉成分數
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


@app.on_event("startup")
def startup_event():
    global db_cache
    db_cache = build_db_cache()
    print(f"已載入作品數：{len(db_cache)}")


@app.get("/")
def root():
    return {"ok": True, "message": "Calligraphy Matcher API is running"}


@app.post("/match")
async def match_calligraphy(file: UploadFile = File(...)):
    try:
        content = await file.read()
        np_arr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

        if img is None:
            return JSONResponse(
                status_code=400,
                content={"ok": False, "error": "無法讀取上傳圖片"}
            )

        h, w = img.shape[:2]
        max_side = max(h, w)
        if max_side > 1200:
            scale = 1200 / max_side
            img = cv2.resize(img, None, fx=scale, fy=scale)

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

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": str(e)}
        )