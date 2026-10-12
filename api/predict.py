"""RA severity grading as a Vercel Python function (ONNX Runtime, no PyTorch).

POST /api/predict   body = the raw image file (JPEG / PNG / BMP / TIFF), at most 4.5 MB
GET  /api/predict   health check

Every step repeats what the notebook does in `run_pipeline` and in the cell
"predictions for the images you uploaded":

    raw grey -> CLAHE -> JPEG round-trip -> YOLO joints (on the raw image) -> union ROI
    -> crop CLAHE image -> JPEG round-trip -> resize + normalise -> CORAL EfficientNet-B0
    + CBAM (+ mirror TTA) -> cut-point offsets -> conformal answer / defer -> Grad-CAM++

All numbers (mean, std, cut-points, conformal thresholds, ROI settings) are read from
models/config.json, which the export cell at the end of the notebook writes.

Decision support only. A clinician must confirm every grade.
"""
import base64
import io
import json
import os
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("RA_MODELS_DIR", ROOT / "models"))
MAX_BODY = 4_500_000          # Vercel rejects larger request bodies anyway
MAX_PIXELS = 60_000_000
DISCLAIMER = ("Decision-support only. The grade must be confirmed by a clinician; the heat-map shows "
              "where the model looked, not proven pathology.")


class ModelsMissing(RuntimeError):
    pass


class BadImage(ValueError):
    pass


# ----------------------------------------------------------------------------- model loading
_STATE = None
_LOCK = threading.Lock()


def load_models():
    """Load the config and both ONNX sessions once per server instance."""
    global _STATE
    if _STATE is not None:
        return _STATE
    with _LOCK:
        if _STATE is None:
            names = ("config.json", "classifier.onnx", "joint_detector.onnx")
            missing = [n for n in names if not (MODELS_DIR / n).is_file()]
            if missing:
                raise ModelsMissing("Model files not found in /models: " + ", ".join(missing))
            cfg = json.loads((MODELS_DIR / "config.json").read_text())
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "1"))
            providers = ["CPUExecutionProvider"]
            clf = ort.InferenceSession(str(MODELS_DIR / "classifier.onnx"), opts, providers=providers)
            det = ort.InferenceSession(str(MODELS_DIR / "joint_detector.onnx"), opts, providers=providers)
            _STATE = {"cfg": cfg, "clf": clf, "det": det}
    return _STATE


# ----------------------------------------------------------------------------- preprocessing
def load_gray(data):
    """Same as the notebook's load_gray (no resizing)."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        try:
            img = np.array(Image.open(io.BytesIO(data)).convert("L"))
        except Exception as exc:
            raise BadImage("The file could not be read as an image.") from exc
    if img.ndim == 3:
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim != 2 or img.size > MAX_PIXELS:
        raise BadImage("Unsupported image size or format.")
    return img


def apply_clahe(gray, clip, tile):
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(gray)


def jpeg_roundtrip(gray, quality):
    """The dataset build saved every image as JPEG; do the same compression here."""
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise BadImage("The image could not be re-encoded.")
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


# ----------------------------------------------------------------------------- YOLO joint detector
def letterbox(img_bgr, size, stride, auto):
    """Ultralytics LetterBox (centre padding with grey 114). auto=True pads to a multiple of `stride`."""
    h, w = img_bgr.shape[:2]
    r = min(size / h, size / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    dw, dh = size - new_w, size - new_h
    if auto:
        dw, dh = dw % stride, dh % stride
    dw, dh = dw / 2, dh / 2
    if (w, h) != (new_w, new_h):
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(img_bgr, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))


def nms(boxes, scores, iou_thr):
    """Greedy NMS (same rule as torchvision.ops.nms: drop boxes with IoU > threshold)."""
    order = np.argsort(-scores, kind="stable")
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        iw = np.maximum(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0)
        ih = np.maximum(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0)
        inter = iw * ih
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)
        order = rest[iou <= iou_thr]
    return np.array(keep, dtype=np.int64)


def detect_joints(gray, det_sess, ycfg):
    """Same output as `YOLO.predict(image, imgsz, conf, max_det)`: normalised xyxy boxes, conf, class."""
    h0, w0 = gray.shape
    lb = letterbox(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), ycfg["imgsz"], ycfg.get("stride", 32),
                   auto=ycfg.get("dynamic", True))
    h1, w1 = lb.shape[:2]
    x = np.ascontiguousarray(lb[..., ::-1].transpose(2, 0, 1), dtype=np.float32)[None] / 255.0
    out = det_sess.run(None, {det_sess.get_inputs()[0].name: x})[0]
    nc = int(ycfg["nc"])
    pred = out[0]
    if pred.shape[0] == 4 + nc:
        pred = pred.T                                   # -> (anchors, 4 + nc)
    scores = pred[:, 4:4 + nc]
    conf, cls = scores.max(1), scores.argmax(1)
    keep = conf > ycfg["conf"]
    box, conf, cls = pred[keep, :4], conf[keep], cls[keep]
    empty = (np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64))
    if len(conf) == 0:
        return empty
    xyxy = np.stack([box[:, 0] - box[:, 2] / 2, box[:, 1] - box[:, 3] / 2,
                     box[:, 0] + box[:, 2] / 2, box[:, 1] + box[:, 3] / 2], 1).astype(np.float32)
    order = np.argsort(-conf, kind="stable")[:30000]
    xyxy, conf, cls = xyxy[order], conf[order], cls[order]
    idx = nms(xyxy + cls[:, None].astype(np.float32) * 7680.0, conf, ycfg.get("iou", 0.7))[:ycfg.get("max_det", 60)]
    xyxy, conf, cls = xyxy[idx], conf[idx], cls[idx]
    gain = min(h1 / h0, w1 / w0)
    pad_x = round((w1 - w0 * gain) / 2 - 0.1)
    pad_y = round((h1 - h0 * gain) / 2 - 0.1)
    xyxy[:, [0, 2]] -= pad_x
    xyxy[:, [1, 3]] -= pad_y
    xyxy /= gain
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w0)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h0)
    return xyxy / np.array([w0, h0, w0, h0], np.float32), conf, cls


def union_roi(boxes_n, conf, cls, w, h, conf_min, pad, class_ids):
    """ROI from confident DIP/MCP/PIP/Wrist detections (the notebook's union_roi)."""
    m = (conf >= conf_min) & np.isin(cls, list(class_ids))
    if not m.any():
        return None
    x1, y1 = boxes_n[m, 0].min() * w, boxes_n[m, 1].min() * h
    x2, y2 = boxes_n[m, 2].max() * w, boxes_n[m, 3].max() * h
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(w, x2 + px), min(h, y2 + py)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


# ----------------------------------------------------------------------------- CORAL, conformal, Grad-CAM++
def coral_probs(z):
    """Class probabilities from CORAL logits (z already includes the cut-point offsets)."""
    s = 1.0 / (1.0 + np.exp(-z.astype(np.float64)))
    gt = np.minimum.accumulate(s, axis=1)                     # P(y > k), forced monotone
    one, zero = np.ones_like(gt[:, :1]), np.zeros_like(gt[:, :1])
    cum = np.concatenate([one, gt, zero], axis=1)
    return cum[:, :-1] - cum[:, 1:]


def decode_probs(probs):
    return (np.cumsum(probs, axis=1)[:, :-1] < 0.5).sum(1)


def conformal_decide(probs, qhat):
    sets = (1.0 - probs) <= np.asarray(qhat)[None, :]
    return sets, sets.sum(1) == 1


def grad_campp(act, head_w, out, cls, size):
    """Grad-CAM++ on the last EfficientNet stage, explaining class `cls` (the notebook's grad_campp).

    After that stage the network is only global-average-pool -> one shared CORAL score s, so the
    gradient of the notebook's class evidence w.r.t. the activations is known in closed form:
    d evidence / d act[k, i, j] = d * head_w[k] / (H * W), with d = -1, +1 or -2*sign(z_prev + z_curr).
    """
    c, h, w = act.shape
    k1 = out.shape[0]
    if cls == 0:
        d = -1.0
    elif cls == k1:
        d = 1.0
    else:
        d = -2.0 * float(np.sign(out[cls - 1] + out[cls]))
    g = np.broadcast_to((d * head_w / (h * w)).astype(np.float32)[:, None, None], act.shape)
    g2, g3 = g ** 2, g ** 3
    denom = 2.0 * g2 + act.sum(axis=(1, 2), keepdims=True) * g3
    denom = np.where(denom != 0, denom, 1.0)
    alpha = g2 / denom
    weights = (alpha * np.maximum(g, 0)).sum(axis=(1, 2), keepdims=True)
    cam = np.maximum((weights * act).sum(axis=0), 0).astype(np.float32)
    cam = cv2.resize(cam, (size, size), interpolation=cv2.INTER_LINEAR)
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def overlay_cam(img01, cam, alpha=0.45):
    gray = (img01 * 255).astype(np.uint8)
    heat = cv2.cvtColor(cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return (gray * (1 - alpha) + heat * alpha).astype(np.uint8)


# ----------------------------------------------------------------------------- output helpers
def to_data_url(img, max_side=None, quality=85):
    """img: grey (H, W) or RGB (H, W, 3) uint8 -> JPEG data URL."""
    if max_side and max(img.shape[:2]) > max_side:
        s = max_side / max(img.shape[:2])
        img = cv2.resize(img, (max(1, round(img.shape[1] * s)), max(1, round(img.shape[0] * s))),
                         interpolation=cv2.INTER_AREA)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


# ----------------------------------------------------------------------------- full pipeline
def run_inference(data):
    """Returns (json-able result, debug arrays used by tools/parity_check.py)."""
    st = load_models()
    cfg, clf, det = st["cfg"], st["clf"], st["det"]
    names = cfg["class_names"]
    size = int(cfg["img_size"])
    notes = []

    raw = load_gray(data)
    clahe_img = jpeg_roundtrip(apply_clahe(raw, cfg["clahe"]["clip"], cfg["clahe"]["tile"]), cfg["jpeg_quality"])

    boxes_n, conf, cls = detect_joints(raw, det, cfg["yolo"])          # the detector was trained on RAW images
    rcfg = cfg["roi"]
    n_joints = int((conf >= rcfg["conf"]).sum())
    h, w = clahe_img.shape
    box = union_roi(boxes_n, conf, cls, w, h, rcfg["conf"], rcfg["pad"], rcfg["class_ids"]) if len(conf) else None
    if box is None:
        notes.append("No confident joint was found, so the whole image was used (same fallback as in training).")
        crop = clahe_img
    else:
        crop = clahe_img[box[1]:box[3], box[0]:box[2]]
    final = jpeg_roundtrip(crop, cfg["jpeg_quality"])
    if n_joints == 0:
        notes.append("WARNING: 0 joints detected. This may not be a hand or wrist X-ray.")

    # model input: Resize((S, S)) -> ToTensor -> Normalize, exactly like valTransformer
    pil = Image.fromarray(cv2.cvtColor(final, cv2.COLOR_GRAY2RGB)).resize((size, size), Image.Resampling.BILINEAR)
    mean = np.array(cfg["mean"], np.float32)
    std = np.array(cfg["std"], np.float32)
    x = ((np.asarray(pil, dtype=np.float32) / 255.0 - mean) / std).transpose(2, 0, 1)
    batch = np.ascontiguousarray(np.stack([x, x[:, :, ::-1]]), dtype=np.float32)    # image + mirror image

    in_name = clf.get_inputs()[0].name
    logits, act = clf.run(None, {in_name: batch})[:2]
    offsets = np.array(cfg["cut_offsets"], np.float32)
    z = 0.5 * (logits[0] + logits[1]) if cfg.get("tta", True) else logits[0]
    probs = coral_probs((z + offsets)[None, :])[0]
    pred = int(decode_probs(probs[None, :])[0])

    sets, answered = conformal_decide(probs[None, :], cfg["qhat"])
    plausible = [names[k] for k in np.where(sets[0])[0]]

    cam = grad_campp(act[0].astype(np.float32), np.array(cfg["head_weight"], np.float32),
                     (logits[0] + offsets).astype(np.float64), pred, size)
    img01 = np.clip(x.transpose(1, 2, 0) * std + mean, 0, 1)
    heat = overlay_cam(img01, cam)

    vis = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB)
    thick = max(2, w // 300)
    for b, c in zip(boxes_n, conf):
        if c >= rcfg["conf"]:
            cv2.rectangle(vis, (int(b[0] * w), int(b[1] * h)), (int(b[2] * w), int(b[3] * h)), (255, 0, 0), thick)
    if box is not None:
        cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), thick * 2)

    result = {
        "grade": names[pred],
        "confidence": float(probs[pred]),
        "probabilities": {n: float(p) for n, p in zip(names, probs)},
        "decision": "ANSWERED" if bool(answered[0]) else "DEFERRED",
        "plausible_grades": plausible,
        "joints_found": n_joints,
        "roi_box": list(box) if box else None,
        "notes": notes,
        "images": {
            "joints": to_data_url(vis, 720),
            "roi": to_data_url(final, 512),
            "heatmap": to_data_url(heat),
        },
        "disclaimer": DISCLAIMER,
    }
    debug = {"probs": probs, "pred": pred, "answered": bool(answered[0]), "cam": cam,
             "n_joints": n_joints, "roi_box": box}
    return result, debug


# ----------------------------------------------------------------------------- HTTP (Vercel + local)
class handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        present = all((MODELS_DIR / n).is_file() for n in ("config.json", "classifier.onnx", "joint_detector.onnx"))
        self._send(200, {"status": "ok", "models_present": present, "models_loaded": _STATE is not None})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0:
                return self._send(400, {"error": "Send the image file as the request body."})
            if n > MAX_BODY:
                return self._send(413, {"error": "Image is larger than 4.5 MB."})
            result, _ = run_inference(self.rfile.read(n))
            self._send(200, result)
        except BadImage as exc:
            self._send(422, {"error": str(exc)})
        except ModelsMissing as exc:
            self._send(503, {"error": str(exc)})
        except Exception as exc:  # keep details in the server log, not in the response
            print("inference error:", repr(exc))
            self._send(500, {"error": "The image could not be processed."})


def serve(port=8000):
    """Local test server: python api/predict.py  ->  http://127.0.0.1:8000"""
    from http.server import ThreadingHTTPServer
    public = ROOT / "public"
    types = {".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css"}

    class Local(handler):
        def do_GET(self):
            path = self.path.split("?")[0]
            if path.startswith("/api/predict"):
                return super().do_GET()
            f = (public / ("index.html" if path in ("", "/") else path.lstrip("/"))).resolve()
            if public.resolve() not in f.parents or not f.is_file():
                self.send_response(404)
                self.end_headers()
                return
            body = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", types.get(f.suffix, "application/octet-stream"))
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    print(f"Open http://127.0.0.1:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Local).serve_forever()


if __name__ == "__main__":
    serve(int(os.environ.get("PORT", "8000")))
