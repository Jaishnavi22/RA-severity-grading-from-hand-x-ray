"""Checks that the website's ONNX pipeline reproduces the notebook's own answers.

The export cell at the end of the notebook saves a few validation images together with the
notebook's results for them (joints found, ROI box, class probabilities, answered/deferred,
Grad-CAM++). This script runs the same images through api/predict.py and compares.

    python tools/parity_check.py                      # models/ and tools/parity/ in this repo
    python tools/parity_check.py path/to/ra_deploy_export   # the unzipped export folder

Exit code 0 = all checks passed.
"""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
export = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
if export:
    os.environ["RA_MODELS_DIR"] = str(export)
ref_dir = export / "parity" if export else ROOT / "tools" / "parity"

sys.path.insert(0, str(ROOT / "api"))
import predict  # noqa: E402

ref_file = ref_dir / "reference.json"
if not ref_file.is_file():
    sys.exit(f"No reference.json in {ref_dir}. Re-run the export cell after the Grad-CAM++ and manual-upload cells.")
refs = json.loads(ref_file.read_text())

failures = 0
for r in refs:
    _, d = predict.run_inference((ref_dir / r["file"]).read_bytes())
    cam = cv2.resize(d["cam"].astype(np.float32), (96, 96), interpolation=cv2.INTER_AREA)
    ref_cam = np.array(r["cam96"], np.float32)
    box_diff = (max(abs(a - b) for a, b in zip(d["roi_box"], r["roi_box"]))
                if d["roi_box"] and r["roi_box"] else (0 if d["roi_box"] == r["roi_box"] else 999))
    checks = {
        "same joints found": d["n_joints"] == r["n_joints"],
        "ROI box within 3 px": box_diff <= 3,
        "probabilities within 0.02": float(np.abs(d["probs"] - np.array(r["probs"])).max()) <= 0.02,
        "same grade": d["pred"] == r["pred"],
        "same answered/deferred": d["answered"] == r["answered"],
        "heat-map close (mean abs diff <= 0.05)": float(np.abs(cam - ref_cam).mean()) <= 0.05,
        "heat-map correlation >= 0.95": float(np.corrcoef(cam.ravel(), ref_cam.ravel())[0, 1]) >= 0.95,
    }
    ok = all(checks.values())
    failures += not ok
    print(f"{r['file']}: {'PASS' if ok else 'FAIL'}")
    for name, passed in checks.items():
        print(f"   {'ok  ' if passed else 'FAIL'} {name}")
    print(f"   website probs {np.round(d['probs'], 4).tolist()} | notebook probs {np.round(r['probs'], 4).tolist()}"
          f" | ROI {d['roi_box']} vs {r['roi_box']}")

print("\nALL PASSED: safe to deploy." if not failures else f"\n{failures} sample(s) FAILED: do not deploy until this is explained.")
sys.exit(1 if failures else 0)
