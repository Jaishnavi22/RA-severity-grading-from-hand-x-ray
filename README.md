# RA severity grading from hand X-rays

Research prototype that grades rheumatoid arthritis severity (Mild, Moderate, Severe) from a hand radiograph.
Pipeline: CLAHE, YOLO joint detection, hand region crop, CBAM-EfficientNet-B0 with a CORAL ordinal head, conformal
deferral to a clinician for uncertain cases, and Grad-CAM++ heat-maps.

**Decision support only. Not a medical device. A clinician must confirm every grade.**

## What is in this repository

```
notebooks/RA_Severity_Grading_from_Hand_XRAY.ipynb   training + evaluation (original notebook, one export cell added at the end)
api/predict.py         Vercel function: the same preprocessing and prediction as the notebook, run with ONNX Runtime
public/index.html      the web page (upload an X-ray, see grade, probabilities, heat-map)
models/                classifier.onnx, joint_detector.onnx, config.json   <- you add these (see below)
tools/parity_check.py  checks that the website gives the same answers as the notebook
vercel.json, requirements.txt, .gitignore, .vercelignore
```

The notebook is unchanged except for one markdown cell and one code cell appended at the end ("Deployment export").
Vercel cannot run PyTorch in a free function, so the website runs the exported ONNX models instead. The data, the
training checkpoints and `kaggle.json` are never committed (see `.gitignore`).

## 1. Export the models from Colab

1. Open the notebook in Colab and run it as usual (training, calibration, conformal cell, Explainable-AI cell,
   manual-upload cell). Use `DATA_VARIANT = "roi"`.
2. Run the last cell, "Deployment export". It checks that the ONNX classifier matches PyTorch, then downloads
   `ra_deploy_export.zip`.
3. Unzip it and copy:
   - `classifier.onnx`, `joint_detector.onnx` and `config.json` into `models/`
   - the `parity/` folder into `tools/parity/`

## 2. (Recommended) Check it locally

```
pip install -r requirements.txt
python tools/parity_check.py     # every sample must say PASS
python api/predict.py            # then open http://127.0.0.1:8000
```

## 3. Put it on GitHub

Command line:

```
git init
git add .
git commit -m "RA severity grading: notebook and web app"
git branch -M main
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```

Or use "Add file > Upload files" on github.com (each file must be under 25 MB; the three model files are).

## 4. Deploy on Vercel (free Hobby plan)

1. vercel.com > Add New > Project > import the GitHub repository.
2. Framework preset: **Other**. Leave the build and output settings empty. Click Deploy.
3. Open the URL and upload an X-ray. The first request after a quiet period is slow while the models load.

Limits to know about: a request can be at most 4.5 MB (the page shrinks bigger images automatically); the Hobby plan is
for personal, non-commercial use and has usage caps; the function has 2 GB of memory and 1 vCPU.

## Notes

- `api/predict.py` reads every setting (mean/std, cut-points, conformal thresholds, ROI) from `models/config.json`,
  so retraining only means re-running the export cell and replacing the three files.
- The model was trained on rheumatoid arthritis images only (Mild, Moderate, Severe). It does not detect normal hands.
- Uploaded images are processed in memory and are not stored by this app.
