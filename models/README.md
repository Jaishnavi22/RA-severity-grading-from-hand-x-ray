# models/

Put these three files here. They come from the export cell at the end of the notebook.

| File | What it is | Size |
|---|---|---|
| `classifier.onnx` | EfficientNet-B0 + CBAM + CORAL head (best EMA weights) | about 16 MB |
| `joint_detector.onnx` | YOLO joint detector | about 10 to 12 MB |
| `config.json` | mean/std, cut-points, conformal thresholds, ROI and CLAHE settings | about 30 KB |

Commit them to GitHub together with the code: Vercel builds from the repository, so the files must be in it.
Each file is below GitHub's 100 MB limit and below the 25 MB limit of the browser upload page.
