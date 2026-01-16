# Eye Disease Classification (Gradio)

This repository runs a local Gradio app to classify eye images using a PyTorch model.

Files
- `app.py`: Gradio app and model-loading code.
- `requirements.txt`: Python dependencies.
- `1eye_disease_cnn.pth`: Your trained model (place at E:\VisionX or update `MODEL_PATH`).

Setup and run
1. Create (or activate) a Python environment and install requirements:

```bash
pip install -r requirements.txt
```

2. Run the app:

```bash
python app.py
```

3. Open the provided link (http://127.0.0.1:7860) in your browser and upload an image.

Notes
- If `torch.load` fails because the checkpoint uses a different architecture, update the model construction in `app.py` (search for `models.resnet18`) to match your training script.
- If class order differs, update `CLASS_NAMES` in `app.py` to match the trained model's label order.
