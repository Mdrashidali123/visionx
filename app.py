import torch
import gradio as gr
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = r"E:\VisionX\1eye_disease_cnn.pth"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

# MATCHed the TRAINING ORDEr
class_names = [
    "Cataract",
    "Diabetic Retinopathy",
    "Glaucoma",
    "Normal"
]

NUM_CLASSES = len(class_names)

model = efficientnet_b0(weights=None)


model.classifier[1] = torch.nn.Linear(
    model.classifier[1].in_features,
    NUM_CLASSES
)

state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)

model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict(image: Image.Image):
    image = image.convert("RGB")
    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)[0]

    return {class_names[i]: float(probs[i]) for i in range(NUM_CLASSES)}

# launchingggg
app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Eye Image"),
    outputs=gr.Label(num_top_classes=4),
    title="👁️ Eye Disease Classification (EfficientNet-B0)",
    description=(
        "CNN-based eye disease detection system using EfficientNet-B0.\n\n"
        "Classes:\n"
        "• Cataract\n"
        "• Diabetic Retinopathy\n"
        "• Glaucoma\n"
        "• Normal"
    ),
    flagging_mode="never",
    live=False
)

if __name__ == "__main__":
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        debug=True
    )
