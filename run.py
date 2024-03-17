import argparse
import torch
from torchvision.transforms import ToTensor
from PIL import Image
from flask import Flask, request, jsonify
from mnist_classifier import MNISTClassifier

app = Flask(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--path",
    type=str,
    default="mnist_classifier.pth",
    help="Path to model state dict",
)

args = parser.parse_args()

# Load model
params = {"input_size": 784, "hidden_size": 128, "num_classes": 10, "path": args.path}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNISTClassifier(
    params["input_size"], params["hidden_size"], params["num_classes"]
)
model.load_state_dict(torch.load(args.path, map_location=device))
model.eval()


@app.route("/")
def index():
    return "MNIST Classifier API"


@app.route("/predict", methods=["POST"])
def predict():
    # Get image file from request
    image_file = request.files["input"]

    # Open and preprocess the image
    image = Image.open(image_file)
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28 pixels

    # Convert image to tensor
    transform = ToTensor()
    image_tensor = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output.data, 1)
        predicted_class = predicted.item()

    return jsonify({"predicted_class": predicted_class})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=80)
