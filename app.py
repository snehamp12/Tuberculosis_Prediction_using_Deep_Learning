import os
from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = ['Healthy', 'OtherLung', 'Tuberculosis']

# Use ImageNet normalization if models were trained with it (likely)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet means
        std=[0.229, 0.224, 0.225]    # ImageNet stds
    )
])

def load_model(model_name):
    if model_name == 'vgg16':
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(4096, len(class_names))
        model.load_state_dict(torch.load("vgg16_model.pth", map_location=device))
    elif model_name == 'resnet16':
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, len(class_names))
        model.load_state_dict(torch.load("resnet16_model.pth", map_location=device))
    else:
        return None
    model.to(device)
    model.eval()
    return model

def predict_image(img_path, model):
    image = Image.open(img_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    file = request.files['image']
    model_choice = request.form.get('algorithm')

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if model_choice not in ['vgg16', 'resnet16']:
        return jsonify({'error': 'Invalid model choice'}), 400

    save_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(save_path)

    model = load_model(model_choice)
    if model is None:
        return jsonify({'error': 'Failed to load model'}), 500

    prediction = predict_image(save_path, model)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)