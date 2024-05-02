import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import requests
from io import BytesIO

# Function to load the model
def load_model():
    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(2048, 4)  # Assuming 4 classes for four-legged animals
    return model

# Function to preprocess the image
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch

# Function to make predictions
def predict_image(image, model):
    input_batch = preprocess_image(image)
    with torch.no_grad():
        output = model(input_batch)
    _, predicted_idx = torch.max(output, 1)
    return predicted_idx.item()

# Download pre-trained ResNet50 model weights
@st.cache(allow_output_mutation=True)
def download_resnet50_weights():
    model = models.resnet50(pretrained=True)
    torch.save(model.state_dict(), "pretrained_resnet50_weights.pth")
    return "pretrained_resnet50_weights.pth"

# Load the model
@st.cache(allow_output_mutation=True)
def load_model():
    weights_file = download_resnet50_weights()
    model = models.resnet50()
    model.fc = torch.nn.Linear(2048, 4)
    model.load_state_dict(torch.load(weights_file))
    model.eval()
    return model

st.title("Four-Legged Animal Predictor")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    if st.button('Predict'):
        if image is not None:
            model = load_model()
            predicted_class_idx = predict_image(image, model)
            classes = ['Dog', 'Cat', 'Horse', 'Other']
            st.write(f"Predicted Animal: {classes[predicted_class_idx]}")
