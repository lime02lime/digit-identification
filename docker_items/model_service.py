import torch
import torchvision
import torch.nn as nn
from flask import Flask, request, jsonify
from PIL import Image
import io
import torchvision.transforms as transforms
import torch.nn.functional as F


class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # Input layer (784 -> 128)
        self.fc2 = nn.Linear(128, 64)     # Hidden layer (128 -> 64)
        self.fc3 = nn.Linear(64, 10)      # Output layer (64 -> 10)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten images
        x = torch.relu(self.fc1(x))  
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # No activation (logits)
        return x

# Initialize the model
model = MNISTClassifier()

# Load the trained weights (state_dict)
model.load_state_dict(torch.load('docker_items/model.pth'))

model.eval()

# Set up the Flask app
app = Flask(__name__)

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    image_file = request.files['image'].read()
    image = Image.open(io.BytesIO(image_file))
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        prob = F.softmax(output, dim=1)  # Apply softmax to get probabilities
        confidence, predicted = torch.max(prob, 1)

    return jsonify({'prediction': predicted.item(),
                    'probability': confidence.item()
                    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
