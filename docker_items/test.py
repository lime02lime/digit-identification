import requests

# Define the URL of the Flask server
url = "http://localhost:5000/predict"

# Open an image file
with open("test_image.png", "rb") as f:
    # Send POST request to the Flask server
    response = requests.post(url, files={"image": f})

# Print the response from the server
print(response.json())
