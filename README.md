# Digit Identification

Demo version available [here](http://13.40.6.33:8501/), deployed on AWS Lightsail.

## Overview

This is a hobby project to improve on my skills using a combination of ML techniques and application deployment. While the ML aspects of this problem are fairly simple, it is good practice with training and deploying ML models for applications in a containerized environment.

## ML Model
- In the Python notebook, I train a CNN on images of drawn digits from the MNIST dataset.
- ML modeling is done using PyTorch.
- The model takes a 28x28 pixel image as input and the forward pass decides on which digit it is most likely to be.
- 10,000 images are in the training set.

## Application
- The application is deployed as a composition of three Docker containers: Streamlit app, model-service, and PostgreSQL database.
- The main app is built using Streamlit and exposed through port 8501, hosted on an AWS Lightsail instance.
- The model-service is responsible for making predictions and is containerized separately.
- A PostgreSQL database is used to log interactions.

## Prerequisites
- Docker
- Docker Compose

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/lime02lime/digit-identification
    cd digit-identification/docker_items
    ```

2. Build and run the Docker containers:
    ```sh
    docker-compose up --build
    ```

## Usage
- Access the Streamlit app at `http://<your-lightsail-public-ip>:8501`.
- Draw a digit on the canvas and enter the actual number in the input field.
- Click "Submit" to get the model's prediction and confidence score.

## Project Structure
- `model_training.ipynb`: Jupyter notebook for training the CNN model.
- `docker_items/model_service.py`: Flask app serving the trained model.
- `docker_items/app.py`: Streamlit app for user interaction.
- `docker_items/Dockerfile_model.dockerfile`: Dockerfile for the model service.
- `docker_items/Dockerfile_streamlit.dockerfile`: Dockerfile for the Streamlit app.
- `requirements.txt`: Python dependencies.

## Deployment on AWS Lightsail
- The application is hosted on an AWS Lightsail instance running Ubuntu.
- The instance runs a Dockerized setup with `docker-compose` managing the services.
- The Streamlit app is publicly accessible via the instance's public IP on port 8501.

## Future Improvements
- Implement authentication for user sessions.
- Improve the model's accuracy by experimenting with different architectures.
- Automate deployment using CI/CD pipelines.
