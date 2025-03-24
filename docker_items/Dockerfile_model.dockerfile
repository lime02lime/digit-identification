# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python model service script and the model file into the container
COPY model_service.py /app/
COPY model.pth /app/

# Expose the port the app runs on
EXPOSE 5000

# Command to run the Flask app
CMD ["python", "model_service.py"]
