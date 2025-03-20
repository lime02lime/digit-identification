FROM python:3.9-slim

# Install dependencies
RUN pip install --no-cache-dir streamlit torch torchvision

# Copy your Streamlit app script
COPY app.py /app/app.py

# Set working directory
WORKDIR /app

CMD ["streamlit", "run", "app.py"]
