FROM python:3.10-slim

# Install system dependencies required by OpenCV
RUN apt-get update && \
    apt-get install -y \
    libgtk-3-0 \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install psycopg2-binary

# Copy your Streamlit app script
COPY app.py .

# Expose the port the app runs on
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
