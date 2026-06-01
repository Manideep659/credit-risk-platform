# ==============================================================================
# APEX BANK CREDIT RISK PLATFORM - CONTAINERISATION
# ==============================================================================
# Use official slim Python 3.11 base image for low storage footprint
FROM python:3.11-slim

# Set environment variables to optimize Python execution and Streamlit
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Set working directory inside the container
WORKDIR /app

# Install system dependencies
# NOTE: libgomp1 (OpenMP) is strictly required by LightGBM inside Debian/Slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker build cache
COPY requirements.txt .

# Install Python packages, disabling cache to keep image size minimal
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container working directory
COPY . .

# Expose Streamlit standard port
EXPOSE 8501

# Run the Streamlit web application automatically on container startup
CMD ["streamlit", "run", "src/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
