# Use Python 3.11 slim base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Update package list and install system dependencies
RUN apt-get update && apt-get install -y \
    # GDAL and development tools
    gdal-bin \
    libgdal-dev \
    python3-gdal \
    # Build essentials
    build-essential \
    gcc \
    g++ \
    pkg-config \
    # Additional libraries
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libfontconfig1-dev \
    git \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Get GDAL version and set environment variables
RUN export GDAL_VERSION=$(gdal-config --version) && \
    echo "GDAL version: $GDAL_VERSION"

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
ENV GDAL_CONFIG=/usr/bin/gdal-config
ENV GDAL_DATA=/usr/share/gdal

# Upgrade pip and install wheel
RUN pip install --upgrade pip setuptools wheel

# Install GDAL Python bindings with explicit version matching
RUN export GDAL_VERSION=$(gdal-config --version) && \
    echo "Installing GDAL Python bindings version: $GDAL_VERSION" && \
    pip install --no-cache-dir GDAL==$GDAL_VERSION

# Verify GDAL installation
RUN python3 -c "from osgeo import gdal, ogr, osr; print(f'GDAL version: {gdal.__version__}'); print('GDAL successfully installed!')"

# Create working directory
WORKDIR /app

# Copy requirements file first (for better Docker layer caching)
COPY requirements.txt .

# Install Python requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
# COPY . .

# Set the default command
CMD ["python", "-c", "from osgeo import gdal, ogr, osr; print(f'GDAL version: {gdal.__version__}'); print('Container ready!')"]