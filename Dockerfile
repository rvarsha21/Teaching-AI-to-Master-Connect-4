FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY backend.py .
COPY cnn_connect4.h5 .
COPY transformer_connect4.keras .

# Run backend
CMD ["python", "backend.py"]