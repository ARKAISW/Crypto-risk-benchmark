FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY openenv.yaml .
COPY inference.py .
COPY README.md .

# Expose port (7860 for HF Spaces, also works with PORT env var)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860/health').raise_for_status()"

# Run the FastAPI server
# PORT env var is set by HF Spaces automatically
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
