FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY linear_regression/ ./linear_regression/
COPY data/ ./data/

# Ensure /app is in the Python path
ENV PYTHONPATH=/app

WORKDIR /app/linear_regression

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
