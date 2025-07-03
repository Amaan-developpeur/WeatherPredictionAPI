FROM python:3.11-slim

WORKDIR /app

# --- install dependencies ---
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# --- copy source and train ---
COPY . .
RUN python scripts/train_model.py

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
