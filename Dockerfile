FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Uvicorn ile başlat
CMD ["uvicorn", "fast_api:app", "--host", "0.0.0.0", "--port", "8000"]



