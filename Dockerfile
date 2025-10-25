# 1️⃣ Python tabanlı hafif imajı kullan
FROM python:3.11-slim

# 2️⃣ Container içinde çalışma klasörünü belirle
WORKDIR /app

# 3️⃣ Gerekli kütüphaneleri yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4️⃣ Kodları ve modeli kopyala
COPY app ./app
COPY artifacts ./artifacts

# 5️⃣ Uygulama hangi portta dinleyecek (Cloud Run vs. için)
ENV PORT=8080

# 6️⃣ FastAPI’yi uvicorn ile başlat
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8080"]

COPY static ./static