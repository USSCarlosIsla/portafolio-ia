FROM python:3.12-slim

# librerías del sistema que necesita onnxruntime/opencv (motor OCR)
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY esquema.py extractor_llm.py extractor_ocr.py api.py ./

# Las claves se pasan al ejecutar, nunca se copian a la imagen:
#   docker run -p 8000:8000 -e ANTHROPIC_API_KEY=... -e API_KEYS=cliente1,cliente2 extractor-facturas
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
