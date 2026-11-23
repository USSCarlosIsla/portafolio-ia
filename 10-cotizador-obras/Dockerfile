FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY catalogo.py cotizador.py pronostico.py interprete_llm.py valorizaciones.py api.py ./
# El modelo entrenado y su historial viajan con la imagen (entrénalo antes con: python pronostico.py)
COPY modelos/ modelos/
COPY data/historial_precios.csv data/indices_iupc.csv data/
# Claves al ejecutar, nunca dentro de la imagen:
#   docker run -p 8000:8000 -e API_KEYS=cliente1 -e ANTHROPIC_API_KEY=... cotizador-obras
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
