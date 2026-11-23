# OCR de dígitos de punta a punta

**Objetivo:** construir un **sistema** completo, no solo un modelo: preprocesado de imagen,
detección, reconocimiento y post-procesado.

## Pipeline
1. **Binarización** con Otsu + apertura morfológica para quitar ruido.
2. **Detección** de caracteres con contornos de OpenCV y filtrado por tamaño.
3. **Normalización** de cada recorte a 28x28 centrado (idéntico en entrenamiento e inferencia).
4. **Reconocimiento** con una CNN en PyTorch.
5. **Post-procesado**: orden izquierda→derecha y detección de espacios por distancia entre cajas.

## Datos sintéticos
No se descarga nada: se generan miles de dígitos con 5 fuentes, rotación, escala, grosor,
desenfoque y ruido. Es una técnica real (así se entrenan muchos OCR comerciales).

## Métricas
- Accuracy por carácter aislado.
- Accuracy por carácter **de punta a punta** y % de documentos leídos exactamente.
- `outputs/ocr_resultados.png` muestra las cajas detectadas y la lectura.

## Retos
1. Añade letras (A-Z) y signos (`.`, `,`, `$`) para leer facturas.
2. Lee **varias líneas** agrupando cajas por coordenada `y`.
3. Compáralo con **Tesseract** (`pytesseract`) y **EasyOCR** sobre fotos reales.
4. Pasa a un modelo **CRNN + CTC loss** que lee la línea completa sin segmentar carácter a carácter.
5. Extrae campos estructurados (fecha, total, RUC/NIT) de una factura con OCR + reglas o un LLM.
