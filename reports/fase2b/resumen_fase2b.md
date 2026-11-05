# Fase 2B — Exportación de yolo26n-cls a TFLite

## Procedimiento

- Modelo origen: `models/cls/yolo26n-cls.pt` (elegido en la fase 2). imgsz=224.
- Exportación con Ultralytics `model.export(format='saved_model')`: PyTorch → ONNX → TensorFlow SavedModel → TFLite con onnx2tf, que genera a la vez las variantes float32 y float16. No se usó `format='tflite'` porque desde Ultralytics 8.4.83 se redirige al exportador LiteRT, que solo funciona en Linux x86 y macOS (`AssertionError: LiteRT export only supported on Linux x86 and macOS`).
- Entorno: `.venv-tflite`, separado del `.venv` principal porque TensorFlow 2.19 exige numpy < 2.2 y protobuf 5, incompatibles con las versiones usadas en las fases 1–2. Versiones en `requirements-tflite.txt`.
- Verificación sobre TEST (1308 imágenes) en CPU, con el mismo preprocesado de Ultralytics para las tres variantes. Es solo una verificación de la conversión: el modelo ya estaba elegido.
- Latencia: batch=1, 200 imágenes tras 20 de calentamiento, CPU del equipo de desarrollo (no del teléfono). «Total» incluye preprocesado; «inferencia», solo la red.

## .pt vs .tflite FP32 vs .tflite FP16

| variante | top-1 | macro-F1 | Δ top-1 vs .pt | pred. distintas vs .pt | máx. Δ prob. | MB | ms CPU (total) | ms CPU (inferencia) |
|---|---|---|---|---|---|---|---|---|
| pt | 0.9641 | 0.9594 | +0.00 pts | 0 / 1308 | 0.0000 | 3.06 | 12.5 | 8.3 |
| tflite_fp32 | 0.9641 | 0.9594 | +0.00 pts | 0 / 1308 | 0.0000 | 5.89 | 20.0 | 15.2 |
| tflite_fp16 | 0.9641 | 0.9594 | +0.00 pts | 0 / 1308 | 0.0176 | 2.97 | 19.9 | 14.9 |

Control: el .pt evaluado aquí en CPU obtiene top-1 = 0.9641; en la fase 2, en GPU, obtuvo 0.9641.

## Modelo elegido para la app

- **FP16** → `models/export/basura_cls.tflite` (2.97 MB). Criterio: FP16 si pierde ≤ 0.5 puntos de top-1 frente al .pt (pérdida observada: +0.00 puntos).
- Entrada: tensor `float32` de forma [1, 224, 224, 3] (NHWC), RGB, valores en [0, 1] (píxel / 255, sin restar media ni dividir por desviación).
- Salida: `float32` de forma [1, 7], probabilidades (softmax incluido; suma de control = 1.0).
- Orden de las clases (índice de la salida) en `models/export/basura_cls_labels.txt`: 0=general, 1=metal, 2=organico, 3=papel_carton, 4=peligroso, 5=plastico, 6=vidrio. **Ojo:** es orden alfabético, distinto del de `configs/clases.yaml`.
- Preprocesado que debe replicar la app (el de Ultralytics para clasificación): redimensionar el lado menor a 224 px manteniendo la proporción (interpolación bilineal), recortar el centro a 224×224 y normalizar a [0, 1]. Si la app solo estira la imagen a 224×224, la precisión puede bajar respecto a estas cifras.

## Observaciones

- **FP16** guarda los pesos en media precisión (la mitad de tamaño), pero la entrada y la salida siguen siendo float32 y en CPU los pesos se convierten a float32 al cargar; por eso las probabilidades cambian muy poco y ninguna predicción cambia.
- **Latencia en PC:** el intérprete TFLite es más lento que PyTorch en este i7 de escritorio. Es lo esperado: TFLite está optimizado para ARM (XNNPACK, delegados GPU/NNAPI del teléfono), así que estas cifras no predicen la latencia en el celular; habrá que medirla en la app.
- Para evaluar el .tflite sobre fotos reales: `.venv-tflite\Scripts\python.exe scripts/07_evaluar_real.py --modelo models/export/basura_cls.tflite --ref reports/fase2b/tflite_fp16/metricas.json` (el .venv principal no tiene TensorFlow).
