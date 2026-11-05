# Fase 4A — App EcoScan (modo Clasificar)

App Flutter en `app/ecoscan/`, librería **ultralytics_yolo 0.7.1** (Android: LiteRT 2.3 + CameraX).
Modelo: `models/export/basura_cls.tflite` (yolo26n-cls FP16, entrada 224×224).

## Por qué ultralytics_yolo
- Lee las clases del `metadata.json` que la exportación de Ultralytics anexa al `.tflite` (ZIP al
  final del archivo). Nuestro modelo lo tiene y sus `names` coinciden con `basura_cls_labels.txt`
  (la app lo verifica al arrancar y usa siempre el índice + el archivo de labels).
- Preprocesado de clasificación (`ImageUtils.prepareBitmapForModel(centerCrop=true)`): escala
  `max(224/w, 224/h)` + recorte central, bilineal, RGB/255 → equivale a Ultralytics salvo el
  antialias de PIL al reducir.
- Es la misma librería que se usará para detección (Fase 3).

## Validación del preprocesado (mismo .tflite; difiere solo el preprocesado)
Muestra: 50 imágenes de `data/cls/test/` estratificadas (semilla 42); las 10 primeras cubren las 7 clases.

| Comparación con Ultralytics (Python) | 10 obligatorias | 50 | Δ conf. media (máx) |
|---|---|---|---|
| Réplica del plugin en Python (`09_validar_preproc_app.py`) | 10/10 | 48/50 | 0,7 pp (19 pp) |
| Celular, plugin en GPU (`10_comparar_celular.py`) | 10/10 | 48/50 | 0,9 pp (22 pp) |
| Celular, plugin en CPU | 10/10 | 48/50 | 0,9 pp (22 pp) |

Las 2 discrepancias son las mismas en los tres casos y ya eran dudosas para Ultralytics
(`battery818`: peligroso 0,56; `cardboard132`: general 0,52, real papel_carton); en la app caen bajo
el umbral del 60 % ("No estoy seguro…"). Causa: Android no aplica antialias al reducir.

## Tiempos en el celular (Samsung SM-A556E, Android 16, APK release, cámara en vivo)
Media por bloques de 100 frames (logcat `EcoScan|…`), régimen estable:

| Acelerador | Inferencia | Preproc. + inferencia + post | FPS |
|---|---|---|---|
| **CPU (por defecto)** | **~13 ms** | **~15 ms** | ~24 |
| GPU | ~22 ms | ~27 ms | ~24 |

Con un modelo tan pequeño la CPU es más rápida (el traspaso a la GPU cuesta más de lo que ahorra);
los FPS los limita la cámara. GPU disponible con `--dart-define=USAR_GPU=true`.
