# Fase 3 — Exportación del detector a TFLite

- Origen: `models/det/yolo26n.pt`, imgsz=416. Ruta de la fase 2B: `format='saved_model'` (PyTorch → ONNX → TF SavedModel → TFLite con onnx2tf), en `.venv-tflite`.
- Verificación en CPU sobre TEST (1524 imágenes), con el preprocesado de Ultralytics. Coincidencia de detecciones con el umbral de la app (conf ≥ 0.4): una detección del .pt se reproduce si la variante tiene una de la misma clase con IoU ≥ 0.5.

| variante | P | R | mAP@0.5 | mAP@0.5:0.95 | Δ mAP@0.5:0.95 | detecc. del .pt reproducidas | imágenes idénticas | IoU medio | Δ conf. media | MB | ms CPU (inferencia) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pt | 0.801 | 0.651 | 0.746 | 0.549 | +0.00 pts | 100.0 % | 100.0 % | 1.000 | 0.0000 | 5.18 | 28.1 |
| tflite_fp32 | 0.779 | 0.655 | 0.740 | 0.541 | -0.76 pts | 100.0 % | 100.0 % | 1.000 | 0.0000 | 9.25 | 45.9 |
| tflite_fp16 | 0.781 | 0.653 | 0.741 | 0.541 | -0.76 pts | 99.9 % | 99.3 % | 0.999 | 0.0007 | 4.69 | 45.8 |

## Compatibilidad con ultralytics_yolo 0.7.1

- Metadatos anexados al .tflite (ZIP con `metadata.json`, que el plugin lee en `YOLOFileUtils.loadMetadataFromAppendedZip`): task = `detect`, names = {0: 'metal', 1: 'organico', 2: 'papel_carton', 3: 'plastico', 4: 'vidrio'} (coinciden con data.yaml), imgsz = [416, 416], end2end = False.
- Entrada [1, 416, 416, 3] (NHWC, RGB/255). Salida [1, 9, 3549] (float32): [4 + 5 clases, anchors] = xywh en píxeles del input + score por clase. Como la última dimensión es mayor que la anterior, el plugin usa su decodificador nativo `ObjectDetector.postprocess` (JNI) con NMS (iou 0.7) y umbral de confianza; detecta que las coordenadas están en píxeles (máx. 211 > 2) y deshace el letterbox.
- **Sobre «YOLO26 sin NMS»:** la cabeza end-to-end (salida [1, 300, 6], sin NMS) solo se conserva exportando con `nms=False`; con los pesos de prueba esa variante no reproducía las confianzas del .pt (ya en el ONNX), mientras que la exportación por defecto sí las reproduce exactamente. Por eso la app usa la salida one-to-many + NMS del plugin, que da las mismas detecciones que el .pt evaluado.
- La verificación con Ultralytics requiere `corregir_lector_tflite()`: su lector de .tflite (8.4.191) asume coordenadas normalizadas y las multiplica por 416; sin el arreglo todas las cajas salen en (416, 416).
- Resultado: **compatible**.

## Modelo para la app

- **FP32** → `models/export/basura_det.tflite` (9.25 MB); criterio: FP16 si pierde ≤ 0,5 puntos de mAP@0.5:0.95 (pérdida observada +0.76 puntos). Etiquetas en `models/export/basura_det_labels.txt` (orden de índices del modelo).
