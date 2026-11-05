# Basura-del-basurero-ING.-Rosendo

**EcoScan**: clasificación y detección de residuos con YOLO26 (Ultralytics) para una app
Android en Flutter. Este repositorio contiene los **modelos finales** y la **documentación**
(informes, métricas y gráficos) de cada fase.

## Modelos

| Archivo | Tarea | Clases | Entrada | Uso |
|---|---|---|---|---|
| `models/export/basura_cls.tflite` | Clasificación (yolo26n-cls, FP16) | 7 | 224×224 | App, modo Clasificar |
| `models/export/basura_det.tflite` | Detección (yolo26n, FP32) | 5 | 416×416 | App, modo Detectar |
| `models/export/basura_*_labels.txt` | Etiquetas en el orden de los índices del modelo | | | |
| `models/cls/yolo26n-cls.pt`, `models/cls/yolo26s-cls.pt` | Pesos PyTorch de clasificación (se eligió el n) | 7 | 224 | |
| `models/det/yolo26n.pt` | Pesos PyTorch de detección | 5 | 416 | |

Los `.tflite` llevan los metadatos de Ultralytics (tarea y nombres de clase) y funcionan con el
plugin de Flutter `ultralytics_yolo` 0.7.1.

## Resultados principales (split TEST)

- **Clasificación** (yolo26n-cls, 7 clases): top-1 = 0,964, macro-F1 = 0,959.
- **Detección** (yolo26n, 5 clases): mAP@0.5 = 0,746, mAP@0.5:0.95 = 0,549, P = 0,801, R = 0,651.
  Clases más débiles: orgánico (mAP@0.5:0.95 0,407) y papel/cartón (0,523).

## Documentación

| Informe | Contenido |
|---|---|
| [reports/datasets.md](reports/datasets.md) | Registro de datasets y pesos usados, licencias y citas APA 7 |
| [reports/fase1/resumen_fase1.md](reports/fase1/resumen_fase1.md) | Dataset de clasificación: limpieza y división |
| [reports/fase2/resumen_fase2.md](reports/fase2/resumen_fase2.md) | Entrenamiento y comparación yolo26n-cls vs yolo26s-cls ([recomendación](reports/fase2/recomendacion.md)) |
| [reports/fase2b/resumen_fase2b.md](reports/fase2b/resumen_fase2b.md) | Exportación del clasificador a TFLite |
| [reports/fase2c/resumen_fase2c.md](reports/fase2c/resumen_fase2c.md) | Diagnóstico: sobreajuste, dataset y fuera de distribución |
| [reports/fase3/resumen_fase3.md](reports/fase3/resumen_fase3.md) | Detector: datos, métricas en test y [exportación a TFLite](reports/fase3/exportacion/exportacion_tflite.md) |
| [reports/fase4a/resumen_fase4a.md](reports/fase4a/resumen_fase4a.md) | App Flutter: validación del preprocesado en Python y en el celular |
| [reports/anexo_epocas.md](reports/anexo_epocas.md) | Anexo: métricas por época de los tres entrenamientos |

## Datos y licencias

- Clasificación: Kaggle `mostafaabla/garbage-classification` (ODbL-1.0).
- Detección: Kaggle `viswaprakash1990/garbage-detection` (CC BY 4.0).
- Pesos base de Ultralytics YOLO26 (AGPL-3.0); los modelos derivados se distribuyen bajo AGPL-3.0.

Los detalles y las citas están en [reports/datasets.md](reports/datasets.md). Las imágenes de los
datasets no se incluyen en el repositorio; algunos gráficos muestran ejemplos de ellas.
