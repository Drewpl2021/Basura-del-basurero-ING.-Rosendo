# Registro de datos y pesos usados en el proyecto

Metadatos de Kaggle obtenidos el **2026-10-28** con la API oficial (`kaggle datasets metadata` y
`KaggleApi.dataset_list`, cliente kaggle 2.2.5), no de memoria. Los conteos "usado" salen de los
scripts del proyecto (`reports/fase1/`, `reports/fase3/`). Citas en APA 7 (versión en español).

## Tabla resumen

| Recurso | Uso | Autor / propietario | Versión | Licencia (según la fuente) | Descarga | Original | Usado |
|---|---|---|---|---|---|---|---|
| Garbage Classification (12 classes) — Kaggle `mostafaabla/garbage-classification` | Clasificación (fases 1, 2, 2B, 2C, 4A) | Mostafa Mohamed | v1 (actualizado 2021-01-24) | ODbL-1.0 (base de datos); contenidos © autores originales | 2026-10-28 | 15.515 imágenes, 12 clases | 8.714 imágenes, 7 clases |
| Garbage Detection – 6 Waste Categories — Kaggle `viswaprakash1990/garbage-detection` | Detección (fase 3) | Viswa Prakash | v1 (actualizado 2025-05-20) | CC BY 4.0 | 2026-10-28 | 10.464 imágenes, 74.090 cajas, 6 clases | 10.167 imágenes, 45.349 cajas, 5 clases |
| ↳ fuente original: Roboflow Universe `material-identification/garbage-classification-3` | (origen del anterior) | workspace *material-identification* | v2 | CC BY 4.0 (según `data.yaml` del dataset) | — | — | — |
| Pesos `yolo26n-cls.pt` (y `yolo26s-cls.pt`, comparado en la fase 2) | Preentrenamiento del clasificador | Ultralytics | Ultralytics 8.3.100, 2025-10-23 | AGPL-3.0 | 2026-10-28 | preentrenado en ImageNet, 224 px | ajuste fino en data/cls |
| Pesos `yolo26n.pt` | Preentrenamiento del detector | Ultralytics | Ultralytics 8.3.222, 2025-12-15 | AGPL-3.0 | 2026-10-28 | preentrenado en COCO (80 clases), 640 px | ajuste fino en data/det a 416 px |
| `bus.jpg`, `zidane.jpg` (assets del paquete `ultralytics`) | Solo la prueba fuera de distribución de la fase 2C | Ultralytics | paquete 8.4.191 | AGPL-3.0 (paquete) | con el paquete | 2 fotos | 2 fotos, sin entrenar |

## 1. Clasificación: Garbage Classification (12 classes)

- **URL:** https://www.kaggle.com/datasets/mostafaabla/garbage-classification
- **Autor:** Mostafa Mohamed (usuario `mostafaabla`). Título según Kaggle: *Garbage Classification (12 classes)*.
- **Versión:** 1 (`current_version_number = 1`), última actualización el 2021-01-24.
- **Licencia:** Kaggle declara **ODbL-1.0**, con `license_name` "Database: Open Database, Contents: © Original Authors". La licencia abierta cubre la base de datos, pero las **imágenes siguen siendo de sus autores**: según la descripción del dataset, la mayoría se obtuvo por *web scraping*, solo con fines de investigación y sin ánimo de lucro. Uso académico, sin redistribuir las imágenes.
- **Fuentes que declara el autor** (campo *Acknowledgements*):
  - Las clases *clothes* y el 22 % de *shoes* vienen del Clothing dataset (https://www.kaggle.com/agrigorev/clothing-dataset-full).
  - Alrededor del 29 % de las otras 9 clases viene de https://www.kaggle.com/asdasdasasdas/garbage-classification.
  - El resto se obtuvo por web scraping.
- **Imágenes:** la descripción de Kaggle dice 15.150, pero el archivo descargado contiene **15.515** (se contaron en disco).
- **Usadas tras la limpieza (fase 1):** **8.714**, divididas 6.099 / 1.307 / 1.308 (70/15/15, semilla 42). Se eliminaron 6.801:
  - duplicados exactos y casi duplicados (pHash y dHash con distancia ≤ 4);
  - grupos con etiquetas en conflicto;
  - submuestreo de *clothes* y *shoes* a 300 imágenes cada una.

  El detalle está en `reports/fase1/limpieza.csv`.
- **Mapeo de clases** (`configs/clases.yaml`):

| original | clase del proyecto |
|---|---|
| plastic | plastico |
| paper, cardboard | papel_carton |
| brown-glass, green-glass, white-glass | vidrio |
| metal | metal |
| biological | organico |
| trash, clothes, shoes | general |
| battery | peligroso |

- **Cita APA 7:**
  Mohamed, M. (2021). *Garbage classification (12 classes)* (Versión 1) [Conjunto de datos]. Kaggle. https://www.kaggle.com/datasets/mostafaabla/garbage-classification

## 2. Detección: Garbage Detection – 6 Waste Categories

- **URL:** https://www.kaggle.com/datasets/viswaprakash1990/garbage-detection
- **Autor:** Viswa Prakash (usuario `viswaprakash1990`). Título según Kaggle: *Garbage Detection – 6 Waste Categories*, con el subtítulo "Dataset in YOLO Format".
- **Versión:** 1, última actualización el 2025-05-20.
- **Licencia:** **CC BY 4.0** (Attribution 4.0 International). Obliga a citar la fuente.
- **Fuente original:** Kaggle indica "Originally sourced from Roboflow". El `data.yaml` incluido identifica el proyecto: `workspace: material-identification`, `project: garbage-classification-3`, `version: 2`, `license: CC BY 4.0`, `url: https://universe.roboflow.com/material-identification/garbage-classification-3/dataset/2`.
  - *No se pudo verificar la página de Roboflow en línea* (respondió HTTP 403 el 2026-10-28). Los datos de la fuente original vienen del `data.yaml` distribuido con el dataset.
- **Formato:** YOLO (cajas `clase x y w h` normalizadas), con todas las imágenes de 416×416. Viene dividido en train/valid/test (7.324 / 2.098 / 1.042), pero ese split **no se usó** porque test no tenía vidrio y val casi no tenía papel.
- **Original:** 10.464 imágenes y 74.090 cajas, en 6 clases (BIODEGRADABLE, CARDBOARD, GLASS, METAL, PAPER, PLASTIC).
- **Usado tras la limpieza (fase 3, `scripts/12_preparar_det.py`):** **10.167 imágenes y 45.349 cajas**.
  - Se excluyeron 297 imágenes con más de 50 cajas (montones etiquetados caja a caja: el 2,8 % de las imágenes, pero el 39 % de las cajas).
  - Se eliminaron 7 cajas inválidas y 36 cajas de menos de 4 px.
  - Nuevo split 70/15/15 estratificado por la clase principal de cada imagen, con las imágenes de pHash idéntico en el mismo split (semilla 42): 7.119 / 1.524 / 1.524.

  El detalle está en `reports/fase3/division_det.csv`, `dataset_det.csv` y `limpieza_det.csv`.
- **Mapeo de clases:**

| original | clase del proyecto |
|---|---|
| BIODEGRADABLE | organico |
| CARDBOARD, PAPER | papel_carton |
| GLASS | vidrio |
| METAL | metal |
| PLASTIC | plastico |

  (No existen `general` ni `peligroso` en detección.)
- **Citas APA 7:**
  - Prakash, V. (2025). *Garbage detection – 6 waste categories* (Versión 1) [Conjunto de datos]. Kaggle. https://www.kaggle.com/datasets/viswaprakash1990/garbage-detection
  - Material Identification. (s.f.). *Garbage classification 3* (Versión 2) [Conjunto de datos]. Roboflow Universe. https://universe.roboflow.com/material-identification/garbage-classification-3/dataset/2

## 3. Pesos preentrenados (Ultralytics YOLO26)

El conjunto de preentrenamiento se comprobó leyendo `train_args` dentro de cada `.pt`:

| archivo | `train_args.data` | imgsz | creado con | licencia en el checkpoint |
|---|---|---|---|---|
| `yolo26n-cls.pt` | `../datasets/imagenet` → **ImageNet** | 224 | Ultralytics 8.3.100 (2025-10-23) | AGPL-3.0 |
| `yolo26s-cls.pt` | `../datasets/imagenet` → **ImageNet** | 224 | Ultralytics 8.3.100 (2025-10-22) | AGPL-3.0 |
| `yolo26n.pt` | `coco.yaml` → **COCO** (80 clases) | 640 | Ultralytics 8.3.222 (2025-12-15) | AGPL-3.0 |

- **Repositorio:** https://github.com/ultralytics/ultralytics
- **Documentación de YOLO26:** https://docs.ultralytics.com/models/yolo26/. Se consultó el 2026-10-28. La página confirma que `yolo26n.pt` está preentrenado en COCO y que el código y los modelos se distribuyen bajo AGPL-3.0 o licencia Enterprise.
- **Versión usada:** paquete `ultralytics` **8.4.191** para entrenar, evaluar y exportar.
- **Cita** (es la que piden el `CITATION.cff` del repositorio, como *preferred-citation*, y la documentación de YOLO26):
  Jocher, G., Qiu, J., Liu, M., Lyu, S., Akyon, F. C., & Kalfaoglu, M. E. (2026). *Ultralytics YOLO26: Unified real-time end-to-end vision models* (arXiv:2606.03748). arXiv. https://doi.org/10.48550/arXiv.2606.03748
- **Cita del software** (metadatos del `CITATION.cff`: autores Jocher, Qiu y Chaurasia; licencia AGPL-3.0. Se indica la versión usada en el proyecto):
  Jocher, G., Qiu, J., & Chaurasia, A. (2023). *Ultralytics YOLO* (Versión 8.4.191) [Software]. GitHub. https://github.com/ultralytics/ultralytics
- **Conjuntos de preentrenamiento** (referencias de los artículos originales):
  - Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., & Fei-Fei, L. (2009). ImageNet: A large-scale hierarchical image database. En *2009 IEEE Conference on Computer Vision and Pattern Recognition* (pp. 248–255). IEEE. https://doi.org/10.1109/CVPR.2009.5206848
  - Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., & Zitnick, C. L. (2014). Microsoft COCO: Common objects in context. En *Computer Vision – ECCV 2014* (pp. 740–755). Springer. https://doi.org/10.1007/978-3-319-10602-1_48

## 4. Otros datos usados

- **Validación del preprocesado de la app (fase 4A):** 50 imágenes del split *test* de clasificación (las mismas de la sección 1; no son datos nuevos).
- **Prueba fuera de distribución (fase 2C):** imágenes sintéticas generadas por el script (colores lisos, ruido, texturas) y `bus.jpg`/`zidane.jpg` del paquete `ultralytics`. No se usaron para entrenar.
- **`data/real/`** (fotos propias con el celular): todavía **no se ha recogido**.

## 5. Datos propuestos pero NO usados

- **RealWaste** (UCI Machine Learning Repository) se **propuso** en la fase 2C (`reports/fase2c/resumen_fase2c.md`) como fuente de fotos reales de residuos, pero **NO se descargó ni se usó** en ningún modelo de este proyecto. Lo mismo vale para TACO, HaGRID, DTD y Garbage Classification v2, mencionados en esa misma propuesta. Sus datos no se verificaron en línea, así que no se citan aquí.
