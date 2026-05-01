# Resumen de resultados para el informe

## Dataset y particion

- Dataset: IMDb Movie Ratings Sentiment Analysis, Kaggle.
- Archivo usado: `data/raw/movie.csv`.
- Total de reseñas: 40000.
- Columnas: `text`, `label`.
- Particion estratificada: 70% entrenamiento, 15% validacion, 15% prueba.
- Tamaños: 28000 entrenamiento, 6000 validacion, 6000 prueba.
- Etiquetas: 0 = negativo, 1 = positivo.

## Preprocesamiento

- Limpieza de HTML y etiquetas `<br />`.
- Conversion a minusculas.
- Tokenizacion con expresion regular para palabras y numeros.
- Vocabulario construido solo con entrenamiento.
- Vocabulario maximo: 30000 tokens.
- Longitud maxima por reseña: 250 tokens.
- Padding con indice 0 y token desconocido con indice 1.

## Hiperparametros

- Optimizador: AdamW.
- Funcion de perdida: BCEWithLogitsLoss.
- Learning rate: 0.002.
- Weight decay: 0.0001.
- Batch size: 256.
- Epocas maximas: 6.
- Early stopping: paciencia de 2 epocas sobre F1 de validacion.
- Embedding: 256 dimensiones.
- Dimension recurrente: 256.
- Dropout: 0.35.
- Precision mixta: activada con CUDA.
- GPU: NVIDIA GeForce RTX 4090.

## Comparacion en test

| Modelo | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| BiGRU | 0.8920 | 0.8649 | 0.9289 | 0.8958 |
| BiLSTM | 0.8835 | 0.8540 | 0.9249 | 0.8880 |
| CNN-BiLSTM | 0.8748 | 0.8658 | 0.8869 | 0.8762 |

## Interpretacion breve

El mejor modelo fue BiGRU. Su F1 superior indica un mejor balance entre precision y recall. El recall alto muestra que identifica gran parte de las reseñas positivas, aunque con una precision menor que el recall, lo que sugiere algunos falsos positivos. BiLSTM obtuvo resultados cercanos, pero con mas parametros. CNN-BiLSTM aprendio rapido patrones locales, aunque en las ultimas epocas mostro mayor riesgo de sobreajuste: la perdida de validacion subio mientras la perdida de entrenamiento seguia bajando.

## Archivos generados

- `outputs/metrics_summary.csv`: tabla cuantitativa final.
- `outputs/training_history.csv`: curvas por epoca.
- `outputs/test_predictions.csv`: predicciones de test.
- `outputs/qualitative_examples.csv`: ejemplos de aciertos y errores.
- `outputs/training_curves.png`: curva de F1/perdida.
- `outputs/confusion_matrices.png`: matrices de confusion.
- `outputs/best_model.pt`: checkpoint del mejor modelo.
