# Microproyecto RNN - IMDb Sentiment Analysis

Proyecto para clasificar reseñas IMDb como positivas o negativas usando redes neuronales recurrentes en PyTorch.

## Dataset

El dataset fue descargado desde Kaggle:

<https://www.kaggle.com/datasets/yasserh/imdb-movie-ratings-sentiment-analysis/data>

Archivo local esperado:

```text
data/raw/movie.csv
```

Si necesitas descargarlo de nuevo, configura tus credenciales de Kaggle (`~/.kaggle/kaggle.json` o variables `KAGGLE_USERNAME` y `KAGGLE_KEY`) y ejecuta:

```powershell
py -m kaggle.cli datasets download -d yasserh/imdb-movie-ratings-sentiment-analysis -p data\raw --unzip
```

## Ejecución

Entrenamiento completo con GPU:

```powershell
py train_compare_rnn.py --epochs 6 --batch-size 256 --max-vocab 30000 --max-len 250 --amp
```

Prueba rápida:

```powershell
py train_compare_rnn.py --epochs 1 --batch-size 128 --sample-size 3000 --amp
```

Salidas generadas:

```text
outputs/metrics_summary.csv
outputs/test_predictions.csv
outputs/training_curves.png
outputs/confusion_matrices.png
outputs/qualitative_examples.csv
outputs/best_model.pt
```

## Modelos Comparados

1. **BiLSTM**: buena capacidad para dependencias largas en texto.
2. **BiGRU**: similar a LSTM pero más liviano y rápido.
3. **CNN-BiLSTM**: combina patrones locales de n-gramas con contexto secuencial.

Las métricas principales son accuracy, precision, recall y F1-score, tal como pide el enunciado.
