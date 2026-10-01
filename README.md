# Caso 1 — Reconocimiento de dígitos MNIST en Android (TensorFlow Lite)

App Android que reconoce dígitos escritos a mano (0-9) **directamente en el dispositivo**, sin conexión a Internet, usando una CNN entrenada con MNIST y convertida a TensorFlow Lite (LiteRT).

## Estructura

```
.
├── entrenamiento/        # Entrenamiento del modelo en Python
│   ├── train_mnist.py    # Dataset → CNN → TFLite → verificación → métricas
│   ├── requirements.txt
│   └── salida/           # Modelo .tflite/.keras, curvas, matriz de confusión, métricas
├── caso1/                # Proyecto Android (Kotlin)
│   └── app/src/main/
│       ├── java/com/grupo/caso1/
│       │   ├── MainActivity.kt      # UI, barras de probabilidad, muestras
│       │   ├── DrawView.kt          # Lienzo táctil negro con trazo blanco
│       │   └── DigitClassifier.kt   # Preprocesado MNIST + Interpreter TFLite
│       └── assets/                  # mnist.tflite y muestras reales de test
├── evidencias/           # Capturas de las pruebas en emulador y celular real
└── caso1.apk             # APK compilado listo para instalar
```

## Entrenamiento

```bash
cd entrenamiento
pip install -r requirements.txt
python train_mnist.py
```

- CNN: Conv(32) → Pool → Conv(64) → Pool → Dropout(0.3) → Dense(128) → Dense(10, softmax)
- Aumento de datos: rotación, zoom y traslación leves
- 8 épocas, batch 128, cuantización de rango dinámico (pesos int8)

| Métrica | Valor |
|---|---|
| Accuracy Keras (test) | 98.94 % |
| Accuracy TFLite (test) | 98.94 % |
| Parámetros | 225 034 |
| Tamaño .keras → .tflite | 2676 KB → 228 KB |
| Entrada / salida | `[1,28,28,1]` float32 / `[1,10]` |

## App Android

1. El usuario dibuja en `DrawView` (fondo negro, trazo blanco, como MNIST).
2. `DigitClassifier` recorta el dígito, lo escala a 20×20, lo centra en 28×28 por centro de masa y normaliza a `[0,1]`.
3. `Interpreter.run` produce 10 probabilidades; se muestran el dígito, la confianza, el tiempo de inferencia, el tensor de entrada y las barras por clase.
4. El botón **Muestra** carga imágenes reales del set de test de MNIST.

El manifiesto no pide permiso `INTERNET`: la inferencia es 100 % local.

Compilar: abrir `caso1/` en Android Studio, o `./gradlew assembleDebug`. Requiere minSdk 24.
