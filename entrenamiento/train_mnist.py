
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

SEED = 42
EPOCHS = 8
BATCH = 128
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "salida")
os.makedirs(OUT, exist_ok=True)
tf.random.set_seed(SEED)
np.random.seed(SEED)

# 1. Dataset: 60 000 imágenes de entrenamiento y 10 000 de test, 28x28 en gris (0-255)
(x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.mnist.load_data()
x_tr = (x_tr[..., None] / 255.0).astype("float32")
x_te = (x_te[..., None] / 255.0).astype("float32")
print("train:", x_tr.shape, "test:", x_te.shape)

fig, axes = plt.subplots(2, 8, figsize=(10, 3))
for ax, img, lbl in zip(axes.flat, x_tr[:16], y_tr[:16]):
    ax.imshow(img[..., 0], cmap="gray")
    ax.set_title(str(lbl))
    ax.axis("off")
fig.suptitle("Ejemplos del dataset MNIST")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "muestras.png"), dpi=120)
plt.close(fig)

# 2. Aumento de datos leve: los dígitos dibujados en el celular no están tan
#    centrados ni derechos como los de MNIST.
augment = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.08, fill_mode="constant"),
    tf.keras.layers.RandomZoom(0.1, fill_mode="constant"),
    tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode="constant"),
])
train_ds = (tf.data.Dataset.from_tensor_slices((x_tr, y_tr))
            .shuffle(10_000, seed=SEED)
            .batch(BATCH)
            .map(lambda x, y: (augment(x, training=True), y),
                 num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE))
test_ds = tf.data.Dataset.from_tensor_slices((x_te, y_te)).batch(BATCH)

# 3. Modelo: CNN pequeña (entrada 28x28x1 -> 10 probabilidades)
model = tf.keras.Sequential([
    tf.keras.layers.Input((28, 28, 1), name="imagen"),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax", name="probabilidades"),
])
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

t0 = time.time()
hist = model.fit(train_ds, epochs=EPOCHS, validation_data=test_ds, verbose=2)
train_secs = time.time() - t0
_, keras_acc = model.evaluate(test_ds, verbose=0)
print(f"Precisión Keras en test: {keras_acc:.4f}")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.5))
ep = range(1, EPOCHS + 1)
a1.plot(ep, hist.history["accuracy"], "o-", label="entrenamiento")
a1.plot(ep, hist.history["val_accuracy"], "o-", label="test")
a1.set_title("Accuracy"); a1.set_xlabel("época"); a1.legend(); a1.grid(alpha=.3)
a2.plot(ep, hist.history["loss"], "o-", label="entrenamiento")
a2.plot(ep, hist.history["val_loss"], "o-", label="test")
a2.set_title("Loss"); a2.set_xlabel("época"); a2.legend(); a2.grid(alpha=.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "curvas.png"), dpi=120)
plt.close(fig)

# 4. Conversión a TensorFlow Lite con cuantización de rango dinámico
#    (pesos en int8, entrada/salida siguen en float32)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_bytes = converter.convert()
tflite_path = os.path.join(OUT, "mnist.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_bytes)

# 5. Verificar el .tflite con el intérprete (lo mismo que hará Android)
interp = tf.lite.Interpreter(model_path=tflite_path)
interp.allocate_tensors()
inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]
print("Entrada TFLite:", inp["shape"], inp["dtype"])
print("Salida  TFLite:", out["shape"], out["dtype"])

preds = np.empty(len(x_te), dtype=np.int64)
t0 = time.time()
for i, img in enumerate(x_te):
    interp.set_tensor(inp["index"], img[None])
    interp.invoke()
    preds[i] = interp.get_tensor(out["index"])[0].argmax()
ms_per_img = (time.time() - t0) * 1000 / len(x_te)
tflite_acc = float((preds == y_te).mean())
print(f"Precisión TFLite en test: {tflite_acc:.4f}  ({ms_per_img:.3f} ms/imagen)")

cm = tf.math.confusion_matrix(y_te, preds, num_classes=10).numpy()
fig, ax = plt.subplots(figsize=(6, 5.5))
ax.imshow(cm, cmap="Blues")
for r in range(10):
    for c in range(10):
        ax.text(c, r, cm[r, c], ha="center", va="center", fontsize=7,
                color="white" if cm[r, c] > cm.max() / 2 else "black")
ax.set_xticks(range(10)); ax.set_yticks(range(10))
ax.set_xlabel("predicho"); ax.set_ylabel("real")
ax.set_title(f"Matriz de confusión TFLite (acc = {tflite_acc:.2%})")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "matriz_confusion.png"), dpi=120)
plt.close(fig)

# 6. Tres imágenes reales de test para las pruebas de la app
for digit in (3, 7, 0):
    idx = int(np.where((y_te == digit) & (preds == digit))[0][0])
    Image.fromarray((x_te[idx, ..., 0] * 255).astype("uint8")).save(
        os.path.join(OUT, f"muestra_{digit}.png"))

keras_path = os.path.join(OUT, "mnist.keras")
model.save(keras_path)
metrics = {
    "epocas": EPOCHS,
    "batch": BATCH,
    "segundos_entrenamiento": round(train_secs, 1),
    "parametros": int(model.count_params()),
    "accuracy_keras_test": round(float(keras_acc), 4),
    "accuracy_tflite_test": round(tflite_acc, 4),
    "ms_por_imagen_pc": round(ms_per_img, 3),
    "kb_keras": round(os.path.getsize(keras_path) / 1024, 1),
    "kb_tflite": round(len(tflite_bytes) / 1024, 1),
    "entrada": [int(d) for d in inp["shape"]],
    "salida": [int(d) for d in out["shape"]],
    "tensorflow": tf.__version__,
    "history": {k: [round(float(v), 4) for v in vals]
                for k, vals in hist.history.items()},
}
with open(os.path.join(OUT, "metricas.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)
print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=2))
