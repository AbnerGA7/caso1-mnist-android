package com.grupo.caso1

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.os.SystemClock
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * Envuelve el modelo mnist.tflite.
 * Flujo: Bitmap (entrada) -> ByteBuffer float32 [1,28,28,1] (tensor)
 *        -> Interpreter.run (modelo) -> FloatArray[10] (resultado)
 */
class DigitClassifier(context: Context) : AutoCloseable {

    class Result(
        val digit: Int,
        val confidence: Float,
        val probabilities: FloatArray,
        val inferenceMs: Double,
        val tensorPreview: Bitmap,
    )

    private val interpreter: Interpreter

    init {
        val options = Interpreter.Options().apply { setNumThreads(2) }
        interpreter = Interpreter(loadModel(context, MODEL_FILE), options)
    }

    val inputShape: IntArray get() = interpreter.getInputTensor(0).shape()
    val outputShape: IntArray get() = interpreter.getOutputTensor(0).shape()

    fun classify(drawing: Bitmap): Result {
        val input28 = toMnistFormat(drawing)
        val input = toTensor(input28)
        val output = Array(1) { FloatArray(NUM_CLASSES) }

        val start = SystemClock.elapsedRealtimeNanos()
        interpreter.run(input, output)          // inferencia en el dispositivo
        val ms = (SystemClock.elapsedRealtimeNanos() - start) / 1e6

        val probs = output[0]
        val digit = probs.indices.maxBy { probs[it] }
        return Result(digit, probs[digit], probs, ms, input28)
    }

    /**
     * Reproduce el formato de MNIST: dígito blanco sobre negro, recortado,
     * escalado a una caja de 20x20 y centrado en un lienzo de 28x28.
     */
    private fun toMnistFormat(src: Bitmap): Bitmap {
        val w = src.width
        val h = src.height
        val pixels = IntArray(w * h)
        src.getPixels(pixels, 0, w, 0, 0, w, h)

        var minX = w; var minY = h; var maxX = -1; var maxY = -1
        for (y in 0 until h) for (x in 0 until w) {
            if (Color.red(pixels[y * w + x]) > 30) {
                if (x < minX) minX = x
                if (x > maxX) maxX = x
                if (y < minY) minY = y
                if (y > maxY) maxY = y
            }
        }

        val out = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(out)
        canvas.drawColor(Color.BLACK)
        if (maxX < 0) return out                // lienzo vacío

        val boxW = maxX - minX + 1
        val boxH = maxY - minY + 1
        val scale = DIGIT_BOX.toFloat() / max(boxW, boxH)
        val dw = boxW * scale
        val dh = boxH * scale
        val left = (SIZE - dw) / 2f
        val top = (SIZE - dh) / 2f
        val paint = Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG)
        canvas.drawBitmap(
            src, Rect(minX, minY, maxX + 1, maxY + 1),
            RectF(left, top, left + dw, top + dh), paint
        )
        return centerByMass(out)
    }

    /** MNIST centra cada dígito por su centro de masa, no por su caja. */
    private fun centerByMass(bmp: Bitmap): Bitmap {
        val px = IntArray(SIZE * SIZE)
        bmp.getPixels(px, 0, SIZE, 0, 0, SIZE, SIZE)
        var sum = 0.0; var sx = 0.0; var sy = 0.0
        for (y in 0 until SIZE) for (x in 0 until SIZE) {
            val v = Color.red(px[y * SIZE + x]).toDouble()
            sum += v; sx += x * v; sy += y * v
        }
        if (sum == 0.0) return bmp
        // Limitar el desplazamiento al margen libre para no recortar el dígito
        val margin = (SIZE - DIGIT_BOX) / 2
        val dx = (SIZE / 2.0 - sx / sum).roundToInt().coerceIn(-margin, margin)
        val dy = (SIZE / 2.0 - sy / sum).roundToInt().coerceIn(-margin, margin)
        val shifted = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888)
        Canvas(shifted).apply {
            drawColor(Color.BLACK)
            drawBitmap(bmp, dx.toFloat(), dy.toFloat(), null)
        }
        return shifted
    }

    /** 28x28 píxeles -> 784 floats en [0,1] (mismo /255 que en el entrenamiento). */
    private fun toTensor(bmp28: Bitmap): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(4 * SIZE * SIZE).order(ByteOrder.nativeOrder())
        val px = IntArray(SIZE * SIZE)
        bmp28.getPixels(px, 0, SIZE, 0, 0, SIZE, SIZE)
        for (p in px) {
            val gray = (Color.red(p) + Color.green(p) + Color.blue(p)) / 3f
            buffer.putFloat(gray / 255f)
        }
        buffer.rewind()
        return buffer
    }

    override fun close() = interpreter.close()

    companion object {
        const val MODEL_FILE = "mnist.tflite"
        const val SIZE = 28
        const val DIGIT_BOX = 20
        const val NUM_CLASSES = 10

        private fun loadModel(context: Context, name: String): MappedByteBuffer =
            context.assets.openFd(name).use { fd ->
                FileInputStream(fd.fileDescriptor).channel.use { ch ->
                    ch.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
                }
            }
    }
}
