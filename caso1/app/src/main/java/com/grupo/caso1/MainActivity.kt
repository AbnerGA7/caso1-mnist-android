package com.grupo.caso1

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Typeface
import android.graphics.drawable.BitmapDrawable
import android.os.Bundle
import android.util.Log
import android.view.Gravity
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.google.android.material.button.MaterialButton
import java.util.Locale

class MainActivity : AppCompatActivity() {

    private lateinit var classifier: DigitClassifier
    private lateinit var drawView: DrawView
    private lateinit var txtResult: TextView
    private lateinit var txtInfo: TextView
    private lateinit var imgTensor: ImageView
    private val bars = mutableListOf<ProgressBar>()
    private val labels = mutableListOf<TextView>()

    private val samples = listOf("muestra_3.png", "muestra_7.png", "muestra_0.png")
    private var sampleIndex = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.root)) { v, insets ->
            val sys = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(sys.left, sys.top, sys.right, sys.bottom)
            insets
        }

        classifier = DigitClassifier(this)
        Log.i(TAG, "Entrada ${classifier.inputShape.contentToString()} " +
                "Salida ${classifier.outputShape.contentToString()}")

        drawView = findViewById(R.id.drawView)
        txtResult = findViewById(R.id.txtResult)
        txtInfo = findViewById(R.id.txtInfo)
        imgTensor = findViewById(R.id.imgTensor)
        buildProbabilityBars(findViewById(R.id.probsContainer))

        drawView.onStrokeFinished = { predict() }
        findViewById<MaterialButton>(R.id.btnPredict).setOnClickListener { predict() }
        findViewById<MaterialButton>(R.id.btnClear).setOnClickListener { reset() }
        findViewById<MaterialButton>(R.id.btnSample).setOnClickListener { loadSample() }
    }

    private fun predict() {
        val bmp = drawView.getBitmap() ?: return
        val r = classifier.classify(bmp)

        txtResult.text = r.digit.toString()
        txtInfo.text = String.format(
            Locale.US,
            "Confianza %.1f%% · Inferencia %.2f ms · en el dispositivo (sin Internet)",
            r.confidence * 100, r.inferenceMs
        )
        showTensor(r.tensorPreview)
        r.probabilities.forEachIndexed { i, p ->
            bars[i].progress = (p * 1000).toInt()
            labels[i].text = String.format(Locale.US, "%d %5.1f%%", i, p * 100)
        }
        Log.i(TAG, "Predicción ${r.digit} (${r.confidence}) en ${r.inferenceMs} ms: " +
                r.probabilities.joinToString { String.format(Locale.US, "%.3f", it) })
    }

    private fun loadSample() {
        val name = samples[sampleIndex++ % samples.size]
        val bmp = assets.open(name).use { BitmapFactory.decodeStream(it) }
        drawView.setImage(bmp)
        predict()
        txtInfo.append("\nEntrada: $name (imagen real de MNIST test)")
    }

    private fun reset() {
        drawView.clear()
        txtResult.text = "—"
        txtInfo.text = ""
        imgTensor.setImageDrawable(null)
        bars.forEach { it.progress = 0 }
        labels.forEachIndexed { i, t -> t.text = "$i" }
    }

    /** Muestra los 28x28 píxeles que realmente entran al modelo, sin suavizado. */
    private fun showTensor(bmp28: Bitmap) {
        val big = Bitmap.createScaledBitmap(bmp28, 224, 224, false)
        imgTensor.setImageDrawable(BitmapDrawable(resources, big).apply { isFilterBitmap = false })
    }

    private fun buildProbabilityBars(container: LinearLayout) {
        val labelWidth = (72 * resources.displayMetrics.density).toInt()
        for (i in 0 until DigitClassifier.NUM_CLASSES) {
            val row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
            val label = TextView(this).apply {
                text = "$i"
                typeface = Typeface.MONOSPACE
                textSize = 11f
                width = labelWidth
            }
            val bar = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply {
                max = 1000
                layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
                    .apply { gravity = Gravity.CENTER_VERTICAL }
            }
            row.addView(label)
            row.addView(bar)
            container.addView(row)
            labels += label
            bars += bar
        }
    }

    override fun onDestroy() {
        classifier.close()
        super.onDestroy()
    }

    companion object {
        private const val TAG = "caso1"
    }
}
