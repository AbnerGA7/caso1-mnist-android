package com.grupo.caso1

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.Rect
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View

/** Lienzo cuadrado: fondo negro y trazo blanco, igual que las imágenes de MNIST. */
class DrawView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    private var bitmap: Bitmap? = null
    private var canvas: Canvas? = null
    private val path = Path()
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
    }

    var onStrokeFinished: (() -> Unit)? = null

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        val size = MeasureSpec.getSize(widthMeasureSpec)
        setMeasuredDimension(size, size)
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        canvas = Canvas(bitmap!!).apply { drawColor(Color.BLACK) }
        paint.strokeWidth = w / 14f     // grosor proporcional al trazo de MNIST
    }

    override fun onDraw(c: Canvas) {
        bitmap?.let { c.drawBitmap(it, 0f, 0f, null) }
        c.drawPath(path, paint)
    }

    @SuppressLint("ClickableViewAccessibility")
    override fun onTouchEvent(e: MotionEvent): Boolean {
        when (e.action) {
            MotionEvent.ACTION_DOWN -> path.moveTo(e.x, e.y)
            MotionEvent.ACTION_MOVE -> path.lineTo(e.x, e.y)
            MotionEvent.ACTION_UP -> {
                path.lineTo(e.x, e.y)
                canvas?.drawPath(path, paint)
                path.reset()
                onStrokeFinished?.invoke()
            }
        }
        invalidate()
        return true
    }

    fun clear() {
        path.reset()
        canvas?.drawColor(Color.BLACK)
        invalidate()
    }

    /** Dibuja una imagen (p. ej. una muestra real de MNIST) ocupando el lienzo. */
    fun setImage(img: Bitmap) {
        val c = canvas ?: return
        c.drawColor(Color.BLACK)
        c.drawBitmap(img, null, Rect(0, 0, width, height), Paint(Paint.FILTER_BITMAP_FLAG))
        invalidate()
    }

    fun getBitmap(): Bitmap? = bitmap
}
