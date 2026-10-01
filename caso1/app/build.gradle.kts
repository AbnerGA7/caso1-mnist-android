plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.grupo.caso1"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.grupo.caso1"
        minSdk = 24
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = "11"
    }
    // El .tflite se lee con memory-mapping: no debe comprimirse dentro del APK
    androidResources {
        noCompress += "tflite"
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    // LiteRT = TensorFlow Lite (nuevo nombre). Mismo API org.tensorflow.lite.Interpreter
    implementation("com.google.ai.edge.litert:litert:1.4.0")
}
