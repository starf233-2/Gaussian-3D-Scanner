plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

val withoutHtp = (rootProject.findProperty("withoutHtp") ?: "false").toString().toBoolean()

android {
    namespace = "com.starfall.gaussianscanner"
    compileSdk = 34

    signingConfigs {
        create("release") {
            storeFile = file("../release.jks")
            storePassword = "android"
            keyAlias = "gaussianscanner"
            keyPassword = "android"
        }
    }

    defaultConfig {
        applicationId = "com.starfall.gaussianscanner"
        minSdk = 24
        targetSdk = 34
        versionCode = 14
        versionName = "10.4"

        // QNN (Qualcomm AI Engine Direct) only ships arm64-v8a binaries.
        // Restrict packaging to arm64 to keep the APK small.
        ndk {
            abiFilters += listOf("arm64-v8a")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("release")
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        viewBinding = true
    }

    @Suppress("DEPRECATION")
    aaptOptions {
        noCompress("tflite")
    }

    // QNN HTP (888+ NPU) runtime lives in a separate source set so it can be
    // dropped when building with -PwithoutHtp=true (saves ~69 MB APK size).
    sourceSets {
        getByName("main") {
            if (!withoutHtp) {
                jniLibs.srcDir("jniLibsHtp")
                assets.srcDir("assetsHtp")
            }
        }
    }

    packaging {
        jniLibs {
            // keep all prebuilt QNN libs; do not strip
            keepDebugSymbols += arrayOf("**/*.so")
            // Official QNN docs REQUIRE legacy packaging: native libs must be
            // extracted to real files on disk (nativeLibraryDir) so the QNN
            // delegate can dlopen them and the fastRPC stub can locate them.
            useLegacyPackaging = true
        }
        resources {
            excludes += setOf("META-INF/*", "META-INF/**")
        }
    }
}

dependencies {
    // guice is a spurious transitive dependency of the TFLite POMs and is
    // unused by the runtime classes; excluding it keeps minSdk at 24.
    configurations.all {
        exclude(group = "com.google.inject", module = "guice")
    }

    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("com.google.android.material:material:1.11.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("androidx.exifinterface:exifinterface:1.3.7")

    val cameraxVersion = "1.3.1"
    implementation("androidx.camera:camera-core:$cameraxVersion")
    implementation("androidx.camera:camera-camera2:$cameraxVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraxVersion")
    implementation("androidx.camera:camera-view:$cameraxVersion")

    // TFLite runtime — pinned to 2.15.0, the version the QNN TFLite delegate
    // in QNN SDK 2.32.0.250228 was developed and tested against.
    implementation("org.tensorflow:tensorflow-lite:2.15.0")
    implementation("org.tensorflow:tensorflow-lite-gpu:2.15.0")
    // GpuDelegateFactory lives in the gpu-api artifact in 2.15.0; without it the
    // GpuDelegate() constructor crashes at runtime with NoClassDefFoundError.
    implementation("org.tensorflow:tensorflow-lite-gpu-api:2.15.0")

    // QNN TFLite delegate Java binding (unpacked from qtld-release.aar within
    // QNN SDK). Backend runtime libs live in src/main/jniLibs/arm64-v8a.
    implementation(files("libs/qtld.jar"))

    testImplementation("junit:junit:4.13.2")
}
