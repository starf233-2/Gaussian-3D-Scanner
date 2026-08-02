GaussianScanner turns any 2D photo into an interactive 3D parallax image on Android. It uses Depth Anything 3 Small (a state-of-the-art monocular depth estimation model running on-device via LiteRT/TFLite) to predict a per-pixel depth map, then projects every pixel into a colored 3D point cloud rendered in real-time with OpenGL ES 3.0.

Key features:

📷 Capture or import — Shoot with CameraX or pick from gallery
🧠 On-device AI — DA3 Small (52MB, FP16) with GPU delegate + XNNPACK fallback
🖐️ Intuitive controls — Drag to orbit, pinch to zoom (10–500%), double-tap to jump zoom levels, slider-adjustable point size (2.1–16px)
⚙️ Quality presets — Low (6GB) / Medium (8GB) / High (12GB+ RAM)
🎨 Material 3 Dark UI — Clean camera-style interface
Tech stack: Kotlin, CameraX, TensorFlow Lite, OpenGL ES 3.0. No cloud — everything runs locally on the device.
