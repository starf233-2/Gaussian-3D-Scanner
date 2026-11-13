-keep class org.tensorflow.lite.** { *; }
-keep class org.tensorflow.lite.support.** { *; }
-keep class org.tensorflow.lite.gpu.** { *; }
-dontwarn org.tensorflow.lite.**
-keepattributes Signature
-keepattributes *Annotation*

# Qualcomm QNN TFLite delegate Java binding
-keep class com.qualcomm.qti.** { *; }
-dontwarn com.qualcomm.qti.**
