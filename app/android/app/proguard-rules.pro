# ONNX Runtime は JNI から Java クラスを参照するため、R8 で削る/名前替えすると
# 実行時に GetMethodID(java_class == null) で SIGABRT する。全 keep が必須。
-keep class ai.onnxruntime.** { *; }
-dontwarn ai.onnxruntime.**
