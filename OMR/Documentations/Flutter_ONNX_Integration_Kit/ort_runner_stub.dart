import 'dart:typed_data';

import 'omr_inference_service.dart';

/// Replace this stub with your chosen ONNX Runtime Flutter package.
///
/// Keep the public behavior intact so the rest of the pipeline does not change.
class OrtRunnerStub implements OrtRunner {
  bool _initialized = false;

  @override
  Future<void> init({required String modelAssetPath}) async {
    // TODO: Load model bytes from Flutter assets and initialize ONNX session.
    // Example responsibilities:
    // 1) rootBundle.load(modelAssetPath)
    // 2) create ORT environment/session
    // 3) cache session instance
    _initialized = true;
  }

  @override
  Future<List<double>> runLogits({
    required String inputName,
    required String outputName,
    required Float32List inputTensor,
    required List<int> inputShape,
  }) async {
    if (!_initialized) {
      throw StateError('OrtRunnerStub.init() must be called before runLogits().');
    }

    // TODO: Replace with real ONNX session invocation.
    // Expected output: [blankLogit, filledLogit]
    throw UnimplementedError('Wire this method to the selected ONNX runtime package.');
  }
}
