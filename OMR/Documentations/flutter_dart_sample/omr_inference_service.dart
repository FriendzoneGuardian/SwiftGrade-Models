import 'dart:typed_data';

import 'omr_postprocessing.dart';
import 'omr_preprocessing.dart';
import 'omr_types.dart';

/// Adapter boundary for whichever ONNX Runtime plugin is chosen in Flutter.
///
/// Implement this against your selected package and keep the rest of the
/// pipeline package-agnostic.
abstract class OrtRunner {
  Future<void> init({
    required String modelAssetPath,
  });

  Future<List<double>> runLogits({
    required String inputName,
    required String outputName,
    required Float32List inputTensor,
    required List<int> inputShape,
  });
}

class OmrInferenceService {
  OmrInferenceService({
    required OrtRunner runner,
    required OmrModelManifest manifest,
    OmrPreprocessor preprocessor = const OmrPreprocessor(),
    OmrPostprocessor postprocessor = const OmrPostprocessor(),
  })  : _runner = runner,
        _manifest = manifest,
        _preprocessor = preprocessor,
        _postprocessor = postprocessor;

  final OrtRunner _runner;
  final OmrModelManifest _manifest;
  final OmrPreprocessor _preprocessor;
  final OmrPostprocessor _postprocessor;

  Future<void> init({required String modelAssetPath}) async {
    _manifest.validateContract();
    await _runner.init(modelAssetPath: modelAssetPath);
  }

  Future<BubbleScore> scoreBubble(Uint8List bubbleCropBytes) async {
    final inputTensor = _preprocessor.preprocessToNchw(bubbleCropBytes);
    final logits = await _runner.runLogits(
      inputName: _manifest.inputName,
      outputName: _manifest.outputName,
      inputTensor: inputTensor,
      inputShape: _manifest.inputShape,
    );

    return _postprocessor.bubbleFromLogits(
      logits,
      operatingThreshold: _manifest.operatingThreshold,
    );
  }

  Future<RowDecision> scoreRow({
    required List<Uint8List> bubbleCrops,
    required List<String> bubbleLabels,
  }) async {
    if (bubbleCrops.length != bubbleLabels.length) {
      throw StateError('bubbleCrops and bubbleLabels lengths must match.');
    }

    final probs = <double>[];
    for (final crop in bubbleCrops) {
      final score = await scoreBubble(crop);
      probs.add(score.pFilled);
    }

    return _postprocessor.decideRow(
      bubbleProbs: probs,
      bubbleLabels: bubbleLabels,
      minFillProb: _manifest.minFillProb,
      tieThreshold: _manifest.tieThreshold,
      multiMarkThreshold: _manifest.multiMarkThreshold,
      reviewBand: _manifest.reviewBand,
    );
  }
}
