import 'package:meta/meta.dart';

@immutable
class BubbleScore {
  const BubbleScore({
    required this.pBlank,
    required this.pFilled,
    required this.isFilled,
  });

  final double pBlank;
  final double pFilled;
  final bool isFilled;
}

@immutable
class RowDecision {
  const RowDecision({
    required this.selected,
    required this.status,
    required this.isBlank,
    required this.isTie,
    required this.isMultiMark,
    required this.needsReview,
    required this.confidence,
    required this.marginTop2,
    required this.scores,
  });

  final String? selected;
  final String status;
  final bool isBlank;
  final bool isTie;
  final bool isMultiMark;
  final bool needsReview;
  final double confidence;
  final double marginTop2;
  final Map<String, double> scores;
}

@immutable
class OmrModelManifest {
  const OmrModelManifest({
    required this.modelId,
    required this.inputName,
    required this.outputName,
    required this.inputShape,
    required this.outputShape,
    required this.operatingThreshold,
    required this.minFillProb,
    required this.tieThreshold,
    required this.multiMarkThreshold,
    required this.reviewBand,
  });

  final String modelId;
  final String inputName;
  final String outputName;
  final List<int> inputShape;
  final List<int> outputShape;
  final double operatingThreshold;
  final double minFillProb;
  final double tieThreshold;
  final double multiMarkThreshold;
  final double reviewBand;

  factory OmrModelManifest.fromJson(Map<String, dynamic> json) {
    final input = json['input'] as Map<String, dynamic>;
    final output = json['output'] as Map<String, dynamic>;
    final thresholds = json['thresholds'] as Map<String, dynamic>;

    return OmrModelManifest(
      modelId: json['model_id'] as String,
      inputName: input['name'] as String,
      outputName: output['name'] as String,
      inputShape: (input['shape'] as List).map((e) => (e as num).toInt()).toList(growable: false),
      outputShape: (output['shape'] as List).map((e) => (e as num).toInt()).toList(growable: false),
      operatingThreshold: (thresholds['operating_threshold'] as num).toDouble(),
      minFillProb: (thresholds['min_fill_prob'] as num).toDouble(),
      tieThreshold: (thresholds['tie_threshold'] as num).toDouble(),
      multiMarkThreshold: (thresholds['multi_mark_threshold'] as num).toDouble(),
      reviewBand: (thresholds['review_band'] as num).toDouble(),
    );
  }

  void validateContract() {
    if (inputShape.length != 4 || inputShape[0] != 1 || inputShape[1] != 3 || inputShape[2] != 64 || inputShape[3] != 64) {
      throw StateError('Invalid input shape: $inputShape. Expected [1, 3, 64, 64].');
    }
    if (outputShape.length != 2 || outputShape[0] != 1 || outputShape[1] != 2) {
      throw StateError('Invalid output shape: $outputShape. Expected [1, 2].');
    }
  }
}
