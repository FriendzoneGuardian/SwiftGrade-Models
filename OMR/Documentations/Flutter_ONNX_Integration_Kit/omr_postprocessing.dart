import 'dart:math' as math;

import 'omr_types.dart';

class OmrPostprocessor {
  const OmrPostprocessor();

  List<double> softmax(List<double> logits) {
    if (logits.isEmpty) {
      throw StateError('Logits cannot be empty.');
    }

    final maxLogit = logits.reduce(math.max);
    final exps = logits.map((v) => math.exp(v - maxLogit)).toList(growable: false);
    final sumExp = exps.fold<double>(0.0, (a, b) => a + b);
    return exps.map((v) => v / sumExp).toList(growable: false);
  }

  BubbleScore bubbleFromLogits(
    List<double> logits, {
    required double operatingThreshold,
  }) {
    if (logits.length != 2) {
      throw StateError('Expected 2 logits [blank, filled], got ${logits.length}.');
    }
    final probs = softmax(logits);
    final pBlank = probs[0];
    final pFilled = probs[1];
    return BubbleScore(
      pBlank: pBlank,
      pFilled: pFilled,
      isFilled: pFilled >= operatingThreshold,
    );
  }

  RowDecision decideRow({
    required List<double> bubbleProbs,
    required List<String> bubbleLabels,
    double minFillProb = 0.30,
    double tieThreshold = 0.15,
    double multiMarkThreshold = 0.50,
    double reviewBand = 0.06,
  }) {
    if (bubbleProbs.length != bubbleLabels.length) {
      throw StateError('bubbleProbs and bubbleLabels lengths must match.');
    }
    if (bubbleProbs.isEmpty) {
      throw StateError('bubbleProbs cannot be empty.');
    }

    final indexed = List.generate(
      bubbleProbs.length,
      (i) => MapEntry(bubbleLabels[i], bubbleProbs[i]),
      growable: false,
    )..sort((a, b) => b.value.compareTo(a.value));

    final topLabel = indexed.first.key;
    final topProb = indexed.first.value;
    final secondProb = indexed.length > 1 ? indexed[1].value : 0.0;
    final marginTop2 = topProb - secondProb;

    final isBlank = topProb < minFillProb;
    final isTie = indexed.length > 1 && marginTop2 < tieThreshold;
    final isMultiMark =
        indexed.length > 1 &&
        topProb >= multiMarkThreshold &&
        secondProb >= multiMarkThreshold &&
        marginTop2 <= reviewBand;

    final meanOthers = indexed.length > 1
        ? indexed.skip(1).fold<double>(0.0, (acc, e) => acc + e.value) / (indexed.length - 1)
        : 0.0;
    final confidence = indexed.length > 1 ? (topProb - meanOthers) : topProb;

    String status;
    String? selected;
    if (isBlank) {
      status = 'blank';
      selected = null;
    } else if (isMultiMark) {
      status = 'multi_mark_review';
      selected = topLabel;
    } else if (isTie) {
      status = 'tie_review';
      selected = topLabel;
    } else {
      status = 'selected';
      selected = topLabel;
    }

    final scores = {for (final item in indexed) item.key: item.value};

    return RowDecision(
      selected: selected,
      status: status,
      isBlank: isBlank,
      isTie: isTie,
      isMultiMark: isMultiMark,
      needsReview: status == 'tie_review' || status == 'multi_mark_review',
      confidence: confidence,
      marginTop2: marginTop2,
      scores: scores,
    );
  }
}
