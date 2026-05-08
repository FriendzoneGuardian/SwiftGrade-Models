import 'dart:typed_data';

import 'package:image/image.dart' as img;

class OmrPreprocessor {
  const OmrPreprocessor({this.targetSize = 64});

  final int targetSize;

  /// Returns tensor in NCHW layout with shape [1, 3, 64, 64].
  Float32List preprocessToNchw(Uint8List encodedImageBytes) {
    final decoded = img.decodeImage(encodedImageBytes);
    if (decoded == null) {
      throw StateError('Could not decode bubble image bytes.');
    }

    final resized = img.copyResize(
      decoded,
      width: targetSize,
      height: targetSize,
      interpolation: img.Interpolation.average,
    );

    final hw = targetSize * targetSize;
    final tensor = Float32List(1 * 3 * hw);

    for (var y = 0; y < targetSize; y++) {
      for (var x = 0; x < targetSize; x++) {
        final p = resized.getPixel(x, y);

        // Model contract: RGB in [0, 1], CHW layout.
        final r = p.r.toDouble() / 255.0;
        final g = p.g.toDouble() / 255.0;
        final b = p.b.toDouble() / 255.0;

        final idx = y * targetSize + x;
        tensor[idx] = r;
        tensor[hw + idx] = g;
        tensor[(2 * hw) + idx] = b;
      }
    }

    return tensor;
  }
}
