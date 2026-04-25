# Phase 1 – Form Extraction: Mathematical Foundations

## 1. Perspective Transform & Homography

A **homography matrix H** is a 3×3 projective transformation that maps every point
`p_src = [x, y, 1]ᵀ` in the source (camera) plane to a corresponding point
`p_dst = [x', y', 1]ᵀ` in the destination (rectified) plane:

```
p_dst ~ H · p_src

    | h00  h01  h02 |   | x |   | x' w |
H = | h10  h11  h12 | · | y | = | y' w |
    | h20  h21  h22 |   | 1 |   |  w   |

x' = (h00·x + h01·y + h02) / (h20·x + h21·y + h22)
y' = (h10·x + h11·y + h12) / (h20·x + h21·y + h22)
```

H is computed from four point correspondences (the corners of the detected form)
using `cv2.getPerspectiveTransform`, which solves the resulting 8-equation linear
system (the 9th degree of freedom is a free scale factor, conventionally set so
`h22 = 1`).

## 2. Four-Point Corner Ordering

Given the four detected corners of the form, we sort them into a canonical order
**(top-left, top-right, bottom-right, bottom-left)** before passing them to
`getPerspectiveTransform`:

| Corner       | Rule                              |
|--------------|-----------------------------------|
| Top-left     | Smallest `x + y` sum             |
| Bottom-right | Largest  `x + y` sum             |
| Top-right    | Smallest `x - y` difference      |
| Bottom-left  | Largest  `x - y` difference      |

This ordering is robust to mild rotations and ensures the warp always produces a
consistently oriented, upright form regardless of how the sheet was photographed.

## 3. Output Size: max-width / max-height Formulas

The destination rectangle dimensions are computed from the *actual* distances
between the source corners so that no content is cropped or squeezed:

```
max_width  = max( dist(bottom-right, bottom-left),
                  dist(top-right,    top-left)    )

max_height = max( dist(top-right,    bottom-right),
                  dist(top-left,     bottom-left)  )
```

where `dist(p1, p2) = √((p2.x - p1.x)² + (p2.y - p1.y)²)`.

This preserves the natural aspect ratio of the form region rather than forcing a
fixed ratio that could distort bubble spacing.

## 4. Aspect Ratio Preservation

After computing `max_width` and `max_height` from the four-point formula above,
the target output is set to `(max_width, max_height)` unless a fixed `target_size`
is explicitly requested by the caller (e.g., `800 × 1000` px for downstream
model inputs). When a fixed size is used, both width and height are independently
scaled, which is acceptable because the perspective warp already corrects for
projective distortion.

## 5. YOLO Detection Confidence Threshold (0.5)

A confidence threshold of **0.5** (50 %) is chosen for the following reasons:

- **Precision vs. recall trade-off**: below 0.5 the model begins returning
  false positives (background regions misclassified as forms), which would feed
  garbage into the perspective transform.
- **Mobile photo quality**: mobile images exhibit varying contrast, motion blur,
  and perspective skew. 0.5 provides a stable operating point that rejects weak
  detections without being so strict that partially-visible or blurry forms are
  missed entirely.
- **Downstream tolerance**: Phase 2 CLAHE preprocessing can recover some
  contrast loss, so slightly uncertain detections that clear 0.5 are recoverable.

The threshold is exposed as a constructor parameter (`conf_threshold`) so it can
be tuned per deployment without code changes.

## 6. Mobile Photo Artifacts

Real-world OMR sheets photographed on mobile devices suffer from:

| Artifact              | Effect on extraction                                      |
|-----------------------|-----------------------------------------------------------|
| Perspective skew      | Form appears trapezoidal → corrected by homography warp   |
| Barrel / pincushion distortion | Straight lines curve → partially corrected by warp |
| Uneven illumination   | Shadow gradients across the sheet → addressed in Phase 2  |
| Motion blur           | Soft edges reduce YOLO keypoint precision                 |
| JPEG compression      | Block artefacts near bubble edges                        |
| Variable focal length | Form occupies different fractions of the frame           |

YOLO's learned feature representations are invariant to most of these conditions
(unlike classical Hough-line or contour approaches), making it the preferred
detector for production-quality extraction.
