# Phase 2 – Preprocessing: Mathematical Foundations

## 1. CLAHE – Contrast Limited Adaptive Histogram Equalization

Standard histogram equalization computes a *global* mapping based on the entire
image histogram:

```
s = T(r) = (L - 1) · CDF(r)
```

where `r` is the input intensity, `s` the output intensity, `L` the number of
intensity levels, and `CDF` the cumulative distribution function of the image
histogram.

**Adaptive HE** (AHE) instead applies this mapping *locally*: the image is
divided into a grid of non-overlapping rectangular **tiles** (controlled by
`tileGridSize`), and a separate histogram and mapping function is computed for
each tile. Bilinear interpolation between the four nearest tile mappings
removes visible tile boundaries.

### Clip Limit

Unconstrained AHE can over-amplify noise in near-uniform regions. CLAHE adds a
**clip limit** `L_clip`: any histogram bin that exceeds `L_clip` is *clipped*,
and the excess counts are redistributed uniformly across all bins before the CDF
is computed:

```
excess = Σ max(0, H(i) - L_clip)   for all bins i
H_clipped(i) = min(H(i), L_clip) + excess / num_bins
```

A higher `clip_limit` allows stronger local contrast enhancement at the cost of
more noise amplification. The default value of **2.0** (relative to a
uniform histogram) gives a good balance for OMR sheet images.

### Tile Grid Size

`tileGridSize = (8, 8)` divides the image into an 8 × 8 grid (64 tiles).
Smaller tiles react to finer local structure; larger tiles approximate global HE.
8 × 8 is a standard choice for A4/letter-sized document images because it
captures illumination gradients (e.g., a shadow across one corner) without
over-fitting to individual bubble textures.

## 2. Multi-Channel CLAHE in LAB Color Space

CLAHE is applied only to the **luminance** channel to avoid shifting hue or
saturation:

```
BGR → LAB
          L channel → CLAHE → L'
          A channel (unchanged)
          B channel (unchanged)
LAB (L', A, B) → BGR
```

### Why LAB over HSV

| Property                   | LAB                                  | HSV                                    |
|----------------------------|--------------------------------------|----------------------------------------|
| Luminance separation       | L is perceptually uniform luminance  | V is brightness but coupled to S       |
| CLAHE application          | CLAHE on L avoids hue distortion     | CLAHE on V can shift perceived colour  |
| Perceptual uniformity      | Equal ΔE distances ≈ equal perception | Non-uniform; hue is circular           |
| Document suitability       | Excellent – pen marks have neutral hue | Adequate but inferior for near-grey tones |

Applying CLAHE to the HSV Value channel often produces colour fringing on
bubble edges because V is not fully decoupled from H and S. The LAB approach
is numerically more stable for near-achromatic (black ink on white paper) inputs.

## 3. Gaussian Blur for Noise Reduction

A small Gaussian blur (kernel `3 × 3`, `σ ≈ 1.0`) applied *before* CLAHE
smooths high-frequency JPEG block artefacts. Without it, the CLAHE histogram
is polluted by artefact intensities that inflate certain bins and waste clip
budget.

**Sigma rationale**: `σ = 1.0` attenuates frequencies above ~1/(2·σ) ≈ 0.5
cycles/pixel, which is sufficient to suppress 8 × 8 JPEG block boundaries
while preserving the coarser bubble fill gradients used in Phase 3.
Larger `σ` values blur bubble edges and reduce classification accuracy.

## 4. Mobile Photo Artifacts and Preprocessing Mitigations

| Artifact                    | Mitigation in Phase 2                                       |
|-----------------------------|-------------------------------------------------------------|
| Uneven illumination         | CLAHE local equalization normalises brightness across tiles |
| JPEG compression artefacts  | Pre-CLAHE Gaussian blur smooths block boundaries            |
| Motion blur                 | Cannot be fully corrected; CLAHE sharpens local contrast    |
| Vignetting (dark corners)   | Local tile mapping brightens dark peripheral regions        |
| Overexposure (clipping)     | CLAHE clip limit prevents washing out bright regions        |
| Colour cast (warm/cool WB)  | LAB L-channel processing ignores colour shift               |
