# Performance and Bottleneck Analysis

This section documents timing and memory usage across three test sets (A, B, C).  
Optimizations include **object reuse**, **pre-allocated buffers**, and **const references** to reduce copying and memory allocation overhead.

---

## Timing Details (ms)

| Stage           | Set A     | Set B     | Set C     |
|-----------------|-----------|-----------|-----------|
| **Image Loading** | 190.246   | 186.635   | 219.014   |
| **ORB Detection** | 461.141   | 710.966   | 1168.830  |
| **ORB Matching**  | 0.806     | 1.302     | 1.695     |
| **RANSAC**        | 7.161     | 4.083     | 5.311     |
| **Warping**       | 153.042   | 114.500   | 120.299   |

![Timing Bar Chart](timing.png)

**Observations:**
- ORB detection is the primary computational bottleneck.
- Warping is significant but consistent.
- Matching and RANSAC times are negligible.

---

## Memory Usage (KB)

| Stage           | Set A     | Set B     | Set C     |
|-----------------|-----------|-----------|-----------|
| **Image Loading** | 36,376    | 34,848    | 36,088    |
| **ORB Detection** | 4,432     | 4,176     | 4,280     |
| **ORB Matching**  | 0         | 0         | 0         |
| **RANSAC**        | 0         | 0         | 0         |
| **Warping**       | 176,540   | 174,800   | 176,516   |

![Memory Bar Chart](memory.png)

**Observations:**
- Warping stage dominates memory usage due to large intermediate images.
- Image loading memory is stable across test sets.
- ORB operations consume minimal memory.

---

## Key Optimizations

- Pre-allocated panorama buffers to avoid repeated allocations.
- Reuse of two image objects throughout all stitching operations.
- Use of **pointers** and **const references** to prevent unnecessary data copying.
- RANSAC early termination based on threshold to speed up outlier rejection.
- ROI-based ORB detection reduces search space and speeds up feature extraction.

---
