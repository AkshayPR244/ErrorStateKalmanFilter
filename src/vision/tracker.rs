use image::GrayImage;
use nalgebra::{Matrix2, Vector2};
use crate::vision::features::KeyPoint;

/// A tracked feature: the same physical point seen in two consecutive frames.
///
/// RUST: plain data struct — no methods yet. We'll add them if needed.
///       `f32` for sub-pixel positions (pixel coords don't need f64 precision).
#[derive(Debug, Clone)]
pub struct Track {
    /// Original keypoint in the previous frame (integer pixel coords).
    pub prev: KeyPoint,
    /// Tracked position in the current frame (sub-pixel, f32).
    pub curr_x: f32,
    pub curr_y: f32,
}

/// Parameters for the Lucas-Kanade tracker.
pub struct LKParams {
    /// Half-width of the tracking patch. Full patch = (2*half+1)² pixels.
    /// Default 5 → 11×11 patch.
    pub half_win: i32,
    /// Maximum number of Newton–Raphson iterations per feature.
    pub max_iters: u32,
    /// Convergence threshold: stop when |Δ(dx,dy)| < epsilon.
    pub epsilon: f32,
    /// Drop a track if the final SSD residual exceeds this. Prevents
    /// tracking into flat / occluded regions.
    pub max_residual: f32,
}

impl LKParams {
    pub fn default() -> Self {
        Self { half_win: 5, max_iters: 20, epsilon: 0.03, max_residual: 1500.0 }
    }
}

/// Sample a pixel from a GrayImage at floating-point coordinates using
/// bilinear interpolation.
///
/// RUST: the function is `inline` by nature — the compiler will inline small
///       helpers like this automatically in release mode.
///       Returns `Option<f32>`: None if (x, y) is out of bounds.
///
///       `Option` is Rust's way of expressing "this might not exist" without
///       using a sentinel value like -1 or NaN. The caller must handle None.
fn sample_bilinear(img: &GrayImage, x: f32, y: f32) -> Option<f32> {
    let (w, h) = img.dimensions();
    if x < 0.0 || y < 0.0 || x >= (w - 1) as f32 || y >= (h - 1) as f32 {
        return None;
    }
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;

    // Bilinear blend of the four surrounding pixels.
    let p00 = img.get_pixel(x0,     y0    )[0] as f32;
    let p10 = img.get_pixel(x0 + 1, y0    )[0] as f32;
    let p01 = img.get_pixel(x0,     y0 + 1)[0] as f32;
    let p11 = img.get_pixel(x0 + 1, y0 + 1)[0] as f32;

    Some(p00 * (1.0-fx) * (1.0-fy)
       + p10 * fx       * (1.0-fy)
       + p01 * (1.0-fx) * fy
       + p11 * fx       * fy)
}

/// Compute the x-gradient at integer pixel (x, y) using a central difference.
/// Returns 0.0 if the pixel is on the image border.
fn grad_x(img: &GrayImage, x: u32, y: u32) -> f32 {
    let (w, _) = img.dimensions();
    if x == 0 || x >= w - 1 { return 0.0; }
    (img.get_pixel(x + 1, y)[0] as f32 - img.get_pixel(x - 1, y)[0] as f32) * 0.5
}

/// Compute the y-gradient at integer pixel (x, y) using a central difference.
fn grad_y(img: &GrayImage, x: u32, y: u32) -> f32 {
    let (_, h) = img.dimensions();
    if y == 0 || y >= h - 1 { return 0.0; }
    (img.get_pixel(x, y + 1)[0] as f32 - img.get_pixel(x, y - 1)[0] as f32) * 0.5
}

/// Track a single feature from `prev_img` to `curr_img` using
/// iterative Lucas-Kanade.
///
/// Returns `Some((dx, dy))` — the sub-pixel displacement — or `None` if
/// the track diverged or wandered out of bounds.
///
/// RUST: `Option<(f32, f32)>` — a tuple inside an Option. Tuples are anonymous
///       structs: `(f32, f32)` is its own type. No need to define a struct
///       just to return two floats.
fn track_single(
    prev_img: &GrayImage,
    curr_img: &GrayImage,
    kp: &KeyPoint,
    params: &LKParams,
) -> Option<(f32, f32)> {
    let cx = kp.x as i32;
    let cy = kp.y as i32;
    let hw = params.half_win;
    let (w, h) = prev_img.dimensions();

    // Guard: patch must fit entirely inside the image.
    if cx - hw < 0 || cy - hw < 0
    || cx + hw >= w as i32 || cy + hw >= h as i32 {
        return None;
    }

    // ── Build the structure tensor H ───────────────────────
    // Loop over every pixel (dx, dy) in the patch:  dx in -hw..=hw, dy in -hw..=hw
    // For each pixel:
    //   - px = (cx + dx) as u32,  py = (cy + dy) as u32
    //   - ix = grad_x(prev_img, px, py)
    //   - iy = grad_y(prev_img, px, py)
    //   - accumulate:  sum_ix2  += ix * ix
    //                  sum_iy2  += iy * iy
    //                  sum_ixiy += ix * iy
    //
    // Then build the 2×2 matrix:
    //   let h_mat = Matrix2::new(sum_ix2, sum_ixiy,
    //                             sum_ixiy, sum_iy2);
    //
    // RUST: `Matrix2<f32>` is nalgebra's fixed 2×2 matrix.
    //       Ranges: `-hw..=hw` is an *inclusive* range (includes hw itself).
    //       `as u32` cast: safe here because we checked bounds above.
    
    // for each pixel in the patch, compute gradients and accumulate into the structure tensor
    let mut sum_ix2 = 0.0f32;
    let mut sum_iy2 = 0.0f32;
    let mut sum_ixiy = 0.0f32; 
    for dy in -hw..=hw {
        for dx in -hw..=hw {
            let px = (cx + dx) as u32;
            let py = (cy + dy) as u32;
            let ix = grad_x(prev_img, px, py);
            let iy = grad_y(prev_img, px, py);
            sum_ix2 += ix * ix;
            sum_iy2 += iy * iy;
            sum_ixiy += ix * iy;
        }
    }

    let h_mat = Matrix2::new(sum_ix2, sum_ixiy,
                              sum_ixiy, sum_iy2);

    // Check that H is well-conditioned (not a flat region).
    // `det` = determinant. If it's tiny the patch has no gradient → can't track.
    if h_mat.determinant().abs() < 1e-6 {
        return None;
    }

    // Invert H once — reuse across all iterations.
    // `try_inverse()` returns Option<Matrix2> — None if singular.
    //
    // RUST: the `?` operator at the end of `try_inverse()?` means:
    //   "if this is None, return None from the whole function immediately."
    //   It is syntactic sugar for:
    //       match h_mat.try_inverse() { Some(m) => m, None => return None }
    //
    //   You've seen `?` with Result (propagate errors); it works the same
    //   with Option (propagate None). Same operator, two types.
    let h_inv = h_mat.try_inverse()?;

    // Current displacement estimate — start at zero (predict no motion).
    let mut dx = 0.0f32;
    let mut dy = 0.0f32;

    for _ in 0..params.max_iters {
        // ── Build the error vector b ───────────────────────
        // Same patch loop as above. For each pixel:
        //   - ix = grad_x(prev_img, px, py)         (same as before)
        //   - iy = grad_y(prev_img, px, py)
        //   - i1 = prev_img.get_pixel(px, py)[0] as f32
        //   - i2 = sample_bilinear(curr_img,
        //               (cx + ddx) as f32 + dx,
        //               (cy + ddy) as f32 + dy)?   ← use `?` here too!
        //     where ddx,ddy are the inner loop offsets
        //   - it = i2 - i1   (temporal difference)
        //   - accumulate:  sum_bx += ix * it
        //                  sum_by += iy * it
        //
        // Then: let b = Vector2::new(-sum_bx, -sum_by);
        //
        // RUST: `Vector2<f32>` is nalgebra's 2×1 column vector.
        //       Note the `?` inside a for loop: returning None from the whole
        //       function, not just the loop iteration.
        //
        
        let mut sum_bx = 0.0f32;
        let mut sum_by = 0.0f32;
        for ddy in -hw..=hw {
            for ddx in -hw..=hw {
                let px = (cx + ddx) as u32;
                let py = (cy + ddy) as u32;
                let ix = grad_x(prev_img, px, py);
                let iy = grad_y(prev_img, px, py);
                let i1 = prev_img.get_pixel(px, py)[0] as f32;
                let i2 = sample_bilinear(curr_img,
                    (cx + ddx) as f32 + dx,
                    (cy + ddy) as f32 + dy)?;
                let it = i2 - i1;
                sum_bx += ix * it;
                sum_by += iy * it;
            }
        }
        let b = Vector2::new(-sum_bx, -sum_by);

        // ── Update step ────────────────────────────────────
        // delta = h_inv * b
        // dx += delta[0]
        // dy += delta[1]
        // if delta.norm() < params.epsilon { break }
        //
        // RUST: `h_inv * b` is matrix-vector multiply via the `Mul` trait —
        //       same operator overloading as we saw with quaternions.
        //       `delta.norm()` = Euclidean length = √(Δx² + Δy²).
        let delta = h_inv * b;
        dx += delta[0];
        dy += delta[1];
        if delta.norm() < params.epsilon {
            break;
        }
    }

    // Compute final residual (SSD) to reject bad tracks.
    let mut ssd = 0.0f32;
    let mut count = 0u32;
    for ddy in -hw..=hw {
        for ddx in -hw..=hw {
            let px = (cx + ddx) as u32;
            let py = (cy + ddy) as u32;
            let i1 = prev_img.get_pixel(px, py)[0] as f32;
            if let Some(i2) = sample_bilinear(
                curr_img,
                (cx + ddx) as f32 + dx,
                (cy + ddy) as f32 + dy,
            ) {
                let diff = i2 - i1;
                ssd += diff * diff;
                count += 1;
            }
        }
    }
    let mean_ssd = if count > 0 { ssd / count as f32 } else { f32::MAX };
    if mean_ssd > params.max_residual { return None; }

    Some((dx, dy))
}

/// Track all keypoints from `prev_img` to `curr_img`.
/// Returns only the features that tracked successfully.
///
/// RUST: `.iter()` borrows each element; `.filter_map()` is like `.map()` but
///       drops None results automatically — equivalent to map + filter in one
///       pass. C++: `std::transform` + `std::remove_if` as two passes, or
///       C++23 `std::ranges::views::filter` + `std::ranges::views::transform`.
pub fn track_features(
    prev_img: &GrayImage,
    curr_img: &GrayImage,
    keypoints: &[KeyPoint],
    params: &LKParams,
) -> Vec<Track> {
    keypoints
        .iter()
        .filter_map(|kp| {
            // RUST: closure captures `prev_img`, `curr_img`, `params` from the
            //       enclosing scope by shared reference — the borrow checker
            //       verifies they outlive all calls to this closure.
            //       C++ equivalent: `[&](const KeyPoint& kp) { ... }`.
            let (dx, dy) = track_single(prev_img, curr_img, kp, params)?;
            Some(Track {
                prev: *kp,
                curr_x: kp.x as f32 + dx,
                curr_y: kp.y as f32 + dy,
            })
        })
        .collect()
}
