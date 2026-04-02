use nalgebra::{DMatrix, DVector, Matrix3, Vector3};
use rand::seq::SliceRandom;
use crate::sensors::camera::CameraIntrinsics;
use crate::vision::tracker::Track;

/// A matched pair of points in *normalized* image coordinates.
/// "Normalized" means undistorted and divided by focal length —
/// i.e. what the camera would see if f=1 and principal point = (0,0).
#[derive(Debug, Clone, Copy)]
pub struct Correspondence {
    pub p1: [f64; 2],  // (x, y) in normalized coords, frame t
    pub p2: [f64; 2],  // (x, y) in normalized coords, frame t+1
}

/// Errors that can occur during Essential matrix estimation.
///
/// RUST: `enum` as an *error type*. Instead of returning -1 or throwing an
///       exception, we return `Result<T, VoError>` — the caller knows exactly
///       which failure occurred and handles each case explicitly.
///
///       `#[derive(Debug)]` lets you print the error with `{:?}`.
///       C++ equivalent: a hierarchy of exceptions, or `std::error_code`.
#[derive(Debug)]
pub enum VoError {
    /// Fewer than 8 correspondences — can't run the 8-point algorithm.
    NotEnoughPoints,
    /// SVD failed to converge or produced a singular matrix.
    NumericalFailure,
    /// RANSAC found no inlier consensus.
    NoConsensus,
}

/// Convert pixel tracks to normalized image correspondences using K.
///
/// RUST: `impl CameraIntrinsics` — we call methods on the struct defined
///       in sensors/camera.rs. Since we only need fx/fy/cx/cy (all pub), we
///       just read the fields directly — no getter methods needed.
pub fn tracks_to_correspondences(
    tracks: &[Track],
    k: &CameraIntrinsics,
) -> Vec<Correspondence> {
    // RUST: `.iter().map().collect()` — transform every element of a slice
    //       into a new type and collect into a Vec. No explicit loop needed.
    tracks.iter().map(|t| {
        let x1 = (t.prev.x as f64 - k.cx) / k.fx;
        let y1 = (t.prev.y as f64 - k.cy) / k.fy;
        let x2 = (t.curr_x as f64 - k.cx as f64) / k.fx;
        let y2 = (t.curr_y as f64 - k.cy as f64) / k.fy;
        Correspondence { p1: [x1, y1], p2: [x2, y2] }
    }).collect()
}

/// Estimate the Essential matrix from exactly 8 correspondences
/// using the normalized 8-point algorithm.
///
/// Returns `Result<Matrix3<f64>, VoError>` — either the matrix or a
/// named error variant.
///
/// RUST: `Result<T, E>` is an enum with two variants:
///   `Ok(value)`   — success
///   `Err(reason)` — failure, with a reason of type E
///
///   The caller uses `match`, `?`, or `unwrap_or_else` to handle both cases.
///   Compare to C++ where you'd return `std::optional<Matrix3d>` and lose
///   the reason, or throw an exception and lose the type safety.
pub fn eight_point(corrs: &[Correspondence]) -> Result<Matrix3<f64>, VoError> {
    if corrs.len() < 8 {
        return Err(VoError::NotEnoughPoints);
    }

    // Build the 8×9 constraint matrix A.
    // Each row comes from one correspondence: the Kronecker product p2 ⊗ p1.
    //
    // RUST: `DMatrix::zeros(rows, cols)` — a heap-allocated matrix with
    //       *runtime* dimensions. (SMatrix has compile-time dimensions.)
    //       We use DMatrix here because the number of correspondences isn't
    //       known at compile time.
    
    let mut a = DMatrix::<f64>::zeros(8, 9);

    for (i, c) in corrs.iter().take(8).enumerate() {

        // Fill row i of matrix A with the 9 values of the Kronecker product
        // p2 ⊗ p1, where:
        //   p1 = [c.p1[0], c.p1[1], 1.0]   (homogeneous normalized coord)
        //   p2 = [c.p2[0], c.p2[1], 1.0]
        //
        // The 9 entries are (in order):
        //   x2*x1,  x2*y1,  x2,
        //   y2*x1,  y2*y1,  y2,
        //   x1,     y1,     1.0
        //
        // Set them with: a[(row, col)] = value
        //   e.g. a[(i, 0)] = x2 * x1
        //
        // Hint: extract the scalars first:
        //   let (x1, y1) = (c.p1[0], c.p1[1]);
        //   let (x2, y2) = (c.p2[0], c.p2[1]);
        //

        let (x1, y1) = (c.p1[0], c.p1[1]);
        let (x2, y2) = (c.p2[0], c.p2[1]);
        a[(i, 0)] = x2 * x1;  a[(i, 1)] = x2 * y1;  a[(i, 2)] = x2;
        a[(i, 3)] = y2 * x1;  a[(i, 4)] = y2 * y1;  a[(i, 5)] = y2;
        a[(i, 6)] = x1;        a[(i, 7)] = y1;        a[(i, 8)] = 1.0;
    }

    // Solve Af = 0 via SVD — the solution is the last column of V (= last row of V^T).
    //
    // RUST: nalgebra's SVD lives in `nalgebra::linalg::SVD`.
    //       `SVD::new(matrix, compute_u, compute_v_t)` — we need V^T so pass true.
    //       `.v_t` is `Option<DMatrix>` — use `ok_or` to convert None → Err.
    //
    //       `ok_or(e)` on an Option: converts Some(x) → Ok(x), None → Err(e).
    //       Idiomatic when you want to turn a missing value into a specific error.
    let svd = nalgebra::linalg::SVD::new(a, false, true);
    let v_t = svd.v_t.ok_or(VoError::NumericalFailure)?;

    // Last row of V^T = last column of V = null-space vector = our solution f.
    // For an 8×9 matrix, nalgebra returns a thin V^T of shape 8×9,
    // so the last row is index v_t.nrows()-1, NOT a hardcoded 8.
    let f_vec = v_t.row(v_t.nrows() - 1);

    // Reshape the 9-vector into a 3×3 matrix (row-major).
    let e_raw = Matrix3::new(
        f_vec[0], f_vec[1], f_vec[2],
        f_vec[3], f_vec[4], f_vec[5],
        f_vec[6], f_vec[7], f_vec[8],
    );

    // Enforce the rank-2 constraint: compute SVD of E, zero the smallest
    // singular value, reconstruct. This ensures E is a valid Essential matrix.
    let e_svd = nalgebra::linalg::SVD::new(e_raw, true, true);
    let u   = e_svd.u.ok_or(VoError::NumericalFailure)?;
    let v_t = e_svd.v_t.ok_or(VoError::NumericalFailure)?;
    let mut s = e_svd.singular_values;
    let avg = (s[0] + s[1]) * 0.5;  // average the two non-zero singular values
    s[0] = avg; s[1] = avg; s[2] = 0.0;

    // E = U * diag(s) * V^T
    let s_mat = Matrix3::new(s[0], 0.0,  0.0,
                              0.0, s[1], 0.0,
                              0.0,  0.0, s[2]);
    Ok(u * s_mat * v_t)
}

/// Algebraic epipolar error for a single correspondence given E.
/// |x2^T E x1| should be ~0 for a true inlier.
fn epipolar_error(c: &Correspondence, e: &Matrix3<f64>) -> f64 {
    let x1 = Vector3::new(c.p1[0], c.p1[1], 1.0);
    let x2 = Vector3::new(c.p2[0], c.p2[1], 1.0);
    (x2.transpose() * e * x1)[0].abs()
}

/// RANSAC wrapper: robustly estimate E despite outlier tracks.
///
/// Returns the best Essential matrix and the indices of its inliers.
pub fn ransac_essential(
    corrs: &[Correspondence],
    threshold: f64,
    iterations: usize,
    rng: &mut impl rand::Rng,
) -> Result<(Matrix3<f64>, Vec<usize>), VoError> {
    if corrs.len() < 8 {
        return Err(VoError::NotEnoughPoints);
    }

    let mut best_e: Option<Matrix3<f64>> = None;
    let mut best_inliers: Vec<usize> = Vec::new();

    // RUST: `0..iterations` is a range — same as `for (int i=0; i<iterations; i++)`.
    //       We don't need `i` so we use `_` to silence the unused-variable warning.
    for _ in 0..iterations {
        // Sample 8 random indices without replacement.
        // RUST: `SliceRandom` is a trait (from the `rand` crate) that adds
        //       `.choose_multiple()` to slices. Traits are how Rust adds methods
        //       to existing types — same concept as extension methods in C#.
        let indices: Vec<usize> = (0..corrs.len())
            .collect::<Vec<_>>()
            .choose_multiple(rng, 8)
            .copied()
            .collect();

        let sample: Vec<Correspondence> = indices.iter().map(|&i| corrs[i]).collect();

        // Fit E to the 8-point sample. If it fails (degenerate config), skip.
        // RUST: `let Ok(e) = ... else { continue }` — new in Rust 1.65.
        //       If eight_point returns Err, `continue` to the next iteration.
        //       Equivalent to: `let e = match eight_point(...) { Ok(e) => e, Err(_) => continue };`
        let Ok(e) = eight_point(&sample) else { continue };

        // Count inliers: all correspondences (not just the 8 sampled) where
        // epipolar_error(c, &e) < threshold.
        //
        // Build `inliers: Vec<usize>` — a vector of indices into `corrs`.
        // Hint:
        //   let inliers: Vec<usize> = corrs.iter()
        //       .enumerate()
        //       .filter(|(_, c)| epipolar_error(c, &e) < threshold)
        //       .map(|(i, _)| i)
        //       .collect();
        //
        // Then: if inliers.len() > best_inliers.len() { update best_e and best_inliers }
        //
        // TODO: fill in inlier counting and best update
        let inliers: Vec<usize> = corrs.iter()
            .enumerate()
            .filter(|(_, c)| epipolar_error(c, &e) < threshold)
            .map(|(i, _)| i)
            .collect();
        if inliers.len() > best_inliers.len() {
            best_e = Some(e);
            best_inliers = inliers;
        }
    }

    match best_e {
        Some(e) => Ok((e, best_inliers)),
        // RUST: `match` is exhaustive — must handle every variant.
        //       There is no "default fall-through" like in C++ switch.
        None => Err(VoError::NoConsensus),
    }
}

/// Decompose an Essential matrix into the 4 candidate (R, t) solutions.
/// Returns `[(R1, t), (R1, -t), (R2, t), (R2, -t)]`.
pub fn decompose_essential(e: &Matrix3<f64>) -> [(Matrix3<f64>, Vector3<f64>); 4] {
    let svd = nalgebra::linalg::SVD::new(*e, true, true);
    let u   = svd.u  .unwrap();
    let v_t = svd.v_t.unwrap();

    // The W matrix used in the standard E decomposition.
    let w = Matrix3::new(
         0.0, -1.0,  0.0,
         1.0,  0.0,  0.0,
         0.0,  0.0,  1.0,
    );

    let r1 = u * w     * v_t;
    let r2 = u * w.transpose() * v_t;

    // Ensure proper rotation (det = +1, not -1).
    let r1 = if r1.determinant() < 0.0 { -r1 } else { r1 };
    let r2 = if r2.determinant() < 0.0 { -r2 } else { r2 };

    let t  =  u.column(2).into_owned();
    let tn = -u.column(2).into_owned();

    [(r1, t.clone()), (r1, tn.clone()), (r2, t), (r2, tn)]
}

/// Triangulate a single point from two normalized observations using
/// the Direct Linear Transform (DLT). Returns the 3D point in camera 1 frame.
fn triangulate(
    x1: &Vector3<f64>,   // homogeneous normalized coord in cam1
    x2: &Vector3<f64>,   // homogeneous normalized coord in cam2
    r:  &Matrix3<f64>,   // rotation: cam1 → cam2
    t:  &Vector3<f64>,   // translation: cam1 → cam2
) -> Option<Vector3<f64>> {
    // Build 4×4 DLT system.
    let mut a = DMatrix::<f64>::zeros(4, 4);
    a.row_mut(0).copy_from(&(x1[0] * DVector::from_row_slice(&[1.0, 0.0, 0.0, 0.0])
        - DVector::from_row_slice(&[1.0, 0.0, 0.0, 0.0])).transpose());

    // Simpler cross-product form:  [x1]× P1 X = 0 and [x2]× P2 X = 0
    // P1 = [I|0],  P2 = [R|t]
    let p2 = {
        let mut m = DMatrix::<f64>::zeros(3, 4);
        m.fixed_view_mut::<3,3>(0,0).copy_from(r);
        m.column_mut(3).copy_from(t);
        m
    };

    let mut a4 = DMatrix::<f64>::zeros(4, 4);
    // rows from cam1: [I|0]
    a4.row_mut(0).copy_from(&DVector::from_row_slice(&[0.0, -1.0,  x1[1], 0.0]).transpose());
    a4.row_mut(1).copy_from(&DVector::from_row_slice(&[1.0,  0.0, -x1[0], 0.0]).transpose());
    // rows from cam2
    let r2 = x2[1] * p2.row(2) - p2.row(1);
    let r3 = p2.row(0) - x2[0] * p2.row(2);
    a4.row_mut(2).copy_from(&r2);
    a4.row_mut(3).copy_from(&r3);

    let svd = nalgebra::linalg::SVD::new(a4, false, true);
    let v_t = svd.v_t?;
    let x_h = v_t.row(3); // last row = null vector [X, Y, Z, W]
    if x_h[3].abs() < 1e-10 { return None; }
    Some(Vector3::new(x_h[0]/x_h[3], x_h[1]/x_h[3], x_h[2]/x_h[3]))
}

/// Select the correct (R, t) from the four candidates using the cheirality
/// condition: the reconstructed 3D point must be in front of both cameras.
///
/// Returns `Some((R, t))` or `None` if no candidate is clearly best.
pub fn select_pose(
    candidates: &[(Matrix3<f64>, Vector3<f64>); 4],
    corrs: &[Correspondence],
) -> Option<(Matrix3<f64>, Vector3<f64>)> {
    let test_corrs: Vec<&Correspondence> = corrs.iter().take(20).collect();

    // For each of the 4 candidates (r, t):
    //   - Count how many of the test_corrs triangulate to a point with
    //     positive depth (z > 0) in BOTH camera frames.
    //
    // For each test correspondence c:
    //   let x1 = Vector3::new(c.p1[0], c.p1[1], 1.0);
    //   let x2 = Vector3::new(c.p2[0], c.p2[1], 1.0);
    //   if let Some(pt) = triangulate(&x1, &x2, r, t) {
    //       // depth in cam1: just pt[2]   (z component in cam1 frame)
    //       // depth in cam2: (r * pt + t)[2]
    //       if pt[2] > 0.0 && (r * pt + t)[2] > 0.0 {
    //           count += 1;
    //       }
    //   }
    //
    // Keep the candidate with the highest count.
    // Return Some((best_r.clone(), best_t.clone())) or None if count == 0.
    //
    // RUST: `candidates.iter().enumerate()` gives (index, (r, t)) pairs.
    //       Use a plain for loop here — clearer than an iterator chain
    //       when you're accumulating a "best so far" across iterations.
    //
        let mut best_count = 0;
        let mut best_pose: Option<(Matrix3<f64>, Vector3<f64>)> = None;
    
        for (i, (r, t)) in candidates.iter().enumerate() {
            let mut count = 0;
            for c in &test_corrs {
                let x1 = Vector3::new(c.p1[0], c.p1[1], 1.0);
                let x2 = Vector3::new(c.p2[0], c.p2[1], 1.0);
                if let Some(pt) = triangulate(&x1, &x2, r, t) {
                    if pt[2] > 0.0 && (r * pt + t)[2] > 0.0 {
                        count += 1;
                    }
                }
            }
            if count > best_count {
                best_count = count;
                best_pose = Some((r.clone(), t.clone()));
            }
        }

    best_pose
}
