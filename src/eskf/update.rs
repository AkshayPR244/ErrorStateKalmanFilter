//! ESKF measurement update step.
//!
//! Uses a MeasurementModel trait so GPS and camera updates share the same
//! update machinery — only H and R differ between sensor types.
//!
//! RUST: Traits are the primary abstraction mechanism.
//!       A trait defines a contract (interface) that types can implement.
//! C++: Traits ≈ pure virtual base class / C++20 concepts.

use nalgebra::{DMatrix, DVector, SMatrix, SVector, UnitQuaternion, Vector3};
use crate::eskf::state::NominalState;
use crate::eskf::propagate::{Covariance, ErrorState, N};

/// A sensor measurement model.
/// H maps error state → measurement space.
/// R is the measurement noise covariance.
///
/// RUST: trait with associated const for measurement dimension.
///       This is how Rust does generic interfaces at compile time.
/// C++: template<int M> class MeasurementModel { virtual ... };
pub trait MeasurementModel {
    /// Measurement dimension (e.g. 3 for GPS position)
    fn dim(&self) -> usize;

    /// Measurement matrix H (dim × 15) as a dynamic matrix
    /// Dynamic here because dim varies by sensor type at runtime
    fn h_matrix(&self) -> DMatrix<f64>;

    /// Measurement noise covariance R (dim × dim)
    fn r_matrix(&self) -> DMatrix<f64>;

    /// Residual: z - h(nominal_state)
    /// This is what the filter is trying to drive to zero
    fn residual(&self, state: &NominalState) -> DVector<f64>;
}

/// GPS position update — 3-DOF position in ENU frame.
/// Measurement: z = p_enu (3×1)
/// Model:       h(x) = p  (nominal position)
/// H = [I₃ | 0₃ₓ₁₂]  — only position block is observed
pub struct GpsUpdate {
    pub position_enu: Vector3<f64>,
    pub noise_std: f64,  // position noise std dev in metres
}

impl MeasurementModel for GpsUpdate {
    fn dim(&self) -> usize { 3 }

    fn h_matrix(&self) -> DMatrix<f64> {
        // H is 3×15: identity in position columns (0-2), zeros elsewhere
        // RUST: DMatrix::zeros(rows, cols) — dynamically sized matrix
        // C++: Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, 15);
        //      H.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
        let mut h = DMatrix::<f64>::zeros(3, N);
        h[(0, 0)] = 1.0;
        h[(1, 1)] = 1.0;
        h[(2, 2)] = 1.0;
        h
    }

    fn r_matrix(&self) -> DMatrix<f64> {
        // R = σ² * I₃  — isotropic GPS noise
        DMatrix::<f64>::identity(3, 3) * (self.noise_std * self.noise_std)
    }

    fn residual(&self, state: &NominalState) -> DVector<f64> {
        // z - h(x) = GPS position - nominal position
        let diff = self.position_enu - state.position;
        // Returns a DVector<f64> from diff
        DVector::from_column_slice(diff.as_slice())

    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GPS Doppler Velocity Update
// ─────────────────────────────────────────────────────────────────────────────
//
// GNSS receivers derive velocity from the Doppler shift of carrier signals.
// This is a *separate* measurement path from pseudorange (position) — it has
// much lower latency and is often more accurate than differenced positions.
//
// We observe the 2D horizontal velocity in ENU:
//   z = [vn, ve]   (north and east components, m/s)
//   h(x) = [v_y, v_x]  (ENU: x=East, y=North, z=Up)
//
// H = [ 0₂ₓ₃ | [0 1 0; 1 0 0] | 0₂ₓ₉ ]
//
// We deliberately skip the vertical (z) component: GNSS vertical velocity is
// poor (GDOP ~3×), and the IMU+barometric model is more accurate than GPS Up.
//
// Noise: R = vel_accuracy² * I₂
//   vel_accuracy is the per-frame 1-sigma from the GNSS chipset (field 24).
//   It accounts for satellite geometry (DOP) and signal quality automatically.
//   In urban areas it may be 0.5–2 m/s; on open highway it can be ~0.05 m/s.
pub struct GpsVelocityUpdate {
    pub vel_north: f64,      // m/s, Doppler north velocity
    pub vel_east:  f64,      // m/s, Doppler east  velocity
    pub noise_std: f64,      // m/s, GNSS-reported 1-sigma (vel_accuracy field)
}

impl MeasurementModel for GpsVelocityUpdate {
    fn dim(&self) -> usize { 2 }

    fn h_matrix(&self) -> DMatrix<f64> {
        // H is 2×15.  Only velocity columns (3–5) are non-zero.
        // ENU convention: v[0]=East, v[1]=North, v[2]=Up
        // We measure [vNorth, vEast] = [v[1], v[0]].
        //
        // RUST: DMatrix::zeros allocates a zero matrix on the heap.
        //       Indexing (row, col) is same as Eigen's (row, col) syntax.
        let mut h = DMatrix::<f64>::zeros(2, N);
        h[(0, 4)] = 1.0;   // row 0 = vNorth = v[1]
        h[(1, 3)] = 1.0;   // row 1 = vEast  = v[0]
        h
    }

    fn r_matrix(&self) -> DMatrix<f64> {
        // Isotropic horizontal velocity noise.
        // Using vel_accuracy² directly as the per-axis variance — this is the
        // standard practice when the GNSS chip reports a single RMS figure.
        DMatrix::<f64>::identity(2, 2) * (self.noise_std * self.noise_std)
    }

    fn residual(&self, state: &NominalState) -> DVector<f64> {
        // z - h(x): measured velocity minus nominal velocity
        let v = state.velocity;   // (East, North, Up) in ENU
        DVector::from_column_slice(&[
            self.vel_north - v[1],   // north residual
            self.vel_east  - v[0],   // east  residual
        ])
    }
}

/// Apply one measurement update to the ESKF.
/// Updates both the covariance P and injects the error into nominal state.
pub fn apply_update(
    state: &mut NominalState,
    p: &mut Covariance,
    model: &dyn MeasurementModel,  // dyn = runtime polymorphism (virtual dispatch)
) {
    let h = model.h_matrix();
    let r = model.r_matrix();

    // Convert P to dynamic for mixed-size arithmetic with H
    // (nalgebra requires matching static/dynamic types for some ops)
    let p_dyn = DMatrix::from_iterator(N, N, p.iter().cloned());

    // Innovation covariance: S = H*P*H^T + R
    let s = &h * &p_dyn * h.transpose() + &r;

    // Kalman gain: K = P * H^T * S^-1
    // Hint: s.try_inverse() returns Option<DMatrix> — use .expect("S not invertible")
    let k = &p_dyn * h.transpose() * s.try_inverse().expect("S not invertible");

    // Error state: δx = K * residual
    let residual = model.residual(state);
    let dx = &k * residual;

    // Inject error state into nominal state
    // δp, δv → add directly; δθ → apply as small rotation; δba, δbg → add
    inject_error_state(state, &dx);

    // Update covariance — Joseph form:
    //
    //   P⁺ = (I − KH) P⁻ (I − KH)ᵀ + K R Kᵀ
    //
    // Numerically equivalent to the standard form P⁺ = (I−KH)P⁻ in exact
    // arithmetic, but symmetric and positive-semidefinite by construction.
    //
    // Why it matters: the standard form computes a difference of two similar-
    // magnitude matrices.  Floating-point cancellation gradually breaks
    // symmetry (P_ij ≠ P_ji) and positive-definiteness (negative eigenvalues).
    // In a long session (thousands of IMU steps) this makes the filter diverge.
    //
    // Joseph form analysis:
    //   (I−KH) P (I−KH)ᵀ — any ABAᵀ is symmetric if B is symmetric ✓
    //   K R Kᵀ            — another ABAᵀ with R > 0, adds back positive definiteness ✓
    //   Sum of two PSD matrices is PSD ✓
    //
    // RUST: `&a * &b` borrows both operands so they stay valid for the `+`.
    //       `a.transpose()` returns a new matrix (nalgebra is immutable-by-default).
    let i_kh = DMatrix::<f64>::identity(N, N) - &k * &h;
    let p_new_dyn = &i_kh * &p_dyn * i_kh.transpose() + &k * &r * k.transpose();

    // Copy back into static Covariance type
    for i in 0..N {
        for j in 0..N {
            p[(i, j)] = p_new_dyn[(i, j)];
        }
    }
}

/// Inject a 15-dim error state δx into the nominal state and reset δx.
/// This is the "composition" step that makes ESKF different from standard EKF.
fn inject_error_state(state: &mut NominalState, dx: &DVector<f64>) {
    // δp: indices 0-2
    state.position  += Vector3::new(dx[0], dx[1], dx[2]);
    // δv: indices 3-5
    state.velocity  += Vector3::new(dx[3], dx[4], dx[5]);
    // δθ: indices 6-8 — apply as small rotation quaternion
    // q_new = q ⊗ exp(δθ/2)  where δθ is the rotation vector
    let dtheta = Vector3::new(dx[6], dx[7], dx[8]);
    let dq = UnitQuaternion::from_scaled_axis(dtheta);
    state.orientation = state.orientation * dq;
    // δba: indices 9-11
    state.accel_bias += Vector3::new(dx[9],  dx[10], dx[11]);
    // δbg: indices 12-14
    state.gyro_bias  += Vector3::new(dx[12], dx[13], dx[14]);
}

/// Visual rotation update — uses the relative rotation R between two camera
/// frames (from the Essential matrix decomposition) to correct orientation.
///
/// The residual is the rotation error between the IMU-predicted orientation
/// change and the visually observed rotation:
///   δθ = log( R_pred⁻¹ · R_visual )
///
/// H = [0₃ₓ₆ | I₃ | 0₃ₓ₆]  — only the δθ block of the error state.
/// R = noise_std² * I₃
///
/// RUST: storing a `Matrix3<f64>` directly in the struct — we own it,
///       no pointer or reference needed. The struct is cheap to copy.
pub struct VisualRotationUpdate {
    /// Relative rotation from the Essential matrix, expressed in IMU/world frame.
    pub r_visual: nalgebra::Matrix3<f64>,
    /// Predicted rotation change computed from IMU integration over this frame.
    /// Should be `q_{i+1} * q_i^{-1}` expressed as a rotation matrix.
    pub r_predicted: nalgebra::Matrix3<f64>,
    /// Previous nominal orientation R_{i-1} in world frame.
    /// Needed to correctly Jacobian: ∂z/∂δθ = R_nominal_{i-1}, not I₃.
    /// We absorb this into the residual by pre-multiplying by R_{i-1}^T.
    pub r_nominal_prev: nalgebra::Matrix3<f64>,
    /// Noise std dev in radians (tune this; try 0.01 rad ≈ 0.6°)
    pub noise_std: f64,
}

impl MeasurementModel for VisualRotationUpdate {
    fn dim(&self) -> usize { 3 }

    fn h_matrix(&self) -> DMatrix<f64> {
        // H is 3×15: identity in the orientation columns (6–8), zeros elsewhere.
        // This says: "this measurement only tells us about δθ".
        let mut h = DMatrix::<f64>::zeros(3, N);
        h[(0, 6)] = 1.0;
        h[(1, 7)] = 1.0;
        h[(2, 8)] = 1.0;
        h
    }

    fn r_matrix(&self) -> DMatrix<f64> {
        DMatrix::<f64>::identity(3, 3) * (self.noise_std * self.noise_std)
    }

    fn residual(&self, _state: &NominalState) -> DVector<f64> {
        // Error rotation: how far does the visual R disagree with the IMU prediction?
        // R_err = R_pred⁻¹ · R_visual
        // Then convert to axis-angle (rotation vector) for the residual.
        //
        // RUST: `nalgebra::Rotation3::from_matrix_unchecked` wraps a Matrix3
        //       as a Rotation3 so we can call `.axis_angle()` on it.
        //       `from_matrix` would renormalize; `_unchecked` trusts the input.
        let r_err = self.r_predicted.transpose() * self.r_visual;
        let rot = nalgebra::Rotation3::from_matrix_unchecked(r_err);
        // axis_angle() returns Option<(Unit<Vector3>, f64)> — None if angle ≈ 0.
        // If no rotation, residual is zero.
        let dtheta = match rot.axis_angle() {
            Some((axis, angle)) => axis.into_inner() * angle,
            None => Vector3::zeros(),
        };
        // Correct Jacobian: ∂z/∂δθ = R_nominal_{i-1}, not I₃.
        // Equivalently: pre-rotate the residual by R_nominal_prev^T so H = I₃ is exact.
        let dtheta_corrected = self.r_nominal_prev.transpose() * dtheta;
        DVector::from_column_slice(dtheta_corrected.as_slice())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Non-Holonomic Constraint (NHC)
// ─────────────────────────────────────────────────────────────────────────────
//
// A car cannot slide sideways or jump vertically.  In the vehicle body frame:
//
//   v_lateral  = 0   (no side-slip)
//   v_vertical = 0   (stays on the ground)
//
// This is a "pseudo-measurement" — there is no physical sensor, we just assert
// the constraint and let the filter correct velocity and orientation to satisfy it.
//
// The H matrix is derived by projecting world-frame velocity into body frame:
//
//   v_body = R_bw^T · v_world
//            ↑ rotation from world to body = bodyaxis_in_world columns transposed
//
// We observe rows 1 and 2 of that product (lateral=y, vertical=z).
// H_{NHC} = [ 0₂ₓ₃ | (body lateral and vertical axes in world)^T | 0₂ₓ₉ ]
//
// RUST: Using nalgebra::Matrix3 column accessors to extract individual body axes.
//       Matrix column indexing: m.column(k) → VectorView3
pub struct NhcUpdate {
    /// Body y-axis vector expressed in world frame (= column 1 of R_body_to_world).
    pub body_y_in_world: Vector3<f64>,
    /// Body z-axis vector expressed in world frame (= column 2 of R_body_to_world).
    pub body_z_in_world: Vector3<f64>,
    /// Current nominal velocity expressed in body frame (= R_bw^T · v_world).
    /// Required to compute the δθ column of H — omitting it at high speed causes
    /// the filter to incorrectly attribute orientation drift as velocity error.
    pub v_nominal_body: Vector3<f64>,
    /// Noise std dev in m/s.  Typical value: 0.1 m/s.
    pub noise_std: f64,
}

impl MeasurementModel for NhcUpdate {
    fn dim(&self) -> usize { 2 }

    fn h_matrix(&self) -> DMatrix<f64> {
        // H is 2×15.  Two non-zero blocks:
        //
        // 1) Velocity columns (3–5):
        //    ∂(body_y^T · v_world)/∂δv = body_y^T   (expressed in world frame)
        //    ∂(body_z^T · v_world)/∂δv = body_z^T
        //
        // 2) Orientation columns (6–8) — the cross-term that is critical at speed:
        //    With body-frame perturbation δθ, R_perturbed = R_nominal · exp([δθ]×)
        //    ∂(body_y^T · v_world)/∂δθ = (e₂ × v_body)^T
        //    where e₂ = [0,1,0] (body-y unit vec) and v_body = R^T · v_world
        //    This equals [0,0,−vf] for a car with forward speed vf — O(10) at highway speed.
        //    Without it the filter misattributes heading drift as lateral velocity error.
        //
        // RUST: `Vector3::cross(&self, rhs)` — returns the cross product a×b.
        //       `cross` is defined on any `Vector<f64, U3, _>`.
        let mut h = DMatrix::<f64>::zeros(2, N);

        // δv columns
        h[(0, 3)] = self.body_y_in_world.x;
        h[(0, 4)] = self.body_y_in_world.y;
        h[(0, 5)] = self.body_y_in_world.z;
        h[(1, 3)] = self.body_z_in_world.x;
        h[(1, 4)] = self.body_z_in_world.y;
        h[(1, 5)] = self.body_z_in_world.z;

        // δθ columns — cross-terms
        let e_lat  = Vector3::new(0.0, 1.0, 0.0);  // body y unit vector
        let e_vert = Vector3::new(0.0, 0.0, 1.0);  // body z unit vector
        let d_lat  = e_lat.cross(&self.v_nominal_body);
        let d_vert = e_vert.cross(&self.v_nominal_body);
        h[(0, 6)] = d_lat.x;
        h[(0, 7)] = d_lat.y;
        h[(0, 8)] = d_lat.z;
        h[(1, 6)] = d_vert.x;
        h[(1, 7)] = d_vert.y;
        h[(1, 8)] = d_vert.z;

        h
    }

    fn r_matrix(&self) -> DMatrix<f64> {
        DMatrix::<f64>::identity(2, 2) * (self.noise_std * self.noise_std)
    }

    fn residual(&self, state: &NominalState) -> DVector<f64> {
        let v = state.velocity;
        let z_lat  = -self.body_y_in_world.dot(&v);
        let z_vert = -self.body_z_in_world.dot(&v);
        DVector::from_column_slice(&[z_lat, z_vert])
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Zero Velocity Update (ZUPT)
// ─────────────────────────────────────────────────────────────────────────────
//
// When the vehicle is stationary (detected from IMU noise floor or forward speed),
// inject a zero-velocity pseudo-measurement.
//
// This kills all three components of velocity drift at once — it is the single
// most effective correction for accumulated bias error at traffic stops.
//
// h(x) = v_nominal
// z    = 0  (we assert the car is not moving)
// residual = 0 − v_nominal
//
// H = [ 0₃ₓ₃ | I₃ | 0₃ₓ₉ ]  — observes the full velocity block.
//
// RUST: `pub struct` with a single field — a "newtype-like" struct.
//       In Rust you don't need a class hierarchy for this; the trait impl
//       (MeasurementModel) provides all the needed polymorphism.
pub struct ZuptUpdate {
    /// Noise std dev in m/s.  Typical value: 0.05 m/s when confident the car is stopped.
    /// A tight value (0.01) will aggressively zero velocity; 0.1 is more forgiving.
    pub noise_std: f64,
}

impl MeasurementModel for ZuptUpdate {
    fn dim(&self) -> usize { 3 }

    fn h_matrix(&self) -> DMatrix<f64> {
        // H is 3×15: identity block in velocity columns (3–5).
        let mut h = DMatrix::<f64>::zeros(3, N);
        h[(0, 3)] = 1.0;
        h[(1, 4)] = 1.0;
        h[(2, 5)] = 1.0;
        h
    }

    fn r_matrix(&self) -> DMatrix<f64> {
        DMatrix::<f64>::identity(3, 3) * (self.noise_std * self.noise_std)
    }

    fn residual(&self, state: &NominalState) -> DVector<f64> {
        // z − h(x) = 0 − v_nominal
        DVector::from_column_slice((-state.velocity).as_slice())
    }
}
