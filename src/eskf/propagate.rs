//! ESKF covariance propagation.
//! Propagates the 15×15 error covariance P forward using:
//!   P_new = F * P * F^T + Q
//!
//! where F is the discrete state-transition matrix and Q is process noise.

use nalgebra::{Matrix3, SMatrix, SVector, Vector3};
use crate::sensors::imu::ImuMeasurement;
use crate::eskf::state::NominalState;

/// Error state dimension
pub const N: usize = 15;

/// Type aliases — these make the code readable.
/// RUST: type aliases don't create new types (unlike newtypes),
///       they're just shorthand. Same as C++ `using`.
pub type ErrorState = SVector<f64, N>;        // 15×1 column vector
pub type Covariance = SMatrix<f64, N, N>;     // 15×15 matrix

/// Build the discrete state-transition matrix F (15×15).
/// Uses first-order zero-order hold (ZOH) discretisation.
///
/// Continuous F_c has this block structure:
///   F_c = [ 0   I   0   0   0  ]   ← dp/dt = v
///         [ 0   0  -R[a×]  -R   0  ]   ← dv/dt = R*(a-ba) + g
///         [ 0   0  -[w×]    0  -I  ]   ← dθ/dt = w - bg
///         [ 0   0   0       0   0  ]   ← dba/dt = 0
///         [ 0   0   0       0   0  ]   ← dbg/dt = 0
///
/// Discrete: F ≈ I + F_c * dt  (first-order)
pub fn build_f(state: &NominalState, imu: &ImuMeasurement, dt: f64) -> Covariance {
    let r = state.orientation.to_rotation_matrix();
    let r_mat: Matrix3<f64> = *r.matrix();

    // Bias-corrected measurements
    let accel = imu.linear_acceleration - state.accel_bias;
    let gyro  = imu.angular_velocity    - state.gyro_bias;

    // Skew-symmetric matrix of a vector v: [v×]
    // Used because cross-product a×b = [a×] * b
    // C++/Eigen: Eigen::Matrix3d skew; skew << 0,-v.z(),v.y(), ...
    let skew = |v: Vector3<f64>| -> Matrix3<f64> {
        Matrix3::new(
             0.0,  -v.z,  v.y,
             v.z,   0.0, -v.x,
            -v.y,   v.x,  0.0,
        )
    };

    let skew_a = skew(accel);
    let skew_w = skew(gyro);

    // Build F as identity + F_c * dt
    // F is 15×15, partitioned into 5×5 blocks of 3×3 each
    // Index layout: 0-2=dp, 3-5=dv, 6-8=dθ, 9-11=dba, 12-14=dbg
    let mut f = Covariance::identity();

    // dp/dt = v  →  F[0:3, 3:6] = I*dt
    f.fixed_view_mut::<3, 3>(0, 3).copy_from(&(Matrix3::identity() * dt));

    // dv/dt: -R*[a×]*dt in dv/dθ block, -R*dt in dv/dba block
    f.fixed_view_mut::<3, 3>(3, 6).copy_from(&(-r_mat * skew_a * dt));
    f.fixed_view_mut::<3, 3>(3, 9).copy_from(&(-r_mat * dt));

    // dθ/dt: -[w×]*dt in dθ/dθ block, -I*dt in dθ/dbg block
    f.fixed_view_mut::<3, 3>(6, 6).copy_from(&(Matrix3::identity() - skew_w * dt));
    f.fixed_view_mut::<3, 3>(6, 12).copy_from(&(-Matrix3::identity() * dt));

    f
}

/// Build the discrete process noise matrix Q (15×15).
/// Models IMU noise: accelerometer noise na, gyro noise ng,
/// and bias random walk: ba_rw, bg_rw.
///
/// RUST: struct with named fields for clarity — avoids passing 4 loose f64s.
pub struct ImuNoiseParams {
    pub accel_noise:   f64,  // σ_a  (m/s²/√Hz)
    pub gyro_noise:    f64,  // σ_g  (rad/s/√Hz)
    pub accel_bias_rw: f64,  // σ_ba (m/s³/√Hz)
    pub gyro_bias_rw:  f64,  // σ_bg (rad/s²/√Hz)
}

impl ImuNoiseParams {
    /// Reasonable defaults for a consumer-grade IMU (e.g. KITTI OXTS)
    pub fn kitti_defaults() -> Self {
        Self {
            accel_noise:   0.08,
            gyro_noise:    0.005,
            accel_bias_rw: 0.00004,
            gyro_bias_rw:  0.000002,
        }
    }
}

/// Build discrete Q by integrating continuous noise over dt.
/// Simplified diagonal approximation:
///   Q_dv = σ_a² * dt * I₃
///   Q_dθ = σ_g² * dt * I₃
///   Q_ba = σ_ba² * dt * I₃
///   Q_bg = σ_bg² * dt * I₃
pub fn build_q(params: &ImuNoiseParams, dt: f64) -> Covariance {
    let mut q = Covariance::zeros();

    // TODO: fill in the 4 diagonal 3×3 blocks.
    // Each block is:  σ² * dt * I₃
    // Block positions (row, col):
    //   velocity   noise: (3, 3)
    //   angle      noise: (6, 6)
    //   accel bias noise: (9, 9)
    //   gyro  bias noise: (12, 12)
    // q.fixed_view_mut::<3,3>(3, 3).copy_from(&(Matrix3::identity() * val));

    q.fixed_view_mut::<3, 3>(3, 3).copy_from(&(Matrix3::identity() * params.accel_noise.powi(2) * dt));
    q.fixed_view_mut::<3, 3>(6, 6).copy_from(&(Matrix3::identity() * params.gyro_noise.powi(2) * dt));
    q.fixed_view_mut::<3, 3>(9, 9).copy_from(&(Matrix3::identity() * params.accel_bias_rw.powi(2) * dt));
    q.fixed_view_mut::<3, 3>(12, 12).copy_from(&(Matrix3::identity() * params.gyro_bias_rw.powi(2) * dt));

    q
}

/// Propagate covariance forward one step: P_new = F*P*F^T + Q
pub fn propagate_covariance(
    p: &Covariance,
    f: &Covariance,
    q: &Covariance,
) -> Covariance {
    // P_new = F * P * F.transpose() + Q
    // nalgebra uses f.transpose() and standard * for matrix multiply
    f * p * f.transpose() + q
}