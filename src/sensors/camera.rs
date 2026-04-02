use nalgebra::{Isometry3, Matrix3, Rotation3, Translation3, Vector3};
use std::path::Path;
use anyhow::{Context, Result};
use std::fs;

/// Camera intrinsic parameters for one camera (e.g. cam0 = left).
#[derive(Debug, Clone)]
pub struct CameraIntrinsics {
    pub fx: f64, pub fy: f64,  // focal length in pixels
    pub cx: f64, pub cy: f64,  // principal point
}

/// Extrinsic transform: IMU body frame → camera frame.
/// Stored as nalgebra Isometry3 (rotation + translation, no scale).
///
/// RUST: Isometry3<f64> ≈ Eigen::Isometry3d in C++.
///       It encodes a rigid-body transform: T = [R | t]
///       and enforces that R is always a proper rotation matrix.
#[derive(Debug, Clone)]
pub struct Calibration {
    pub t_imu_to_velo: Isometry3<f64>,  // IMU → Velodyne LiDAR
    pub cam0_intrinsics: CameraIntrinsics,
    pub p_rect_00: nalgebra::Matrix3x4<f64>,  // projection matrix
    /// R that rotates a vector from IMU frame into rectified cam0 frame.
    /// R_imu_to_cam = R_rect_00 * R_velo_to_cam * R_imu_to_velo
    pub r_imu_to_cam: Rotation3<f64>,
}

impl Calibration {
    /// Load calibration from the KITTI calib files for a given date directory.
    /// e.g. date_dir = "data/2011_09_26"
    pub fn load(date_dir: &Path) -> Result<Self> {
        let t_imu_to_velo = load_imu_to_velo(date_dir)?;
        let (cam0_intrinsics, p_rect_00, r_rect_00) = load_cam_to_cam(date_dir)?;
        let r_velo_to_cam = load_velo_to_cam(date_dir)?;

        // Chain: IMU → Velo → cam0 raw → cam0 rectified
        let r_imu_to_cam = r_rect_00
            * r_velo_to_cam
            * t_imu_to_velo.rotation.to_rotation_matrix();

        Ok(Self { t_imu_to_velo, cam0_intrinsics, p_rect_00, r_imu_to_cam })
    }
}

/// Parse 9 floats from a "R: r00 r01 ..." line into a Rotation3
fn parse_rotation(line: &str) -> Result<Rotation3<f64>> {
    let vals = parse_floats(line)?;
    anyhow::ensure!(vals.len() == 9, "Expected 9 values for R matrix");
    // nalgebra Matrix3 is column-major, KITTI is row-major → transpose
    let m = Matrix3::from_row_slice(&vals);
    // RUST: from_matrix_unchecked trusts we have a valid rotation.
    //       In production you'd use from_matrix which checks orthogonality.
    Ok(Rotation3::from_matrix_unchecked(m))
}

/// Parse 3 floats from a "T: tx ty tz" line into a Translation3
fn parse_translation(line: &str) -> Result<Translation3<f64>> {
    let vals = parse_floats(line)?;
    anyhow::ensure!(vals.len() == 3, "Expected 3 values for T vector");
    Ok(Translation3::new(vals[0], vals[1], vals[2]))
}

/// Parse floats from the value part of "KEY: v0 v1 v2 ..."
fn parse_floats(line: &str) -> Result<Vec<f64>> {
    // Skip the "KEY: " prefix — take everything after the first ':'
    let value_part = line.splitn(2, ':').nth(1)
        .with_context(|| format!("No ':' found in line: {line}"))?;
    value_part
        .split_whitespace()
        .map(|s| s.parse::<f64>().with_context(|| format!("Cannot parse float: {s}")))
        .collect()
}

fn load_imu_to_velo(date_dir: &Path) -> Result<Isometry3<f64>> {
    let path = date_dir.join("calib_imu_to_velo.txt");
    let content = fs::read_to_string(&path)
        .with_context(|| format!("Cannot read {}", path.display()))?;

    let mut rot = None;
    let mut trans = None;

    for line in content.lines() {
        if line.starts_with("R:") { rot   = Some(parse_rotation(line)?); }
        if line.starts_with("T:") { trans = Some(parse_translation(line)?); }
    }

    // TODO: construct and return Isometry3::from_parts(translation, rotation)
    //       Use .with_context(|| "...")? on each Option to turn None into an error.
    //       Hint: rot.with_context(|| "Missing R in imu_to_velo")?
    let r = rot.with_context(|| "Missing R in imu_to_velo")?;
    let t = trans.with_context(|| "Missing T in imu_to_velo")?;
    // Rotation3 → UnitQuaternion: nalgebra requires an explicit conversion here.
    // Isometry3::from_parts expects UnitQuaternion, not Rotation3.
    // C++: Eigen::Isometry3d has the same split — you'd do:
    //      iso.linear() = rot_matrix; iso.translation() = t_vec;
    Ok(Isometry3::from_parts(t, nalgebra::UnitQuaternion::from_rotation_matrix(&r)))
}

fn load_cam_to_cam(date_dir: &Path) -> Result<(CameraIntrinsics, nalgebra::Matrix3x4<f64>, Rotation3<f64>)> {
    let path = date_dir.join("calib_cam_to_cam.txt");
    let content = fs::read_to_string(&path)
        .with_context(|| format!("Cannot read {}", path.display()))?;

    let mut k00: Option<Vec<f64>> = None;
    let mut p00: Option<Vec<f64>> = None;
    let mut r_rect: Option<Rotation3<f64>> = None;

    for line in content.lines() {
        if line.starts_with("K_00:")      { k00    = Some(parse_floats(line)?); }
        if line.starts_with("P_rect_00:") { p00    = Some(parse_floats(line)?); }
        if line.starts_with("R_rect_00:") { r_rect = Some(parse_rotation(line)?); }
    }

    let k = k00.with_context(|| "Missing K_00")?;
    let p = p00.with_context(|| "Missing P_rect_00")?;
    let r = r_rect.with_context(|| "Missing R_rect_00")?;

    anyhow::ensure!(k.len() == 9,  "K_00 should have 9 values");
    anyhow::ensure!(p.len() == 12, "P_rect_00 should have 12 values");

    let intrinsics = CameraIntrinsics {
        fx: k[0], fy: k[4], cx: k[2], cy: k[5],
    };
    let p_rect_00 = nalgebra::Matrix3x4::from_row_slice(&p);

    Ok((intrinsics, p_rect_00, r))
}

fn load_velo_to_cam(date_dir: &Path) -> Result<Rotation3<f64>> {
    let path = date_dir.join("calib_velo_to_cam.txt");
    let content = fs::read_to_string(&path)
        .with_context(|| format!("Cannot read {}", path.display()))?;
    for line in content.lines() {
        if line.starts_with("R:") { return parse_rotation(line); }
    }
    anyhow::bail!("Missing R in calib_velo_to_cam.txt")
}