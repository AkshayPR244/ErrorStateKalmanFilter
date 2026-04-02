mod sensors;
mod kitti;
mod eskf;
mod vision;

use std::path::Path;
use rand::SeedableRng;
use nalgebra::Vector3;
use sensors::imu::ImuMeasurement;
use eskf::state::NominalState;
use eskf::propagate::{build_f, build_q, propagate_covariance, ImuNoiseParams, Covariance};
use eskf::update::{GpsUpdate, GpsVelocityUpdate, VisualRotationUpdate, NhcUpdate, ZuptUpdate, apply_update};

/// All tuneable parameters for one filter run.
/// Change these to run on a different KITTI sequence or tune the filter.
///
/// RUST: structs with named fields are idiomatic for config — avoids long
///       argument lists. Same as a C++ config struct or a Python dataclass.
struct RunConfig<'a> {
    /// Path to the sequence dir, e.g. "data/2011_09_26/2011_09_26_drive_0001_sync"
    sequence_dir: &'a str,
    /// Path to the calibration dir, e.g. "data/2011_09_26"
    calib_dir: &'a str,
    /// GPS measurement noise std dev in metres (tune this per-sequence)
    gps_noise_std: f64,
    /// IMU noise parameters
    imu_noise: ImuNoiseParams,
    /// Initial covariance diagonal value
    initial_cov: f64,
    /// Visual rotation update noise std dev in radians (None = disable visual update)
    visual_noise_std: Option<f64>,
    /// Non-holonomic constraint noise std dev in m/s (None = disable NHC)
    /// Constrains lateral and vertical body-frame velocity to zero every frame.
    nhc_noise_std: Option<f64>,
    /// Zero-velocity update noise std dev in m/s (None = disable ZUPT)
    /// Applied only when the vehicle is detected as stationary.
    zupt_noise_std: Option<f64>,
    /// Forward speed below which ZUPT triggers (m/s). Combined with gyro gate.
    zupt_speed_threshold: f64,
    /// Optional GPS blackout window [start_frame, end_frame).
    /// GPS updates are suppressed for frames in this range, simulating a tunnel
    /// or urban canyon.  All other sensors (NHC, ZUPT, Visual) remain active.
    gps_outage_frames: Option<(usize, usize)>,
    /// Enable GPS Doppler velocity update (None = disable).
    /// Uses vn/ve from the OXTS GNSS receiver with per-frame vel_accuracy as noise.
    /// `Some(scale)` multiplies vel_accuracy by scale — use 1.0 as a starting point,
    /// increase if the GNSS velocity sigma feels too optimistic in urban areas.
    gps_vel_noise_scale: Option<f64>,
}

impl<'a> RunConfig<'a> {
    /// Default config for KITTI 2011_09_26 drive 0001.
    /// Change sequence_dir here to run on another sequence.
    fn default() -> Self {
        Self {
            sequence_dir: "data/2011_09_26/2011_09_26_drive_0001_sync",
            calib_dir:    "data/2011_09_26",
            gps_noise_std: 0.5,
            imu_noise: ImuNoiseParams::kitti_defaults(),
            initial_cov: 0.01,
            visual_noise_std: Some(0.5),
            // NHC: 0.1 m/s lateral/vertical noise — tighter than GPS, looser than ZUPT
            nhc_noise_std: Some(0.1),
            // ZUPT: 0.05 m/s — confident the car isn't moving when stationary
            zupt_noise_std: Some(0.05),
            zupt_speed_threshold: 0.5,  // m/s  (≈1.8 km/h)
            gps_outage_frames: None,
            // GPS velocity: use GNSS-reported vel_accuracy × 1.0.
            // The OXTS chipset reports conservative σ values; scale=1.0 works well.
            gps_vel_noise_scale: Some(1.0),
        }
    }
}

/// Run the full ESKF predict+update loop over a loaded sequence.
/// Returns per-frame (filtered_position, gps_position) pairs.
///
/// Now accepts camera images for the visual rotation update.
/// `images`: pre-loaded grayscale frames in frame order.
///
/// RUST: `Option<&[image::GrayImage]>` — the caller passes `Some(&images)`
///       to enable the visual pipeline, or `None` to run GPS-only.
///       This is idiomatic feature-gating without a boolean flag —
///       the presence of the data IS the flag.
fn run_filter(
    records: &[kitti::OxtsRecord],
    gps: &[sensors::gps::GpsMeasurement],
    images: Option<&[image::GrayImage]>,
    cfg: &RunConfig,
    calib: &sensors::camera::Calibration,
) -> Vec<(Vector3<f64>, Vector3<f64>)> {
    let r0 = &records[0];
    let mut state = NominalState::from_oxts(gps[0].position_enu, r0.roll, r0.pitch, r0.yaw);
    let mut p = Covariance::identity() * cfg.initial_cov;
    let n = records.len();
    let mut results = Vec::with_capacity(n);

    // Visual pipeline state — only initialised when images are provided.
    // RUST: `Option<Vec<KeyPoint>>` — None means "no previous keypoints yet".
    let fast_params = vision::features::FastParams::default();
    let lk_params   = vision::tracker::LKParams::default();
    let mut prev_kps: Option<Vec<vision::features::KeyPoint>> = None;
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);

    // Keep track of orientation just before each predict so we can compute
    // the IMU-integrated rotation for the visual residual.
    let mut q_prev = state.orientation;

    for i in 0..n {
        let r = &records[i];
        let imu = ImuMeasurement::new(r.timestamp_ns, r.gyro, r.accel);
        let dt = if i == 0 { 0.1 } else {
            (records[i].timestamp_ns - records[i-1].timestamp_ns) as f64 * 1e-9
        };

        q_prev = state.orientation;

        // ── Predict ────────────────────────────────────────────────────────
        let f = build_f(&state, &imu, dt);
        let q = build_q(&cfg.imu_noise, dt);
        p = propagate_covariance(&p, &f, &q);
        state.propagate(&imu, dt);

        // ── GPS update ─────────────────────────────────────────────────────
        // Skip when the current frame falls inside a simulated GPS outage window
        // (e.g. tunnel, urban canyon).  All other sensors remain active so we
        // can measure how well NHC / ZUPT / Visual bound the dead-reckoning drift.
        let in_outage = cfg.gps_outage_frames
            .map(|(s, e)| i >= s && i < e)
            .unwrap_or(false);
        if !in_outage {
            let gps_update = GpsUpdate {
                position_enu: gps[i].position_enu,
                noise_std: cfg.gps_noise_std,
            };
            apply_update(&mut state, &mut p, &gps_update);

            // ── GPS Doppler Velocity update ────────────────────────────────────
            // Doppler velocity comes from a separate signal path from pseudorange —
            // it remains valid even when position accuracy degrades (multipath, few sats).
            // We suppress it during the outage window alongside GPS position.
            //
            // Adaptive noise: use the per-frame vel_accuracy from the GNSS chip × scale.
            // In open sky vel_accuracy ≈ 0.05 m/s; in urban canyons ≈ 1–2 m/s.
            // This automatically downweights velocity in poor GNSS geometry.
            if let Some(scale) = cfg.gps_vel_noise_scale {
                let noise_std = r.vel_accuracy * scale;
                let vel_update = GpsVelocityUpdate {
                    vel_north: r.vn,
                    vel_east:  r.ve,
                    noise_std,
                };
                apply_update(&mut state, &mut p, &vel_update);
            }
        }

        // ── Non-Holonomic Constraint (NHC) ─────────────────────────────────────
        // A car can't slide sideways or fly.  We extract the body y- and z-axes
        // from the current estimated orientation to build H, then assert that
        // those velocity components are zero.
        //
        // RUST: `.into_inner()` on a Rotation3 unwraps the inner Matrix3
        //       (Rotation3 is a newtype wrapper that guarantees orthogonality).
        //       `.column(k)` returns a column view — `.into()` copies it to Vector3.
        if let Some(nhc_std) = cfg.nhc_noise_std {
            let r_bw = state.orientation.to_rotation_matrix().into_inner();
            // v_body = R_bw^T · v_world  (rotate velocity into body frame)
            // Needed for the δθ cross-term in H — critical at highway speed.
            let v_body = r_bw.transpose() * state.velocity;
            let nhc = NhcUpdate {
                body_y_in_world: Vector3::new(r_bw[(0,1)], r_bw[(1,1)], r_bw[(2,1)]),
                body_z_in_world: Vector3::new(r_bw[(0,2)], r_bw[(1,2)], r_bw[(2,2)]),
                v_nominal_body:  v_body,
                noise_std: nhc_std,
            };
            apply_update(&mut state, &mut p, &nhc);
        }

        // ── Zero Velocity Update (ZUPT) ───────────────────────────────────────
        // Detect stationarity from two independent signals:
        //   1. OXTS forward speed |vf| < threshold (GPS-derived, reliable)
        //   2. IMU gyro norm < 0.05 rad/s (kinematic; this is what a real system
        //      would use if GPS velocity weren't available)
        // Both must agree to trigger, avoiding false positives during slow turns.
        //
        // RUST: `&&` short-circuits — the second condition is only evaluated
        //       if the first is true, matching C++ behaviour exactly.
        if let Some(zupt_std) = cfg.zupt_noise_std {
            let gyro_norm = imu.angular_velocity.norm();
            let is_stopped = r.vf.abs() < cfg.zupt_speed_threshold
                && gyro_norm < 0.05;
            if is_stopped {
                apply_update(&mut state, &mut p, &ZuptUpdate { noise_std: zupt_std });
            }
        }
        // RUST: `if let (Some(imgs), Some(noise_std)) = (...)` — simultaneous
        //       pattern match on two Options. Only enters the block when BOTH
        //       are Some. Equivalent to a nested if-let but in one line.
        if let (Some(imgs), Some(noise_std)) = (images, cfg.visual_noise_std) {
            // Detect corners on this frame.
            let curr_kps = vision::features::detect_fast(&imgs[i], &fast_params);

            // If we have keypoints from the previous frame, track and update.
            // RUST: `if let Some(prev) = &prev_kps` — borrow prev_kps to avoid
            //       moving it out of the Option (we need it to stay in place).
            if let Some(prev) = &prev_kps {
                let tracks = vision::tracker::track_features(
                    &imgs[i-1], &imgs[i], prev, &lk_params);

                if tracks.len() >= 8 {
                    let corrs = vision::epipolar::tracks_to_correspondences(
                        &tracks, &calib.cam0_intrinsics);

                    // RUST: labeled block used as a multi-exit gate — `break 'gate`
                    // jumps past the visual update without skipping `results.push`.
                    // Using `continue` here would exit the outer for-loop, causing
                    // fewer results than frames (index-out-of-bounds on the CSV write).
                    'gate: {
                        let Ok((e_mat, inlier_idx)) =
                            vision::epipolar::ransac_essential(&corrs, 0.005, 100, &mut rng)
                        else { break 'gate };

                        // Gate 1: skip if fewer than 85% of tracks agree with E.
                        let inlier_ratio = inlier_idx.len() as f64 / corrs.len() as f64;
                        if inlier_ratio < 0.85 { break 'gate; }

                        let inlier_corrs: Vec<_> =
                            inlier_idx.iter().map(|&j| corrs[j]).collect();
                        let candidates =
                            vision::epipolar::decompose_essential(&e_mat);

                        let Some((r_visual_cam, _t)) =
                            vision::epipolar::select_pose(&candidates, &inlier_corrs)
                        else { break 'gate };

                        // Gate 2: skip if implied rotation is >8° per frame.
                        let rot_angle = {
                            let r = nalgebra::Rotation3::from_matrix_unchecked(r_visual_cam);
                            r.axis_angle().map(|(_, a)| a).unwrap_or(0.0)
                        };
                        if rot_angle > 8_f64.to_radians() { break 'gate; }

                        // Adaptive noise: scale σ inversely with inlier quality.
                        let noise_eff = noise_std / (inlier_ratio * inlier_ratio);
                        let r_i2c = calib.r_imu_to_cam.into_inner();
                        let r_c2i = r_i2c.transpose();
                        let r_visual_imu = r_c2i
                            * nalgebra::Rotation3::from_matrix_unchecked(r_visual_cam)
                                .into_inner()
                            * r_i2c;

                        let r_pred = (state.orientation.to_rotation_matrix()
                            * q_prev.to_rotation_matrix().transpose()).into_inner();

                        let vis_update = VisualRotationUpdate {
                            r_visual:       r_visual_imu,
                            r_predicted:    r_pred,
                            r_nominal_prev: q_prev.to_rotation_matrix().into_inner(),
                            noise_std:      noise_eff,
                        };
                        apply_update(&mut state, &mut p, &vis_update);
                    } // 'gate
                }
            }
            prev_kps = Some(curr_kps);
        }

        results.push((state.position, gps[i].position_enu));
    }
    results
}

/// Compute horizontal RMSE from (filtered, ground_truth) position pairs.
fn compute_rmse(results: &[(Vector3<f64>, Vector3<f64>)]) -> f64 {
    let sum_sq: f64 = results.iter().map(|(est, gt)| {
        let e = est - gt;
        e.x * e.x + e.y * e.y   // horizontal only — GPS altitude is unreliable
    }).sum();
    (sum_sq / results.len() as f64).sqrt()
}

/// Per-frame horizontal position error helper.
fn point_err(results: &[(Vector3<f64>, Vector3<f64>)], i: usize) -> f64 {
    let (est, gt) = results[i];
    let d = est - gt;
    (d.x * d.x + d.y * d.y).sqrt()
}

/// Phase-split diagnostics for a GPS outage stress test.
struct PhaseMetrics {
    pre_rmse:    f64,   // RMSE before the outage
    during_rmse: f64,   // RMSE while GPS is suppressed
    post_rmse:   f64,   // RMSE after GPS resumes
    peak_error:  f64,   // Worst-case error during outage
    end_error:   f64,   // Error at the moment GPS is re-acquired (outage_end frame)
    drift_rate:  f64,   // peak_error / outage_duration  [m/s]
}

/// Compute per-phase metrics given an outage window `(start_frame, end_frame)`.
/// `dt`: nominal inter-frame time in seconds (0.1 s for 10 Hz KITTI).
fn compute_phase_metrics(
    results: &[(Vector3<f64>, Vector3<f64>)],
    outage: (usize, usize),
    dt: f64,
) -> PhaseMetrics {
    let n = results.len();
    let (s, e) = (outage.0, outage.1.min(n));

    let phase_rmse = |start: usize, end: usize| -> f64 {
        let len = end.saturating_sub(start);
        if len == 0 { return 0.0; }
        let sum: f64 = (start..end).map(|i| point_err(results, i).powi(2)).sum();
        (sum / len as f64).sqrt()
    };

    let peak_error = (s..e)
        .map(|i| point_err(results, i))
        .fold(0f64, f64::max);

    // Error at re-acquisition frame (first frame where GPS resumes).
    let end_error = if e < n { point_err(results, e) } else { point_err(results, n - 1) };

    let dur_s = (e - s) as f64 * dt;

    PhaseMetrics {
        pre_rmse:    phase_rmse(0, s),
        during_rmse: phase_rmse(s, e),
        post_rmse:   phase_rmse(e, n),
        peak_error,
        end_error,
        drift_rate:  if dur_s > 0.0 { peak_error / dur_s } else { 0.0 },
    }
}

/// Load images, run GPS-only and GPS+Visual filter, print comparison.
fn run_sequence(seq_dir: &str, calib_dir: &str, csv_prefix: &str) {
    let records = kitti::load_oxts(Path::new(seq_dir))
        .expect("Failed to load OXTS records");
    let gps  = kitti::oxts_to_gps_enu(&records);
    let calib = sensors::camera::Calibration::load(Path::new(calib_dir))
        .expect("Failed to load calibration");
    let n = records.len();

    let image_dir = format!("{}/image_00/data", seq_dir);
    let images: Vec<image::GrayImage> = (0..n)
        .map(|i| {
            let path = format!("{}/{:010}.png", image_dir, i);
            vision::features::load_frame(Path::new(&path))
        })
        .collect::<anyhow::Result<Vec<_>>>()
        .expect("Failed to load camera frames");

    // Outage window: 30%–55% of the sequence, simulating a ~10-second GPS blackout.
    // This range tends to include interesting dynamics (turns, intersections) and
    // is far enough from the start that the filter is fully converged.
    let outage_start = n * 30 / 100;
    let outage_end   = n * 55 / 100;
    let outage = (outage_start, outage_end);

    // ── Run 1: full GPS + all sensors (warm baseline, no outage) ─────────────
    let cfg_full = RunConfig {
        sequence_dir: seq_dir, calib_dir, visual_noise_std: Some(0.5),
        ..RunConfig::default()
    };
    let res_full = run_filter(&records, &gps, Some(&images), &cfg_full, &calib);
    let rmse_full = compute_rmse(&res_full);

    // ── Run 2: GPS outage + IMU only (pure dead-reckoning, worst case) ────────
    // All auxiliary sensors disabled — shows raw IMU integration drift.
    let cfg_imu_out = RunConfig {
        sequence_dir: seq_dir, calib_dir,
        visual_noise_std: None,
        nhc_noise_std:    None,
        zupt_noise_std:   None,
        gps_outage_frames: Some(outage),
        ..RunConfig::default()
    };
    let res_imu_out = run_filter(&records, &gps, None, &cfg_imu_out, &calib);
    let rmse_imu_out = compute_rmse(&res_imu_out);

    // ── Run 3: GPS outage + all auxiliary sensors (aided dead-reckoning) ─────
    // NHC, ZUPT, and Visual rotation remain active; only GPS is suppressed.
    // Shows what the auxiliary sensors can do to bound drift without GPS.
    let cfg_aided_out = RunConfig {
        sequence_dir: seq_dir, calib_dir, visual_noise_std: Some(0.5),
        gps_outage_frames: Some(outage),
        ..RunConfig::default()
    };
    let res_aided_out = run_filter(&records, &gps, Some(&images), &cfg_aided_out, &calib);
    let rmse_aided_out = compute_rmse(&res_aided_out);

    println!("\n══════════════════════════════════════════════════════");
    println!("Sequence          : {seq_dir}");
    println!("Frames            : {n}");
    println!("GPS outage        : frames {outage_start}–{} ({:.1}s)",
        outage_end - 1, (outage_end - outage_start) as f64 * 0.1);
    println!("Full GPS RMSE     : {rmse_full:.3}m  (no outage, ceiling)");
    println!();

    let dt = 0.1_f64;  // 10 Hz KITTI
    let m_imu   = compute_phase_metrics(&res_imu_out,   outage, dt);
    let m_aided = compute_phase_metrics(&res_aided_out, outage, dt);

    println!("                  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}  {:>8}",
        "Pre-RMSE", "Out-RMSE", "Post-RMSE", "Peak", "End-err", "DriftRate");
    println!("IMU dead-reck   : {:9.3}m {:9.3}m {:9.3}m {:9.3}m {:9.3}m {:7.2}m/s",
        m_imu.pre_rmse, m_imu.during_rmse, m_imu.post_rmse,
        m_imu.peak_error, m_imu.end_error, m_imu.drift_rate);
    println!("Aided dead-r    : {:9.3}m {:9.3}m {:9.3}m {:9.3}m {:9.3}m {:7.2}m/s",
        m_aided.pre_rmse, m_aided.during_rmse, m_aided.post_rmse,
        m_aided.peak_error, m_aided.end_error, m_aided.drift_rate);
    println!("Outage peak benefit: {:+.3}m  ({:+.1}%)",
        m_imu.peak_error - m_aided.peak_error,
        (m_imu.peak_error - m_aided.peak_error) / m_imu.peak_error * 100.0);

    // Write CSV with all three runs + an outage flag column
    {
        use std::fmt::Write as _;
        let mut csv = String::from(
            "frame,outage,gt_x,gt_y,\
             full_gps_x,full_gps_y,\
             imu_dead_x,imu_dead_y,\
             aided_dead_x,aided_dead_y\n");
        for i in 0..n {
            let (full_est,  gt)  = res_full[i];
            let (imu_est,   _)   = res_imu_out[i];
            let (aided_est, _)   = res_aided_out[i];
            let flag = if i >= outage_start && i < outage_end { 1 } else { 0 };
            writeln!(csv,
                "{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
                i, flag, gt.x, gt.y,
                full_est.x,  full_est.y,
                imu_est.x,   imu_est.y,
                aided_est.x, aided_est.y).unwrap();
        }
        let fname = format!("{csv_prefix}_trajectory.csv");
        std::fs::write(&fname, &csv).expect("Failed to write csv");
        println!("CSV written       : {fname}");
    }
}

fn main() {
    run_sequence(
        "data/2011_09_26/2011_09_26_drive_0001_sync",
        "data/2011_09_26",
        "drive_0001",
    );
    run_sequence(
        "data/2011_09_26/2011_09_26_drive_0009_sync",
        "data/2011_09_26",
        "drive_0009",
    );
}
