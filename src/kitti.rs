//! KITTI raw data loader.
//! Each OXTS file is one frame: 30 space-separated f64 values.
//! We load all frames in timestamp order and return them as a Vec.

use std::fs;
use std::path::Path;
use anyhow::{Context, Result};
use nalgebra::Vector3;

use crate::sensors::imu::ImuMeasurement;

/// One OXTS record = one file in oxts/data/.
/// We store IMU fields + GPS fields together for now.
///
/// RUST: `f64` is the default float — same as C++ `double`.
///       `u64` for timestamps — same as C++ `uint64_t`.
#[derive(Debug, Clone)]
pub struct OxtsRecord {
    pub timestamp_ns: u64,
    // GPS
    pub lat: f64,   // degrees
    pub lon: f64,   // degrees
    pub alt: f64,   // metres
    // Orientation (from OXTS, used for initialisation)
    pub roll: f64,  // rad
    pub pitch: f64, // rad
    pub yaw: f64,   // rad
    // IMU — body frame
    pub accel: Vector3<f64>,  // m/s²  (fields 11,12,13)
    pub gyro:  Vector3<f64>,  // rad/s (fields 17,18,19)
    // GPS Doppler velocity in ENU frame from the OXTS GNSS receiver.
    // vn/ve are true Doppler measurements, not the INS fusion output.
    // vel_accuracy is the 1-sigma std dev reported by the GNSS chip (m/s).
    pub vn: f64,              // m/s   north velocity (field 6)
    pub ve: f64,              // m/s   east  velocity (field 7)
    pub vel_accuracy: f64,    // m/s   GNSS velocity 1σ (field 24)
    // Vehicle-frame velocity from OXTS INS solution
    // vf = forward speed (along vehicle x-axis).  Used as a ZUPT trigger:
    // when |vf| < threshold the car is stationary and we inject v = 0.
    pub vf: f64,              // m/s   (field 8)
}

impl OxtsRecord {
    /// Parse one OXTS .txt file into an OxtsRecord.
    ///
    /// RUST: `Path` is a borrowed reference to a path (like `&str` for strings).
    ///       `&Path` ≈ `const std::filesystem::path&` in C++.
    ///       `Result<T>` means "returns T or an error" — the `?` operator
    ///       propagates errors up automatically (like throwing in C++ but explicit).
    pub fn from_file(path: &Path, timestamp_ns: u64) -> Result<Self> {
        let content = fs::read_to_string(path)
            .with_context(|| format!("Failed to read OXTS file: {}", path.display()))?;

        // Split on whitespace → Vec<&str>, then parse each as f64
        // RUST: .collect::<Result<Vec<f64>>>() tries to parse every element,
        //       returning the first error if any field fails.
        //       C++ equivalent: std::istringstream + a loop with error checking.
        let fields: Vec<f64> = content
            .split_whitespace()
            .map(|s| s.parse::<f64>())
            .collect::<std::result::Result<Vec<_>, _>>()
            .with_context(|| format!("Failed to parse floats in: {}", path.display()))?;

        // Sanity check — KITTI OXTS always has 30 fields
        // RUST: anyhow::ensure! is like assert! but returns Err instead of panicking
        anyhow::ensure!(
            fields.len() >= 20,
            "Expected ≥20 OXTS fields, got {} in {}",
            fields.len(),
            path.display()
        );

        // KITTI OXTS has 30 fields; we need up to index 24
        anyhow::ensure!(
            fields.len() >= 25,
            "Expected ≥25 OXTS fields for vel_accuracy, got {} in {}",
            fields.len(), path.display()
        );

        // Field index reference:
        //   0=lat, 1=lon, 2=alt, 3=roll, 4=pitch, 5=yaw
        //   6=vn, 7=ve, 8=vf, 9=vl, 10=vu
        //   11=ax, 12=ay, 13=az,  17=wx, 18=wy, 19=wz
        //   23=pos_accuracy, 24=vel_accuracy
        Ok(Self {
            timestamp_ns,
            lat: fields[0],
            lon: fields[1],
            alt: fields[2],
            roll: fields[3],
            pitch: fields[4],
            yaw: fields[5],
            vn: fields[6],
            ve: fields[7],
            vf: fields[8],
            vel_accuracy: fields[24].max(0.05), // clamp: GNSS sometimes reports 0
            accel: Vector3::new(fields[11], fields[12], fields[13]),
            gyro:  Vector3::new(fields[17], fields[18], fields[19]),
        })
    }
}

/// Load all OXTS records for a sequence, sorted by frame index.
///
/// `sequence_dir` is the path to e.g. data/2011_09_26/2011_09_26_drive_0001_sync/
///
/// RUST: `Vec<T>` is a heap-allocated growable array — same as std::vector<T>.
///       Returning `Result<Vec<OxtsRecord>>` means the whole load can fail cleanly.
pub fn load_oxts(sequence_dir: &Path) -> Result<Vec<OxtsRecord>> {
    let oxts_dir = sequence_dir.join("oxts/data");
    let timestamps = load_timestamps(sequence_dir)?;

    // Collect all .txt files, sort by name (frame order)
    let mut paths: Vec<_> = fs::read_dir(&oxts_dir)
        .with_context(|| format!("Cannot open oxts/data dir: {}", oxts_dir.display()))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |x| x == "txt"))
        .collect();

    paths.sort_by_key(|e| e.file_name());

    // iterate over `paths`, parse each file with OxtsRecord::from_file,
    //       use the matching timestamp from the `timestamps` Vec (same index),
    //       collect into a Vec<OxtsRecord> and return it.
    //
    // Hints:
    //   paths.iter().enumerate()         — gives (index, entry) pairs
    //   timestamps.get(i).copied()       — gets timestamp at index i (Option<u64>)
    //     .unwrap_or(0)                  — fall back to 0 if missing
    //   OxtsRecord::from_file(&path, ts)?  — parse, propagate error with ?
    //   .collect::<Result<Vec<_>>>()?    — collect Results, fail on first error
    let records  = paths.iter().enumerate()
        .map(|(i, entry)| {
            let path = entry.path();
            let ts = timestamps.get(i).copied().unwrap_or(0);
            OxtsRecord::from_file(&path, ts)
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(records)
}

/// Parse oxts/timestamps.txt into nanosecond timestamps.
/// Format: "2011-09-26 13:02:04.338022976\n..."
fn load_timestamps(sequence_dir: &Path) -> Result<Vec<u64>> {
    let ts_path = sequence_dir.join("oxts/timestamps.txt");
    let content = fs::read_to_string(&ts_path)
        .with_context(|| format!("Cannot read timestamps: {}", ts_path.display()))?;

    let timestamps = content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            // "2011-09-26 13:02:04.338022976" → extract fractional seconds
            // We only need relative time so we parse just the time part
            let time_part = line.split_whitespace().nth(1).unwrap_or("0");
            let parts: Vec<&str> = time_part.split(':').collect();
            if parts.len() != 3 {
                return 0u64;
            }
            let h: u64 = parts[0].parse().unwrap_or(0);
            let m: u64 = parts[1].parse().unwrap_or(0);
            let s_frac: f64 = parts[2].parse().unwrap_or(0.0);
            let total_ns = ((h * 3600 + m * 60) as f64 + s_frac) * 1e9;
            total_ns as u64
        })
        .collect();

    Ok(timestamps)
}

/// Convert OxtsRecord → ImuMeasurement (drops GPS fields, keeps IMU)
impl From<OxtsRecord> for ImuMeasurement {
    /// RUST: `From<T>` is a standard conversion trait.
    ///       Implementing it automatically gives you `.into()` for free.
    ///       C++: explicit conversion constructor or a free function — no automatic `.into()`.
    fn from(r: OxtsRecord) -> Self {
        ImuMeasurement::new(r.timestamp_ns, r.gyro, r.accel)
    }
}

/// Convert a slice of OxtsRecords → Vec<GpsMeasurement> in local ENU frame.
/// Uses the first record as the origin (flat-earth approximation — valid for <10km).
///
/// WGS84 flat-earth:
///   Δeast  = (lon - lon0) * cos(lat0) * R_earth * π/180
///   Δnorth = (lat - lat0) * R_earth * π/180
///   Δup    = alt - alt0
pub fn oxts_to_gps_enu(records: &[OxtsRecord]) -> Vec<crate::sensors::gps::GpsMeasurement> {
    use crate::sensors::gps::GpsMeasurement;
    use std::f64::consts::PI;

    const R_EARTH: f64 = 6_378_137.0; // WGS84 semi-major axis in metres

    if records.is_empty() {
        return vec![];
    }

    let lat0_deg = records[0].lat;
    let lat0_rad = lat0_deg * PI / 180.0;
    let lon0 = records[0].lon;
    let alt0 = records[0].alt;

    records.iter().map(|r| {
        let east  = (r.lon - lon0)     * lat0_rad.cos() * R_EARTH * PI / 180.0;
        let north = (r.lat - lat0_deg) * R_EARTH * PI / 180.0;
        let up    = r.alt - alt0;
        GpsMeasurement::new(r.timestamp_ns, Vector3::new(east, north, up))
    }).collect()
}