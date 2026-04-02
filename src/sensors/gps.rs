use nalgebra::Vector3;

/// A GPS position measurement in ENU (East-North-Up) local frame.
/// Converted from WGS84 lat/lon/alt — the conversion happens in kitti.rs.
///
/// RUST: no inheritance — GpsMeasurement is a plain struct, not a subclass
///       of some "Measurement" base. We'll use traits for polymorphism later.
/// C++: same as a plain struct with no virtual methods.
#[derive(Debug, Clone)]
pub struct GpsMeasurement {
    pub timestamp_ns: u64,
    pub position_enu: Vector3<f64>,  // metres, East-North-Up
}

impl GpsMeasurement {
    pub fn new(timestamp_ns: u64, position_enu: Vector3<f64>) -> Self {
        Self {
            timestamp_ns,
            position_enu,
        }
    }
}