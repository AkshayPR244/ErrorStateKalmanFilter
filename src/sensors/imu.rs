use nalgebra::Vector3;

/// A single IMU measurement from the KITTI OXTS file.
///
/// RUST: #[derive(Debug, Clone)] auto-generates two things:
///   - Debug  → lets you print the struct with {:?}
///   - Clone  → lets you explicitly copy it with .clone()
///
/// C++ equivalent:
///   struct ImuMeasurement {
///       uint64_t timestamp_ns;
///       Eigen::Vector3d angular_velocity;
///       Eigen::Vector3d linear_acceleration;
///       ImuMeasurement(const ImuMeasurement&) = default;  // Clone
///   };
#[derive(Debug, Clone)]
pub struct ImuMeasurement 
{
    pub timestamp_ns: u64,
    pub angular_velocity: Vector3<f64>,    // rad/s  (gyroscope)
    pub linear_acceleration: Vector3<f64>, // m/s²   (accelerometer)
}

impl ImuMeasurement 
{
    /// Named constructor — Rust has no overloaded constructors.
    ///
    /// C++: ImuMeasurement(uint64_t t, Eigen::Vector3d w, Eigen::Vector3d a)
    /// Rust: the return type `Self` means "the type this impl block is for"
    pub fn new(
        timestamp_ns: u64,
        angular_velocity: Vector3<f64>,
        linear_acceleration: Vector3<f64>,
    ) -> Self 
    {
        Self {
            timestamp_ns,
            angular_velocity,
            linear_acceleration,
        }
    }
}