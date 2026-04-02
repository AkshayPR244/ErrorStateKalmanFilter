use crate::sensors::imu::ImuMeasurement;
use nalgebra::UnitQuaternion;
use nalgebra::Vector3;

/// The nominal (best-estimate) state of the vehicle.
/// The ESKF maintains this + a 15-dim error state on top of it.
///
/// State vector: x = [p, v, q, b_a, b_g]
///   p  — position in world frame          (3-dim)
///   v  — velocity in world frame          (3-dim)
///   q  — orientation as unit quaternion   (4-dim, but 3 DOF)
///   b_a — accelerometer bias              (3-dim)
///   b_g — gyroscope bias                  (3-dim)
/// Total nominal DOF = 15 (same as error state)
///
/// RUST: UnitQuaternion is nalgebra's type for a quaternion
///       constrained to unit norm. The type system enforces
///       the constraint — you can't accidentally use an
///       unnormalized quaternion where UnitQuaternion is expected.
///
/// C++: Eigen::Quaterniond is the closest equivalent, but it
///      does NOT enforce unit norm at compile time — you have to
///      call .normalize() manually and hope you don't forget.
#[derive(Debug, Clone)]
pub struct NominalState {
    pub position: Vector3<f64>,
    pub velocity: Vector3<f64>,
    pub orientation: UnitQuaternion<f64>,
    pub accel_bias: Vector3<f64>,
    pub gyro_bias: Vector3<f64>,
}

impl NominalState {
    /// Create a state at the origin, at rest, with identity orientation
    /// and zero biases. This is a common initialisation in filter code.
    pub fn identity() -> Self {
        // RUST: nalgebra provides convenient constructors for common vectors/quaternions:
        //   - Vector3::zeros()                   — zero 3-vector
        //   - UnitQuaternion::identity()          — no rotation
        Self {
            position: Vector3::zeros(),
            velocity: Vector3::zeros(),
            orientation: UnitQuaternion::identity(),
            accel_bias: Vector3::zeros(),
            gyro_bias: Vector3::zeros(),
        }
    }

    /// Propagate nominal state forward by one IMU step.
    ///
    /// RUST: `&mut self` means this method mutates the state in place.
    ///       The compiler guarantees no other reference to self exists
    ///       while this runs — aliasing is impossible.
    /// C++: void propagate(ImuMeasurement& imu, double dt)
    ///      — same intent but the compiler doesn't enforce exclusivity.
    pub fn propagate(&mut self, imu: &ImuMeasurement, dt: f64) {
        use nalgebra::{UnitQuaternion, Vector3};

        const GRAVITY: f64 = 9.81;
        let g_world = Vector3::new(0.0, 0.0, -GRAVITY); // ENU: gravity points down

        // 1. Corrected IMU measurements (subtract estimated biases)
        let accel_corrected = imu.linear_acceleration - self.accel_bias;
        let gyro_corrected  = imu.angular_velocity    - self.gyro_bias;

        // 2. Rotate acceleration from body frame → world frame
        //    R = rotation matrix from the current orientation quaternion
        let accel_world = self.orientation * accel_corrected;

        // 3. Integrate position and velocity (Euler integration)
        // TODO: update self.position using self.velocity and dt
        self.position += self.velocity * dt;
        // TODO: update self.velocity using accel_world, g_world, and dt
        self.velocity += (accel_world + g_world) * dt;

        // 4. Integrate orientation via quaternion kinematics
        //    Δq = exp(½ ω dt) — small rotation quaternion
        let half_angle = gyro_corrected * (0.5 * dt);
        let delta_q = UnitQuaternion::from_scaled_axis(half_angle);
        // Update self.orientation by composing with delta_q
        //       (quaternion multiplication = composing rotations)
        self.orientation = self.orientation * delta_q;
        // Note: the order of multiplication matters — this applies the new rotation after the existing one
    }

    /// Initialise state from the first OXTS record's GPS + orientation.
    /// roll/pitch/yaw are in radians, from the OXTS file fields 3,4,5.
    pub fn from_oxts(position: Vector3<f64>, roll: f64, pitch: f64, yaw: f64) -> Self {
        use nalgebra::UnitQuaternion;
        // Build quaternion from Euler angles (roll=X, pitch=Y, yaw=Z)
        // This is the RPY / ZYX convention used by KITTI OXTS
        let orientation = UnitQuaternion::from_euler_angles(roll, pitch, yaw);
        Self {
            position,
            velocity: Vector3::zeros(),
            orientation,
            accel_bias: Vector3::zeros(),
            gyro_bias: Vector3::zeros(),
        }
    }
}