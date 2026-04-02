//! Type-safe coordinate wrappers using the newtype pattern.
//!
//! RUST: A "newtype" is a tuple struct with one field.
//!       struct Degrees(f64) and struct Radians(f64) are distinct types
//!       even though both contain an f64 — the compiler rejects mixing them.
//!
//! C++: `typedef double Degrees` does NOT do this — Degrees and Radians
//!      are still the same type and can be mixed silently.
//!      The equivalent in C++ is: enum class Degrees : double {};
//!      but that's rarely used in practice. In Rust this is idiomatic.

/// Latitude or longitude in degrees (WGS84)
#[derive(Debug, Clone, Copy)]
pub struct Degrees(pub f64);

/// Angle in radians
#[derive(Debug, Clone, Copy)]
pub struct Radians(pub f64);

impl From<Degrees> for Radians {
    /// RUST: Implementing From<Degrees> for Radians automatically gives you
    ///       let r: Radians = deg.into();  — the Into trait is derived for free.
    /// C++: explicit conversion constructor:
    ///       explicit Radians(Degrees d) { value = d.value * M_PI / 180.0; }
    fn from(d: Degrees) -> Self {
        use std::f64::consts::PI;
        Radians(d.0 * PI / 180.0)
    }
}

impl Radians {
    pub fn cos(self) -> f64 { self.0.cos() }
    pub fn sin(self) -> f64 { self.0.sin() }
    pub fn value(self) -> f64 { self.0 }
}

impl Degrees {
    pub fn value(self) -> f64 { self.0 }
}