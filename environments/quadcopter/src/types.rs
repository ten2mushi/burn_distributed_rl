//! Type-level encoding of physical invariants following Curry-Howard correspondence.
//!
//! Each type encodes a proposition about valid values:
//! - `UnitQuaternion`: ||q|| = 1 (rotation representation)
//! - `BoundedRPM`: 0 <= rpm <= MAX_RPM (motor speed)
//! - `NormalizedAction`: -1 <= action <= 1 (policy output)
//! - `PositiveScalar`: x > 0 (timesteps, frequencies, physical constants)
//!
//! These types are used at **API boundaries** for validation.
//! Internal hot paths extract raw values via `.value()` or `.as_array()`.

use crate::constants::{action_to_rpm, MAX_RPM, MIN_RPM};

// ============================================================================
// Unit Quaternion
// ============================================================================

/// A quaternion with unit norm (||q|| = 1), representing a rotation.
///
/// Invariant: w^2 + x^2 + y^2 + z^2 = 1
///
/// Used at API boundaries for orientation get/set operations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnitQuaternion {
    w: f32,
    x: f32,
    y: f32,
    z: f32,
}

impl UnitQuaternion {
    /// Create a unit quaternion, normalizing the input.
    ///
    /// Returns identity quaternion if input has zero norm.
    #[inline]
    pub fn new(w: f32, x: f32, y: f32, z: f32) -> Self {
        let norm_sq = w * w + x * x + y * y + z * z;
        if norm_sq < 1e-10 {
            return Self::identity();
        }
        let inv_norm = 1.0 / norm_sq.sqrt();
        Self {
            w: w * inv_norm,
            x: x * inv_norm,
            y: y * inv_norm,
            z: z * inv_norm,
        }
    }

    /// Create from array [w, x, y, z], normalizing.
    #[inline]
    pub fn from_array(arr: [f32; 4]) -> Self {
        Self::new(arr[0], arr[1], arr[2], arr[3])
    }

    /// Create without validation.
    ///
    /// # Safety
    /// Caller must ensure w^2 + x^2 + y^2 + z^2 = 1
    #[inline(always)]
    pub unsafe fn new_unchecked(w: f32, x: f32, y: f32, z: f32) -> Self {
        debug_assert!(
            (w * w + x * x + y * y + z * z - 1.0).abs() < 1e-4,
            "Quaternion not normalized"
        );
        Self { w, x, y, z }
    }

    /// Identity quaternion (no rotation).
    #[inline]
    pub const fn identity() -> Self {
        Self {
            w: 1.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// Extract as array [w, x, y, z] for internal use.
    #[inline(always)]
    pub fn as_array(&self) -> [f32; 4] {
        [self.w, self.x, self.y, self.z]
    }

    /// Get individual components.
    #[inline(always)]
    pub fn w(&self) -> f32 {
        self.w
    }
    #[inline(always)]
    pub fn x(&self) -> f32 {
        self.x
    }
    #[inline(always)]
    pub fn y(&self) -> f32 {
        self.y
    }
    #[inline(always)]
    pub fn z(&self) -> f32 {
        self.z
    }

    /// Compute squared norm (should always be ~1.0).
    #[inline]
    pub fn norm_squared(&self) -> f32 {
        self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z
    }
}

impl Default for UnitQuaternion {
    fn default() -> Self {
        Self::identity()
    }
}

// ============================================================================
// Coordinate Frame Markers (Zero-Size Types)
// ============================================================================

use std::marker::PhantomData;

/// Marker type for world frame coordinates.
/// World frame is inertial, with Z pointing up (against gravity).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct WorldFrame;

/// Marker type for body frame coordinates.
/// Body frame is attached to the quadcopter, with Z along thrust axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct BodyFrame;

// ============================================================================
// Frame-Typed 3D Vector
// ============================================================================

/// A 3D vector with compile-time coordinate frame tracking.
///
/// This is a zero-cost abstraction - `PhantomData<Frame>` has no runtime cost.
/// The type system prevents mixing vectors from different frames.
///
/// # Example
/// ```ignore
/// let pos: Vec3<WorldFrame> = Vec3::new(1.0, 2.0, 3.0);
/// let vel: Vec3<BodyFrame> = Vec3::new(0.1, 0.2, 0.3);
/// // let bad = pos + vel;  // COMPILE ERROR: frame mismatch
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct Vec3<Frame> {
    data: [f32; 3],
    _frame: PhantomData<Frame>,
}

impl<F> Vec3<F> {
    /// Create a new vector in the specified frame.
    #[inline(always)]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self {
            data: [x, y, z],
            _frame: PhantomData,
        }
    }

    /// Create from array.
    #[inline(always)]
    pub const fn from_array(arr: [f32; 3]) -> Self {
        Self {
            data: arr,
            _frame: PhantomData,
        }
    }

    /// Extract as raw array for SIMD hot paths.
    #[inline(always)]
    pub const fn as_array(&self) -> [f32; 3] {
        self.data
    }

    /// Get x component.
    #[inline(always)]
    pub const fn x(&self) -> f32 {
        self.data[0]
    }

    /// Get y component.
    #[inline(always)]
    pub const fn y(&self) -> f32 {
        self.data[1]
    }

    /// Get z component.
    #[inline(always)]
    pub const fn z(&self) -> f32 {
        self.data[2]
    }

    /// Compute squared magnitude (frame-independent).
    #[inline(always)]
    pub fn norm_squared(&self) -> f32 {
        self.data[0] * self.data[0] + self.data[1] * self.data[1] + self.data[2] * self.data[2]
    }

    /// Compute magnitude (frame-independent).
    #[inline]
    pub fn norm(&self) -> f32 {
        self.norm_squared().sqrt()
    }

    /// Dot product with another vector in the same frame.
    #[inline(always)]
    pub fn dot(&self, other: &Vec3<F>) -> f32 {
        self.data[0] * other.data[0] + self.data[1] * other.data[1] + self.data[2] * other.data[2]
    }

    /// Cross product with another vector in the same frame.
    #[inline(always)]
    pub fn cross(&self, other: &Vec3<F>) -> Vec3<F> {
        Vec3::new(
            self.data[1] * other.data[2] - self.data[2] * other.data[1],
            self.data[2] * other.data[0] - self.data[0] * other.data[2],
            self.data[0] * other.data[1] - self.data[1] * other.data[0],
        )
    }

    /// Zero vector.
    #[inline(always)]
    pub const fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }
}

// Arithmetic operations within the same frame
impl<F> std::ops::Add for Vec3<F> {
    type Output = Vec3<F>;
    #[inline(always)]
    fn add(self, other: Vec3<F>) -> Vec3<F> {
        Vec3::new(
            self.data[0] + other.data[0],
            self.data[1] + other.data[1],
            self.data[2] + other.data[2],
        )
    }
}

impl<F> std::ops::Sub for Vec3<F> {
    type Output = Vec3<F>;
    #[inline(always)]
    fn sub(self, other: Vec3<F>) -> Vec3<F> {
        Vec3::new(
            self.data[0] - other.data[0],
            self.data[1] - other.data[1],
            self.data[2] - other.data[2],
        )
    }
}

impl<F> std::ops::Mul<f32> for Vec3<F> {
    type Output = Vec3<F>;
    #[inline(always)]
    fn mul(self, scalar: f32) -> Vec3<F> {
        Vec3::new(
            self.data[0] * scalar,
            self.data[1] * scalar,
            self.data[2] * scalar,
        )
    }
}

impl<F> std::ops::Neg for Vec3<F> {
    type Output = Vec3<F>;
    #[inline(always)]
    fn neg(self) -> Vec3<F> {
        Vec3::new(-self.data[0], -self.data[1], -self.data[2])
    }
}

impl<F> Default for Vec3<F> {
    fn default() -> Self {
        Self::zero()
    }
}

// ============================================================================
// Rotation (Frame Transformation)
// ============================================================================

/// A rotation matrix for transforming between coordinate frames.
///
/// Created from a `UnitQuaternion`, this type enables type-safe
/// frame transformations between world and body frames.
#[derive(Clone, Copy, Debug)]
pub struct Rotation {
    /// Row-major rotation matrix [r00, r01, r02, r10, r11, r12, r20, r21, r22]
    matrix: [f32; 9],
}

impl Rotation {
    /// Create rotation from raw matrix (row-major).
    ///
    /// # Safety
    /// Caller must ensure the matrix is a valid rotation matrix (orthonormal, det=1).
    #[inline]
    pub const fn from_matrix_unchecked(matrix: [f32; 9]) -> Self {
        Self { matrix }
    }

    /// Get the raw rotation matrix (for SIMD hot paths).
    #[inline(always)]
    pub const fn as_matrix(&self) -> &[f32; 9] {
        &self.matrix
    }

    /// Transform a vector from world frame to body frame (R^T * v).
    #[inline(always)]
    pub fn world_to_body(&self, v: Vec3<WorldFrame>) -> Vec3<BodyFrame> {
        let r = &self.matrix;
        let [vx, vy, vz] = v.as_array();
        Vec3::new(
            r[0] * vx + r[3] * vy + r[6] * vz, // R^T row 0
            r[1] * vx + r[4] * vy + r[7] * vz, // R^T row 1
            r[2] * vx + r[5] * vy + r[8] * vz, // R^T row 2
        )
    }

    /// Transform a vector from body frame to world frame (R * v).
    #[inline(always)]
    pub fn body_to_world(&self, v: Vec3<BodyFrame>) -> Vec3<WorldFrame> {
        let r = &self.matrix;
        let [vx, vy, vz] = v.as_array();
        Vec3::new(
            r[0] * vx + r[1] * vy + r[2] * vz, // R row 0
            r[3] * vx + r[4] * vy + r[5] * vz, // R row 1
            r[6] * vx + r[7] * vy + r[8] * vz, // R row 2
        )
    }

    /// Get the thrust direction (body Z-axis in world coordinates).
    #[inline(always)]
    pub fn thrust_direction(&self) -> Vec3<WorldFrame> {
        let r = &self.matrix;
        Vec3::new(r[2], r[5], r[8]) // Z column
    }

    /// Identity rotation (no transformation).
    #[inline]
    pub const fn identity() -> Self {
        Self {
            matrix: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        }
    }
}

impl Default for Rotation {
    fn default() -> Self {
        Self::identity()
    }
}

// ============================================================================
// Bounded RPM
// ============================================================================

/// Motor RPM bounded to valid range [MIN_RPM, MAX_RPM].
///
/// Invariant: MIN_RPM <= rpm <= MAX_RPM
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct BoundedRPM(f32);

impl BoundedRPM {
    /// Create with clamping to valid range.
    #[inline]
    pub fn new(rpm: f32) -> Self {
        Self(rpm.clamp(MIN_RPM, MAX_RPM))
    }

    /// Create without clamping.
    ///
    /// # Safety
    /// Caller must ensure MIN_RPM <= rpm <= MAX_RPM
    #[inline(always)]
    pub unsafe fn new_unchecked(rpm: f32) -> Self {
        debug_assert!(rpm >= MIN_RPM && rpm <= MAX_RPM, "RPM out of bounds");
        Self(rpm)
    }

    /// Hover RPM (balances gravity).
    #[inline]
    pub fn hover() -> Self {
        Self(crate::constants::HOVER_RPM)
    }

    /// Zero RPM (motors off).
    #[inline]
    pub const fn zero() -> Self {
        Self(0.0)
    }

    /// Maximum RPM.
    #[inline]
    pub fn max() -> Self {
        Self(MAX_RPM)
    }

    /// Extract raw value for internal use.
    #[inline(always)]
    pub fn value(&self) -> f32 {
        self.0
    }

    /// Convert to normalized action [-1, 1].
    #[inline]
    pub fn to_action(&self) -> NormalizedAction {
        NormalizedAction::new(crate::constants::rpm_to_action(self.0))
    }
}

impl Default for BoundedRPM {
    fn default() -> Self {
        Self::hover()
    }
}

// ============================================================================
// Normalized Action
// ============================================================================

/// Action value normalized to [-1, 1].
///
/// Invariant: -1 <= action <= 1
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct NormalizedAction(f32);

impl NormalizedAction {
    /// Create with clamping to valid range.
    #[inline]
    pub fn new(action: f32) -> Self {
        Self(action.clamp(-1.0, 1.0))
    }

    /// Create without clamping.
    ///
    /// # Safety
    /// Caller must ensure -1 <= action <= 1
    #[inline(always)]
    pub unsafe fn new_unchecked(action: f32) -> Self {
        debug_assert!(
            action >= -1.0 && action <= 1.0,
            "Action out of bounds: {}",
            action
        );
        Self(action)
    }

    /// Zero action (neutral).
    #[inline]
    pub const fn zero() -> Self {
        Self(0.0)
    }

    /// Hover action (produces hover RPM).
    #[inline]
    pub fn hover() -> Self {
        Self::new(crate::constants::rpm_to_action(crate::constants::HOVER_RPM))
    }

    /// Extract raw value for internal use.
    #[inline(always)]
    pub fn value(&self) -> f32 {
        self.0
    }

    /// Convert to motor RPM.
    #[inline]
    pub fn to_rpm(&self) -> BoundedRPM {
        BoundedRPM::new(action_to_rpm(self.0))
    }
}

impl Default for NormalizedAction {
    fn default() -> Self {
        Self::zero()
    }
}

// ============================================================================
// Positive Scalar
// ============================================================================

/// A scalar value that must be positive (x > 0).
///
/// Used for timesteps, frequencies, masses, inertias, etc.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct PositiveScalar(f32);

impl PositiveScalar {
    /// Try to create, returning None if value <= 0.
    #[inline]
    pub fn new(value: f32) -> Option<Self> {
        if value > 0.0 {
            Some(Self(value))
        } else {
            None
        }
    }

    /// Create, panicking if value <= 0.
    ///
    /// Use in const contexts or when validity is guaranteed.
    #[inline]
    pub fn new_or_panic(value: f32) -> Self {
        assert!(value > 0.0, "PositiveScalar requires value > 0, got {}", value);
        Self(value)
    }

    /// Create without validation.
    ///
    /// # Safety
    /// Caller must ensure value > 0
    #[inline(always)]
    pub unsafe fn new_unchecked(value: f32) -> Self {
        debug_assert!(value > 0.0, "PositiveScalar requires value > 0");
        Self(value)
    }

    /// Extract raw value for internal use.
    #[inline(always)]
    pub fn value(&self) -> f32 {
        self.0
    }

    /// Compute reciprocal (also positive).
    #[inline]
    pub fn recip(&self) -> Self {
        Self(1.0 / self.0)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unit_quaternion_normalization() {
        let q = UnitQuaternion::new(1.0, 2.0, 3.0, 4.0);
        let norm_sq = q.norm_squared();
        assert!((norm_sq - 1.0).abs() < 1e-6, "Not normalized: {}", norm_sq);
    }

    #[test]
    fn test_unit_quaternion_identity() {
        let q = UnitQuaternion::identity();
        assert_eq!(q.as_array(), [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_unit_quaternion_zero_input() {
        let q = UnitQuaternion::new(0.0, 0.0, 0.0, 0.0);
        assert_eq!(q, UnitQuaternion::identity());
    }

    #[test]
    fn test_bounded_rpm_clamping() {
        assert_eq!(BoundedRPM::new(-100.0).value(), MIN_RPM);
        assert_eq!(BoundedRPM::new(100000.0).value(), MAX_RPM);
        assert_eq!(BoundedRPM::new(10000.0).value(), 10000.0);
    }

    #[test]
    fn test_bounded_rpm_hover() {
        let hover = BoundedRPM::hover();
        assert!((hover.value() - crate::constants::HOVER_RPM).abs() < 1.0);
    }

    #[test]
    fn test_normalized_action_clamping() {
        assert_eq!(NormalizedAction::new(-2.0).value(), -1.0);
        assert_eq!(NormalizedAction::new(2.0).value(), 1.0);
        assert_eq!(NormalizedAction::new(0.5).value(), 0.5);
    }

    #[test]
    fn test_action_rpm_roundtrip() {
        let rpm = BoundedRPM::new(15000.0);
        let action = rpm.to_action();
        let rpm_back = action.to_rpm();
        assert!((rpm.value() - rpm_back.value()).abs() < 1.0);
    }

    #[test]
    fn test_positive_scalar_valid() {
        assert!(PositiveScalar::new(1.0).is_some());
        assert!(PositiveScalar::new(0.001).is_some());
        assert!(PositiveScalar::new(1e10).is_some());
    }

    #[test]
    fn test_positive_scalar_invalid() {
        assert!(PositiveScalar::new(0.0).is_none());
        assert!(PositiveScalar::new(-1.0).is_none());
        assert!(PositiveScalar::new(-0.001).is_none());
    }

    #[test]
    fn test_positive_scalar_recip() {
        let s = PositiveScalar::new(4.0).unwrap();
        assert_eq!(s.recip().value(), 0.25);
    }

    // ========================================================================
    // Frame Type Tests
    // ========================================================================

    #[test]
    fn test_vec3_same_frame_addition() {
        let a: Vec3<WorldFrame> = Vec3::new(1.0, 2.0, 3.0);
        let b: Vec3<WorldFrame> = Vec3::new(4.0, 5.0, 6.0);
        let sum = a + b;
        assert_eq!(sum.as_array(), [5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_vec3_same_frame_subtraction() {
        let a: Vec3<BodyFrame> = Vec3::new(5.0, 7.0, 9.0);
        let b: Vec3<BodyFrame> = Vec3::new(1.0, 2.0, 3.0);
        let diff = a - b;
        assert_eq!(diff.as_array(), [4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_vec3_scalar_multiply() {
        let v: Vec3<WorldFrame> = Vec3::new(1.0, 2.0, 3.0);
        let scaled = v * 2.0;
        assert_eq!(scaled.as_array(), [2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_vec3_negation() {
        let v: Vec3<BodyFrame> = Vec3::new(1.0, -2.0, 3.0);
        let neg = -v;
        assert_eq!(neg.as_array(), [-1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_vec3_norm() {
        let v: Vec3<WorldFrame> = Vec3::new(3.0, 4.0, 0.0);
        assert!((v.norm() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_vec3_dot_product() {
        let a: Vec3<BodyFrame> = Vec3::new(1.0, 0.0, 0.0);
        let b: Vec3<BodyFrame> = Vec3::new(0.0, 1.0, 0.0);
        assert!(a.dot(&b).abs() < 1e-6); // Perpendicular

        let c: Vec3<BodyFrame> = Vec3::new(1.0, 2.0, 3.0);
        let d: Vec3<BodyFrame> = Vec3::new(4.0, 5.0, 6.0);
        assert!((c.dot(&d) - 32.0).abs() < 1e-6); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_vec3_cross_product() {
        let x: Vec3<WorldFrame> = Vec3::new(1.0, 0.0, 0.0);
        let y: Vec3<WorldFrame> = Vec3::new(0.0, 1.0, 0.0);
        let z = x.cross(&y);
        assert!((z.x()).abs() < 1e-6);
        assert!((z.y()).abs() < 1e-6);
        assert!((z.z() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_rotation_identity() {
        let r = Rotation::identity();
        let v_world: Vec3<WorldFrame> = Vec3::new(1.0, 2.0, 3.0);
        let v_body = r.world_to_body(v_world);
        assert!((v_body.x() - 1.0).abs() < 1e-6);
        assert!((v_body.y() - 2.0).abs() < 1e-6);
        assert!((v_body.z() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_rotation_roundtrip() {
        // 90 degree rotation around Z axis
        let r = Rotation::from_matrix_unchecked([
            0.0, -1.0, 0.0, // Row 0
            1.0, 0.0, 0.0,  // Row 1
            0.0, 0.0, 1.0,  // Row 2
        ]);
        let v_world: Vec3<WorldFrame> = Vec3::new(1.0, 0.0, 0.0);
        let v_body = r.world_to_body(v_world);
        let v_back = r.body_to_world(v_body);
        assert!((v_back.x() - v_world.x()).abs() < 1e-5);
        assert!((v_back.y() - v_world.y()).abs() < 1e-5);
        assert!((v_back.z() - v_world.z()).abs() < 1e-5);
    }

    #[test]
    fn test_rotation_thrust_direction_identity() {
        let r = Rotation::identity();
        let thrust = r.thrust_direction();
        // Identity should give [0, 0, 1] (thrust up in world frame)
        assert!(thrust.x().abs() < 1e-6);
        assert!(thrust.y().abs() < 1e-6);
        assert!((thrust.z() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_vec3_zero() {
        let z: Vec3<WorldFrame> = Vec3::zero();
        assert_eq!(z.as_array(), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_vec3_from_array() {
        let arr = [1.0, 2.0, 3.0];
        let v: Vec3<BodyFrame> = Vec3::from_array(arr);
        assert_eq!(v.x(), 1.0);
        assert_eq!(v.y(), 2.0);
        assert_eq!(v.z(), 3.0);
    }

    // Compile-time safety documentation (this code should NOT compile if uncommented):
    // #[test]
    // fn test_frame_mismatch_fails_to_compile() {
    //     let world: Vec3<WorldFrame> = Vec3::new(1.0, 2.0, 3.0);
    //     let body: Vec3<BodyFrame> = Vec3::new(4.0, 5.0, 6.0);
    //     let _ = world + body;  // ERROR: expected Vec3<WorldFrame>, found Vec3<BodyFrame>
    // }
}
