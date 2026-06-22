//! SE(3) / SO(3) spatial math for Caliper.
//!
//! Convention: a twist / screw axis is stored as a 6-vector in **`[v; ω]`** order
//! (linear part first, angular last), matching Pinocchio's `Motion` / `log6` / `exp6`
//! so cross-validation is an element-wise comparison. (Modern Robotics uses `[ω; v]`;
//! swap blocks when porting those formulas.)
use nalgebra::{Isometry3, Matrix3, Matrix6, Translation3, UnitQuaternion, Vector3, Vector6};

/// Small-angle threshold on θ = ‖ω‖.
const EPS: f64 = 1e-8;

/// A twist / screw axis in se(3), stored `[v; ω]` (linear 0..3, angular 3..6).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Twist(pub Vector6<f64>);

impl Twist {
    #[inline]
    pub fn from_vw(v: Vector3<f64>, w: Vector3<f64>) -> Self {
        Twist(Vector6::new(v.x, v.y, v.z, w.x, w.y, w.z))
    }
    #[inline]
    pub fn v(&self) -> Vector3<f64> {
        self.0.fixed_rows::<3>(0).into()
    }
    #[inline]
    pub fn w(&self) -> Vector3<f64> {
        self.0.fixed_rows::<3>(3).into()
    }
}

/// A rigid transform in SE(3). Newtype over nalgebra's quaternion-backed `Isometry3`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Se3(pub Isometry3<f64>);

impl Se3 {
    pub fn identity() -> Self {
        Se3(Isometry3::identity())
    }
    pub fn from_parts(t: Vector3<f64>, r: UnitQuaternion<f64>) -> Self {
        Se3(Isometry3::from_parts(Translation3::from(t), r))
    }
    #[inline]
    pub fn rotation(&self) -> Matrix3<f64> {
        self.0.rotation.to_rotation_matrix().into_inner()
    }
    #[inline]
    pub fn translation_vec(&self) -> Vector3<f64> {
        self.0.translation.vector
    }
    pub fn translation(&self) -> [f64; 3] {
        let t = self.0.translation.vector;
        [t.x, t.y, t.z]
    }
    #[inline]
    pub fn inverse(&self) -> Se3 {
        Se3(self.0.inverse())
    }
    #[inline]
    pub fn compose(&self, rhs: &Se3) -> Se3 {
        Se3(self.0 * rhs.0)
    }

    /// SE(3) exponential `exp([S])`, where `s` holds the full screw (θ pre-multiplied in).
    pub fn exp(s: &Twist) -> Se3 {
        let (v, phi) = (s.v(), s.w());
        let theta = phi.norm();
        let wx = skew(&phi);
        let rot = UnitQuaternion::from_scaled_axis(phi);
        let vmat = if theta < EPS {
            // V ≈ I + ½[φ] + (1/6)[φ]²
            Matrix3::identity() + 0.5 * wx + (1.0 / 6.0) * (wx * wx)
        } else {
            let a = (1.0 - theta.cos()) / (theta * theta); // (1-cosθ)/θ²
            let b = (theta - theta.sin()) / theta.powi(3); // (θ-sinθ)/θ³
            Matrix3::identity() + a * wx + b * (wx * wx)
        };
        Se3::from_parts(vmat * v, rot)
    }

    /// SE(3) log: the full screw with θ absorbed, so `Se3::exp(&t.log()) == t` (away from θ=π).
    pub fn log(&self) -> Twist {
        let p = self.translation_vec();
        let phi = self.0.rotation.scaled_axis(); // = ω·θ (SO(3) log, 0 at identity)
        let theta = phi.norm();
        let wx = skew(&phi);
        let vinv = if theta < EPS {
            Matrix3::identity() - 0.5 * wx + (1.0 / 12.0) * (wx * wx)
        } else {
            let c = 1.0 / (theta * theta) - (1.0 + theta.cos()) / (2.0 * theta * theta.sin());
            Matrix3::identity() - 0.5 * wx + c * (wx * wx)
        };
        Twist::from_vw(vinv * p, phi)
    }

    /// 6×6 adjoint mapping twists in `[v;ω]`: `ξ_a = Ad(T) ξ_b`. `Ad = [[R, [p]×R],[0, R]]`.
    pub fn adjoint(&self) -> Matrix6<f64> {
        let r = self.rotation();
        let p = self.translation_vec();
        let mut ad = Matrix6::zeros();
        ad.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
        ad.fixed_view_mut::<3, 3>(0, 3).copy_from(&(skew(&p) * r));
        ad.fixed_view_mut::<3, 3>(3, 3).copy_from(&r);
        ad
    }

    /// Inverse adjoint (exact, cheaper than inverting the 6×6): `[[Rᵀ, -Rᵀ[p]×],[0, Rᵀ]]`.
    pub fn adjoint_inv(&self) -> Matrix6<f64> {
        let rt = self.rotation().transpose();
        let p = self.translation_vec();
        let mut ad = Matrix6::zeros();
        ad.fixed_view_mut::<3, 3>(0, 0).copy_from(&rt);
        ad.fixed_view_mut::<3, 3>(0, 3).copy_from(&(-(rt * skew(&p))));
        ad.fixed_view_mut::<3, 3>(3, 3).copy_from(&rt);
        ad
    }
}

impl std::ops::Mul for Se3 {
    type Output = Se3;
    fn mul(self, r: Se3) -> Se3 {
        Se3(self.0 * r.0)
    }
}
impl Default for Se3 {
    fn default() -> Self {
        Self::identity()
    }
}

/// The 3×3 skew-symmetric (cross-product) matrix of `w`.
#[inline]
pub fn skew(w: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -w.z, w.y, w.z, 0.0, -w.x, -w.y, w.x, 0.0)
}

/// Lie-bracket adjoint `ad_ξ` for `[v;ω]`: `[[[ω], [v]],[0, [ω]]]` (used by dynamics later).
pub fn small_adjoint(s: &Twist) -> Matrix6<f64> {
    let (v, w) = (s.v(), s.w());
    let mut a = Matrix6::zeros();
    a.fixed_view_mut::<3, 3>(0, 0).copy_from(&skew(&w));
    a.fixed_view_mut::<3, 3>(0, 3).copy_from(&skew(&v));
    a.fixed_view_mut::<3, 3>(3, 3).copy_from(&skew(&w));
    a
}

/// Local transform of a revolute joint rotating by `q` about unit `axis` through its origin.
pub fn exp_revolute(axis: &Vector3<f64>, q: f64) -> Se3 {
    Se3::exp(&Twist::from_vw(Vector3::zeros(), axis * q))
}

/// Local transform of a prismatic joint translating by `q` along unit `axis`.
pub fn exp_prismatic(axis: &Vector3<f64>, q: f64) -> Se3 {
    Se3::exp(&Twist::from_vw(axis * q, Vector3::zeros()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Minimal SplitMix64 → [0,1) so tests stay dependency-free.
    struct Rng(u64);
    impl Rng {
        fn f(&mut self) -> f64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z >> 11) as f64 / ((1u64 << 53) as f64)
        }
        fn unit(&mut self) -> f64 {
            self.f() * 2.0 - 1.0
        }
        fn twist(&mut self) -> Twist {
            let ax = Vector3::new(self.unit(), self.unit(), self.unit()).normalize();
            let angle = 1e-6 + self.f() * (PI - 1e-2 - 1e-6); // stay away from θ=π
            let v = Vector3::new(self.unit(), self.unit(), self.unit());
            Twist::from_vw(v, ax * angle)
        }
    }

    fn se3_diff(a: &Se3, b: &Se3) -> f64 {
        (a.0.to_homogeneous() - b.0.to_homogeneous()).norm()
    }

    #[test]
    fn exp_log_roundtrip() {
        let mut r = Rng(0x1234_5678);
        for _ in 0..1000 {
            let t = r.twist();
            let big_t = Se3::exp(&t);
            assert!(se3_diff(&Se3::exp(&big_t.log()), &big_t) < 1e-11);
            assert!((big_t.log().0 - t.0).norm() < 1e-9);
        }
    }

    #[test]
    fn small_angle_branch() {
        let t = Twist::from_vw(Vector3::new(0.3, -0.2, 0.1), Vector3::new(1e-10, -1e-10, 5e-11));
        let big_t = Se3::exp(&t);
        assert!(se3_diff(&Se3::exp(&big_t.log()), &big_t) < 1e-12);
    }

    #[test]
    fn adjoint_inverse_identity() {
        let mut r = Rng(0xABCD);
        let i6 = Matrix6::<f64>::identity();
        for _ in 0..200 {
            let t = Se3::exp(&r.twist());
            assert!((t.adjoint() * t.adjoint_inv() - i6).norm() < 1e-10);
        }
    }

    #[test]
    fn adjoint_homomorphism() {
        let mut r = Rng(0xFEED);
        for _ in 0..200 {
            let a = Se3::exp(&r.twist());
            let b = Se3::exp(&r.twist());
            assert!(((a * b).adjoint() - a.adjoint() * b.adjoint()).norm() < 1e-10);
        }
    }

    #[test]
    fn conjugation_identity() {
        // exp([Ad_T ξ]) == T · exp([ξ]) · T⁻¹
        let mut r = Rng(0x5A5A);
        for _ in 0..200 {
            let big_t = Se3::exp(&r.twist());
            let xi = r.twist();
            let lhs = Se3::exp(&Twist(big_t.adjoint() * xi.0));
            let rhs = big_t * Se3::exp(&xi) * big_t.inverse();
            assert!(se3_diff(&lhs, &rhs) < 1e-9);
        }
    }

    #[test]
    fn revolute_traces_circle() {
        let offset = Se3::from_parts(Vector3::new(1.0, 0.0, 0.0), UnitQuaternion::identity());
        let p = (exp_revolute(&Vector3::z(), PI / 2.0) * offset).translation();
        assert!((p[0]).abs() < 1e-12 && (p[1] - 1.0).abs() < 1e-12);
    }
}
