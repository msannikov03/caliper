//! Jerk-limited 1-DOF 7-segment S-curve, all branches closed-form.

/// One per-segment cached boundary: cumulative time at its START + state there.
#[derive(Clone, Copy, Debug)]
struct Seg {
    t0: f64,
    p0: f64,
    v0: f64,
    a0: f64,
    jerk: f64,
}

/// Jerk-limited 7-segment profile for a 1-DOF rest-to-rest move of |displacement|
/// `d` >= 0, direction carried by `sign`. Stores 8 boundary times so sample() is O(1).
#[derive(Clone, Debug)]
pub struct ScurveProfile {
    pub d: f64,
    pub sign: f64,
    pub tj: f64, // jerk-ramp duration (segs 1,3,5,7)
    pub ta: f64, // const-accel duration (segs 2,6)
    pub tv: f64, // cruise duration (seg 4)
    pub vpeak: f64,
    pub apeak: f64,
    pub jmax: f64,
    segs: [Seg; 7],
    bnd: [f64; 8], // cumulative boundary times t0..t7
}

impl ScurveProfile {
    pub fn total(&self) -> f64 {
        4.0 * self.tj + 2.0 * self.ta + self.tv
    }

    /// Zero-length rest profile (used for d≈0 dofs).
    fn rest() -> Self {
        let z = Seg {
            t0: 0.0,
            p0: 0.0,
            v0: 0.0,
            a0: 0.0,
            jerk: 0.0,
        };
        ScurveProfile {
            d: 0.0,
            sign: 1.0,
            tj: 0.0,
            ta: 0.0,
            tv: 0.0,
            vpeak: 0.0,
            apeak: 0.0,
            jmax: 1.0,
            segs: [z; 7],
            bnd: [0.0; 8],
        }
    }

    /// (signed displacement from start, velocity, acceleration) at absolute time t.
    pub fn sample(&self, t: f64) -> (f64, f64, f64) {
        let tt = t.clamp(0.0, self.total());
        // locate segment: largest k with bnd[k] <= tt (≤7 boundaries, linear scan)
        let mut k = 0usize;
        while k < 6 && tt >= self.bnd[k + 1] {
            k += 1;
        }
        let s = &self.segs[k];
        let dt = tt - s.t0;
        let a = s.a0 + s.jerk * dt;
        let v = s.v0 + s.a0 * dt + 0.5 * s.jerk * dt * dt;
        let p = s.p0 + s.v0 * dt + 0.5 * s.a0 * dt * dt + s.jerk * dt * dt * dt / 6.0;
        (self.sign * p, self.sign * v, self.sign * a)
    }

    /// Build the segment cache from (tj,ta,tv,jmax,sign,d).
    fn build(d: f64, sign: f64, tj: f64, ta: f64, tv: f64, jmax: f64) -> Self {
        // jerk signs per segment: +,0,-,0,-,0,+
        let js = [jmax, 0.0, -jmax, 0.0, -jmax, 0.0, jmax];
        let durs = [tj, ta, tj, tv, tj, ta, tj];
        let mut segs = [Seg {
            t0: 0.0,
            p0: 0.0,
            v0: 0.0,
            a0: 0.0,
            jerk: 0.0,
        }; 7];
        let mut bnd = [0.0f64; 8];
        let (mut t0, mut p0, mut v0, mut a0) = (0.0, 0.0, 0.0, 0.0);
        for i in 0..7 {
            segs[i] = Seg {
                t0,
                p0,
                v0,
                a0,
                jerk: js[i],
            };
            let dt = durs[i];
            let a1 = a0 + js[i] * dt;
            let v1 = v0 + a0 * dt + 0.5 * js[i] * dt * dt;
            let p1 = p0 + v0 * dt + 0.5 * a0 * dt * dt + js[i] * dt * dt * dt / 6.0;
            t0 += dt;
            p0 = p1;
            v0 = v1;
            a0 = a1;
            bnd[i + 1] = t0;
        }
        let vpeak = segs.iter().map(|s| s.v0).fold(0.0f64, f64::max);
        // peak |accel|: jmax*tj in both the plateau (= amax) and non-plateau cases.
        let apeak = jmax * tj;
        ScurveProfile {
            d,
            sign,
            tj,
            ta,
            tv,
            vpeak,
            apeak,
            jmax,
            segs,
            bnd,
        }
    }
}

const EPS_D: f64 = 1e-12;

/// Plan a rest-to-rest jerk-limited profile for signed displacement `d_signed`.
/// Closed-form branch selection (canonical / no-cruise / no-plateau / triangle).
pub fn plan_scurve(d_signed: f64, vmax: f64, amax: f64, jmax: f64) -> ScurveProfile {
    let d = d_signed.abs();
    let sign = if d_signed < 0.0 { -1.0 } else { 1.0 };
    if d < EPS_D {
        return ScurveProfile::rest();
    }

    let tj_full = amax / jmax; // time to ramp accel 0->amax
    // Does accel reach amax? It does iff vmax >= amax*tj_full (= amax²/jmax).
    let reaches_amax = vmax >= amax * tj_full;

    let (tj, ta_acc, vpk) = if reaches_amax {
        // accel plateaus; cruise vel candidate = vmax
        let ta = vmax / amax - tj_full; // >= 0 here
        (tj_full, ta, vmax)
    } else {
        // accel never plateaus: ta=0, peak accel < amax. Cruise vel candidate=vmax.
        // with ta=0, v gained over accel phase = jmax*tj², so tj=sqrt(vmax/jmax).
        ((vmax / jmax).sqrt(), 0.0, vmax)
    };

    // distance for accel+decel reaching vpk = vpk*(2*tj+ta_acc).
    let d_reach = vpk * (2.0 * tj + ta_acc);

    if d >= d_reach {
        // cruise exists
        let tv = (d - d_reach) / vpk;
        return ScurveProfile::build(d, sign, tj, ta_acc, tv, jmax);
    }

    // No cruise (tv=0): vpeak < vmax. Re-derive under d.
    if reaches_amax {
        // accel still plateaus iff resulting ta>=0.
        // Quadratic in ta: amax*ta² + 3*amax*tj*ta + (2*amax*tj² - d)=0.
        let a = amax;
        let b = 3.0 * amax * tj;
        let c = 2.0 * amax * tj * tj - d;
        let disc = (b * b - 4.0 * a * c).max(0.0);
        let ta = (-b + disc.sqrt()) / (2.0 * a);
        if ta >= 0.0 {
            return ScurveProfile::build(d, sign, tj, ta, 0.0, jmax);
        }
        // ta<0 -> accel does NOT plateau either: fall through to no-plateau triangle.
    }
    // No plateau, no cruise: ta=0, tv=0, d = 2*jmax*tj³ -> tj=cbrt(d/(2*jmax)).
    let tj = (d / (2.0 * jmax)).cbrt();
    ScurveProfile::build(d, sign, tj, 0.0, 0.0, jmax)
}

/// Re-plan so `total()==target_t` WITHOUT exceeding (vmax,amax,jmax), by bisecting
/// a single uniform scale s∈(0,1] applied to all three limits. total() is strictly
/// monotone-decreasing in s, so the bracket is well-posed; s<1 forces every scaled
/// peak strictly under the originals (provably within-limits). ~60 iters → ~1e-12 s.
pub fn plan_scurve_to_duration(
    d_signed: f64,
    target_t: f64,
    vmax: f64,
    amax: f64,
    jmax: f64,
) -> ScurveProfile {
    if d_signed.abs() < EPS_D {
        return ScurveProfile::rest();
    }
    let full = plan_scurve(d_signed, vmax, amax, jmax);
    if full.total() >= target_t {
        return full; // already at/over target; can't go slower w/ s>1
    }
    let (mut lo, mut hi) = (1e-9, 1.0); // s
    for _ in 0..60 {
        let s = 0.5 * (lo + hi);
        let t = plan_scurve(d_signed, s * vmax, s * amax, s * jmax).total();
        if t > target_t {
            lo = s;
        } else {
            hi = s;
        }
    }
    plan_scurve(d_signed, lo * vmax, lo * amax, lo * jmax)
}
