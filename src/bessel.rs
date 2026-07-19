//! Special functions needed by the Ephraim-Malah family of gain estimators:
//! modified Bessel functions of the first kind, orders 0 and 1, and the
//! exponential integral E1.
//!
//! These are implemented from scratch (series for small arguments, asymptotic
//! expansions for large arguments) so the project stays dependency-free.

use std::f64::consts::PI;

const EULER_MASCHERONI: f64 = 0.57721566490153286060;

/// Modified Bessel function of the first kind, order 0: `I_0(x)`, `x >= 0`.
pub fn bessel_i0(x: f64) -> f64 {
    let x = x.abs();
    if x < 3.75 {
        // Polynomial expansion (Abramowitz & Stegun 9.8.1 / Numerical Recipes).
        let t = x / 3.75;
        let t2 = t * t;
        1.0 + 3.5156229 * t2
            + 3.0899424 * t2 * t2
            + 1.2067492 * t2 * t2 * t2
            + 0.2659732 * t2 * t2 * t2 * t2
            + 0.0360768 * t2 * t2 * t2 * t2 * t2
            + 0.0045813 * t2 * t2 * t2 * t2 * t2 * t2
    } else {
        // Large-x asymptotic (A&S 9.8.2).
        let t = 3.75 / x;
        let ax = x.exp() / x.sqrt();
        let c = 0.39894228 + 0.01328592 * t + 0.00225319 * t * t - 0.00157565 * t * t * t
            + 0.00916281 * t * t * t * t
            - 0.02057706 * t * t * t * t * t
            + 0.02635537 * t * t * t * t * t * t
            - 0.01647633 * t * t * t * t * t * t * t
            + 0.00392377 * t * t * t * t * t * t * t * t;
        ax * c
    }
}

/// Modified Bessel function of the first kind, order 1: `I_1(x)`, `x >= 0`.
pub fn bessel_i1(x: f64) -> f64 {
    let x = x.abs();
    if x < 3.75 {
        let t = x / 3.75;
        let t2 = t * t;
        x * (0.5
            + 0.87890594 * t2
            + 0.51498869 * t2 * t2
            + 0.15084934 * t2 * t2 * t2
            + 0.02658733 * t2 * t2 * t2 * t2
            + 0.00301532 * t2 * t2 * t2 * t2 * t2
            + 0.00032411 * t2 * t2 * t2 * t2 * t2 * t2)
    } else {
        let t = 3.75 / x;
        let ax = x.exp() / x.sqrt();
        let c = 0.39894228 - 0.03988024 * t - 0.00362018 * t * t + 0.00163801 * t * t * t
            - 0.01031555 * t * t * t * t
            + 0.02282967 * t * t * t * t * t
            - 0.02895312 * t * t * t * t * t * t
            + 0.01787654 * t * t * t * t * t * t * t
            - 0.00420059 * t * t * t * t * t * t * t * t;
        ax * c
    }
}

/// Exponentially-scaled modified Bessel `I_0(x) * e^{-x}`, numerically stable
/// for all `x >= 0` (no overflow/underflow). Used by MMSE-STSA.
pub fn bessel_i0_scaled(x: f64) -> f64 {
    let x = x.abs();
    if x < 50.0 {
        // Both factors are finite here; direct product is exact.
        bessel_i0(x) * (-x).exp()
    } else {
        // Asymptotic: I0(x) e^{-x} ~ 1/sqrt(2*pi*x) * (1 + 1/(8x) + 9/(128 x^2)).
        let s = 1.0 / (2.0 * PI * x).sqrt();
        s * (1.0 + 1.0 / (8.0 * x) + 9.0 / (128.0 * x * x))
    }
}

/// Exponentially-scaled modified Bessel `I_1(x) * e^{-x}`, numerically stable
/// for all `x >= 0`. Used by MMSE-STSA.
pub fn bessel_i1_scaled(x: f64) -> f64 {
    let x = x.abs();
    if x < 50.0 {
        bessel_i1(x) * (-x).exp()
    } else {
        // Asymptotic: I1(x) e^{-x} ~ 1/sqrt(2*pi*x) * (1 - 3/(8x) - 15/(128 x^2)).
        let s = 1.0 / (2.0 * PI * x).sqrt();
        s * (1.0 - 3.0 / (8.0 * x) - 15.0 / (128.0 * x * x))
    }
}

/// Exponential integral `E1(x) = integral_x^inf e^{-t}/t dt` for `x > 0`.
///
/// Uses the series expansion for small `x` and the continued-fraction /
/// asymptotic expansion for large `x` (Numerical Recipes, `expint`).
pub fn exp_int_e1(x: f64) -> f64 {
    debug_assert!(x > 0.0, "E1 requires x > 0");
    if x <= 1.0 {
        // Series: E1(x) = -gamma - ln(x) - sum_{k>=1} (-x)^k / (k * k!)
        // with a_k = (-x)^k / k!  and  term_k = a_k / k.
        let mut sum = 0.0;
        let mut a = -x; // a_1 = (-x)^1 / 1!
        let mut k = 1.0;
        loop {
            let term = a / k;
            sum += term;
            if term.abs() < 1e-16 * sum.abs().max(1.0) {
                break;
            }
            a = a * (-x) / (k + 1.0); // a_{k+1} = (-x)^{k+1} / (k+1)!
            k += 1.0;
            if k > 1000.0 {
                break;
            }
        }
        -EULER_MASCHERONI - x.ln() - sum
    } else {
        // Continued fraction (Lentz) for E1(x) = e^{-x} * CF.
        let tiny = 1e-30;
        let mut b = x + 1.0;
        let mut c = 1.0 / tiny;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..=1000 {
            let an = -(i as f64) * (i as f64);
            b += 2.0;
            d = an * d + b;
            if d.abs() < tiny {
                d = tiny;
            }
            c = b + an / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-16 {
                break;
            }
        }
        h * (-x).exp()
    }
}

/// `sqrt(2 / pi)` precomputed for the MMSE-STSA estimator.
pub const SQRT_2_OVER_PI: f64 = 0.7978845608028654;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i0_known_values() {
        assert!((bessel_i0(0.0) - 1.0).abs() < 1e-9);
        // I_0(1) ≈ 1.2660658777520082
        assert!((bessel_i0(1.0) - 1.2660658777520082).abs() < 1e-6);
        // I_0(5) ≈ 27.23987182361646
        assert!((bessel_i0(5.0) - 27.23987182361646).abs() < 1e-4);
        // I_0(20) ≈ 4.3566e7  (verified via the I_0 power series; A&S 9.8.2 error < 1e-7)
        let v = bessel_i0(20.0);
        assert!((v.ln() - 17.590).abs() < 0.01, "ln(I0(20))={}", v.ln());
    }

    #[test]
    fn i1_known_values() {
        assert!(bessel_i1(0.0).abs() < 1e-9);
        // I_1(1) ≈ 0.565159103992485
        assert!((bessel_i1(1.0) - 0.565159103992485).abs() < 1e-6);
        // I_1(5) ≈ 24.33564214245033
        assert!((bessel_i1(5.0) - 24.33564214245033).abs() < 1e-4);
    }

    #[test]
    fn e1_known_values() {
        // E1(0.1) ≈ 1.822923958449177
        assert!((exp_int_e1(0.1) - 1.822923958449177).abs() < 1e-7);
        // E1(1.0) ≈ 0.2193839343955203
        assert!((exp_int_e1(1.0) - 0.2193839343955203).abs() < 1e-7);
        // E1(5.0) ≈ 0.0011482955912753457
        assert!((exp_int_e1(5.0) - 0.0011482955912753457).abs() < 1e-9);
    }

    #[test]
    fn scaled_bessels_stable_and_accurate() {
        // Small x: I0(x) e^{-x} and I1(x) e^{-x} must match direct product.
        for &x in &[0.0, 0.5, 1.0, 5.0, 10.0, 49.0] {
            let direct0 = bessel_i0(x) * (-x).exp();
            let direct1 = bessel_i1(x) * (-x).exp();
            assert!((bessel_i0_scaled(x) - direct0).abs() < 1e-9 * direct0.abs().max(1.0));
            assert!((bessel_i1_scaled(x) - direct1).abs() < 1e-9 * direct1.abs().max(1.0));
        }
        // Large x: must be finite (no overflow) and match the asymptotic ~1/sqrt(2pi x).
        for &x in &[500.0, 5000.0, 1e6] {
            let v0 = bessel_i0_scaled(x);
            let v1 = bessel_i1_scaled(x);
            assert!(v0.is_finite() && v0 > 0.0, "i0_scaled({x})={v0}");
            assert!(v1.is_finite() && v1 > 0.0, "i1_scaled({x})={v1}");
            let asymp = 1.0 / (2.0 * PI * x).sqrt();
            assert!((v0 - asymp).abs() / asymp < 0.01, "i0_scaled({x}) off");
        }
        // I0_scaled(0) = 1, I1_scaled(0) = 0.
        assert!((bessel_i0_scaled(0.0) - 1.0).abs() < 1e-12);
        assert!(bessel_i1_scaled(0.0).abs() < 1e-12);
    }
}
