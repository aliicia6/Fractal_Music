use num_complex::Complex64;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QuadraticParameters {
    pub c: Complex64,
    pub z0: Complex64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LogisticParameters {
    pub r: f64,
    pub x0: f64,
}

pub fn logistic_to_quadratic(r: f64, x0: f64) -> QuadraticParameters {
    QuadraticParameters {
        c: Complex64::new(r * (2.0 - r) / 4.0, 0.0),
        z0: Complex64::new(r * (0.5 - x0), 0.0),
    }
}

pub fn quadratic_to_logistic(
    c: Complex64,
    z0: Complex64,
    tolerance: f64,
) -> Option<LogisticParameters> {
    if c.im.abs() > tolerance || z0.im.abs() > tolerance {
        return None;
    }

    let discriminant = 1.0 - 4.0 * c.re;
    if discriminant < -tolerance {
        return None;
    }

    let r_value = 1.0 + discriminant.max(0.0).sqrt();
    if r_value.abs() <= tolerance {
        return None;
    }

    Some(LogisticParameters {
        r: r_value,
        x0: 0.5 - z0.re / r_value,
    })
}
