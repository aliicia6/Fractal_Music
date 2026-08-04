use std::sync::Arc;

use num_complex::Complex64;

use crate::recurrence::Recurrence;

pub fn logistic_map(r: f64, x0: f64) -> Recurrence {
    Recurrence::new(
        format!("logistic_map(r={r}, x0={x0})"),
        Complex64::new(x0, 0.0),
        Arc::new(move |x, _n| {
            let value = r * x.re * (1.0 - x.re);
            Complex64::new(value, 0.0)
        }),
    )
}

pub fn quadratic_complex_map(c: Complex64, z0: Complex64) -> Recurrence {
    Recurrence::new(
        format!("quadratic_complex_map(c={c}, z0={z0})"),
        z0,
        Arc::new(move |z, _n| z * z + c),
    )
}

pub fn cosine_fixed_point(x0: f64) -> Recurrence {
    Recurrence::new(
        format!("cosine_fixed_point(x0={x0})"),
        Complex64::new(x0, 0.0),
        Arc::new(move |x, _n| Complex64::new(x.re.cos(), 0.0)),
    )
}
