use anyhow::{Result, bail};
use indexmap::IndexMap;
use num_complex::Complex64;

use crate::models::Number;

pub type CoordinateMap = IndexMap<String, f64>;

pub trait CoordinateSystem: Send + Sync {
    fn name(&self) -> &'static str;
    fn axes(&self) -> &'static [&'static str];
    fn coordinates_from(&self, value: Number) -> Result<CoordinateMap>;
}

pub struct RealLineCoordinates;
pub struct ComplexCartesianCoordinates;
pub struct ComplexPolarCoordinates;
pub struct ProjectionXCoordinates;
pub struct ProjectionYCoordinates;
pub struct RadiusProjectionCoordinates;
pub struct AngleProjectionCoordinates;

impl CoordinateSystem for RealLineCoordinates {
    fn name(&self) -> &'static str {
        "real_line"
    }

    fn axes(&self) -> &'static [&'static str] {
        &["x"]
    }

    fn coordinates_from(&self, value: Number) -> Result<CoordinateMap> {
        if value.im.abs() > 0.0 {
            bail!("real-line coordinates require a real value");
        }
        Ok(IndexMap::from([(String::from("x"), value.re)]))
    }
}

impl CoordinateSystem for ComplexCartesianCoordinates {
    fn name(&self) -> &'static str {
        "complex_cartesian"
    }

    fn axes(&self) -> &'static [&'static str] {
        &["x", "y"]
    }

    fn coordinates_from(&self, value: Number) -> Result<CoordinateMap> {
        let z = Complex64::new(value.re, value.im);
        Ok(IndexMap::from([
            (String::from("x"), z.re),
            (String::from("y"), z.im),
        ]))
    }
}

impl CoordinateSystem for ComplexPolarCoordinates {
    fn name(&self) -> &'static str {
        "complex_polar"
    }

    fn axes(&self) -> &'static [&'static str] {
        &["r", "theta"]
    }

    fn coordinates_from(&self, value: Number) -> Result<CoordinateMap> {
        let z = Complex64::new(value.re, value.im);
        let mut theta = z.im.atan2(z.re);
        if theta < 0.0 {
            theta += std::f64::consts::TAU;
        }
        Ok(IndexMap::from([
            (String::from("r"), z.norm()),
            (String::from("theta"), theta),
        ]))
    }
}

impl CoordinateSystem for ProjectionXCoordinates {
    fn name(&self) -> &'static str {
        "projection_x"
    }

    fn axes(&self) -> &'static [&'static str] {
        &["x"]
    }

    fn coordinates_from(&self, value: Number) -> Result<CoordinateMap> {
        Ok(IndexMap::from([(String::from("x"), value.re)]))
    }
}

impl CoordinateSystem for ProjectionYCoordinates {
    fn name(&self) -> &'static str {
        "projection_y"
    }

    fn axes(&self) -> &'static [&'static str] {
        &["y"]
    }

    fn coordinates_from(&self, value: Number) -> Result<CoordinateMap> {
        Ok(IndexMap::from([(String::from("y"), value.im)]))
    }
}

impl CoordinateSystem for RadiusProjectionCoordinates {
    fn name(&self) -> &'static str {
        "radius_projection"
    }

    fn axes(&self) -> &'static [&'static str] {
        &["r"]
    }

    fn coordinates_from(&self, value: Number) -> Result<CoordinateMap> {
        Ok(IndexMap::from([(String::from("r"), value.norm())]))
    }
}

impl CoordinateSystem for AngleProjectionCoordinates {
    fn name(&self) -> &'static str {
        "angle_projection"
    }

    fn axes(&self) -> &'static [&'static str] {
        &["theta"]
    }

    fn coordinates_from(&self, value: Number) -> Result<CoordinateMap> {
        let mut theta = value.im.atan2(value.re);
        if theta < 0.0 {
            theta += std::f64::consts::TAU;
        }
        Ok(IndexMap::from([(String::from("theta"), theta)]))
    }
}
