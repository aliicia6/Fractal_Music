use anyhow::{Result, bail};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ContinuousRange {
    pub minimum: f64,
    pub maximum: f64,
}

impl ContinuousRange {
    pub fn new(minimum: f64, maximum: f64) -> Result<Self> {
        if maximum <= minimum {
            bail!("maximum must be greater than minimum");
        }
        Ok(Self { minimum, maximum })
    }

    pub fn length(&self) -> f64 {
        self.maximum - self.minimum
    }

    pub fn contains(&self, value: f64) -> bool {
        self.minimum <= value && value <= self.maximum
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct UniformSubdivision {
    pub range: ContinuousRange,
    pub steps: usize,
}

impl UniformSubdivision {
    pub fn new(range: ContinuousRange, steps: usize) -> Result<Self> {
        if steps < 1 {
            bail!("steps must be at least 1");
        }
        Ok(Self { range, steps })
    }

    pub fn step_size(&self) -> f64 {
        self.range.length() / self.steps as f64
    }

    pub fn edges(&self) -> Vec<f64> {
        (0..=self.steps)
            .map(|index| self.range.minimum + index as f64 * self.step_size())
            .collect()
    }

    pub fn index_of(&self, value: f64) -> Option<usize> {
        if !self.range.contains(value) {
            return None;
        }
        if value == self.range.maximum {
            return Some(self.steps - 1);
        }
        Some(((value - self.range.minimum) / self.step_size()) as usize)
    }

    pub fn center_of(&self, index: usize) -> Result<f64> {
        if index >= self.steps {
            bail!("subdivision index out of range");
        }
        Ok(self.range.minimum + (index as f64 + 0.5) * self.step_size())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiscretizationGrid {
    pub subdivisions: IndexMap<String, UniformSubdivision>,
}

impl DiscretizationGrid {
    pub fn new(subdivisions: IndexMap<String, UniformSubdivision>) -> Result<Self> {
        if subdivisions.is_empty() {
            bail!("grid requires at least one subdivision");
        }
        Ok(Self { subdivisions })
    }

    pub fn axes(&self) -> Vec<String> {
        self.subdivisions.keys().cloned().collect()
    }

    pub fn indices_of(&self, coordinates: &IndexMap<String, f64>) -> Result<Option<IndexMap<String, usize>>> {
        let mut indices = IndexMap::new();
        for (axis, subdivision) in &self.subdivisions {
            let value = coordinates
                .get(axis)
                .ok_or_else(|| anyhow::anyhow!("missing coordinate axis: {axis}"))?;
            let index = match subdivision.index_of(*value) {
                Some(index) => index,
                None => return Ok(None),
            };
            indices.insert(axis.clone(), index);
        }
        Ok(Some(indices))
    }

    pub fn centers_of(&self, indices: &IndexMap<String, usize>) -> Result<IndexMap<String, f64>> {
        let mut centers = IndexMap::new();
        for (axis, subdivision) in &self.subdivisions {
            let index = indices
                .get(axis)
                .ok_or_else(|| anyhow::anyhow!("missing index for axis: {axis}"))?;
            centers.insert(axis.clone(), subdivision.center_of(*index)?);
        }
        Ok(centers)
    }

    pub fn cell_id(&self, indices: &IndexMap<String, usize>) -> Result<usize> {
        let mut cell_id = 0usize;
        let mut multiplier = 1usize;
        for (axis, subdivision) in self.subdivisions.iter().rev() {
            let index = indices
                .get(axis)
                .ok_or_else(|| anyhow::anyhow!("missing index for axis: {axis}"))?;
            cell_id += *index * multiplier;
            multiplier *= subdivision.steps;
        }
        Ok(cell_id)
    }
}
