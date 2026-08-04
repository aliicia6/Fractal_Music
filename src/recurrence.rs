use std::sync::Arc;

use crate::models::Number;

pub type StepFunction = Arc<dyn Fn(Number, usize) -> Number + Send + Sync>;

#[derive(Clone)]
pub struct Recurrence {
    pub name: String,
    pub initial_value: Number,
    pub step: StepFunction,
}

impl Recurrence {
    pub fn new(name: impl Into<String>, initial_value: Number, step: StepFunction) -> Self {
        Self {
            name: name.into(),
            initial_value,
            step,
        }
    }

    pub fn next_value(&self, current_value: Number, index: usize) -> Number {
        (self.step)(current_value, index)
    }
}
