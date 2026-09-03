use std::collections::BTreeMap;

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

use crate::transformations::{DiscreteAnalysisResult, DiscretePoint};

pub const MUSIC_CRITERIA: [&str; 4] = ["melody", "harmony", "rhythm", "texture"];
pub const MUSIC_SEGMENTATIONS: [&str; 3] = ["bands", "positions", "sectors"];
pub const SCALES: [(&str, &[i32]); 4] = [
    ("major", &[0, 2, 4, 5, 7, 9, 11]),
    ("minor", &[0, 2, 3, 5, 7, 8, 10]),
    ("pentatonic", &[0, 3, 5, 7, 10]),
    ("chromatic", &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MusicMappingConfig {
    pub criterion: String,
    pub segmentation: String,
    pub scale: String,
    pub root: String,
    pub octaves: usize,
}

impl Default for MusicMappingConfig {
    fn default() -> Self {
        Self {
            criterion: String::from("melody"),
            segmentation: String::from("bands"),
            scale: String::from("minor"),
            root: String::from("C4"),
            octaves: 2,
        }
    }
}

impl MusicMappingConfig {
    pub fn validate(&self) -> Result<()> {
        if !MUSIC_CRITERIA.contains(&self.criterion.as_str()) {
            bail!("Unknown music criterion: {}", self.criterion);
        }
        if !MUSIC_SEGMENTATIONS.contains(&self.segmentation.as_str()) {
            bail!("Unknown segmentation: {}", self.segmentation);
        }
        if !SCALES.iter().any(|(name, _)| *name == self.scale) {
            bail!("Unknown scale: {}", self.scale);
        }
        if self.octaves < 1 {
            bail!("octaves must be at least 1");
        }
        root_midi(&self.root)?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MusicalAttractor {
    pub case_label: String,
    pub attractor_index: usize,
    pub period: usize,
    pub notes: Vec<String>,
    pub cells: Vec<Option<usize>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StrudelPattern {
    pub config: MusicMappingConfig,
    pub attractors: Vec<MusicalAttractor>,
    pub code: String,
}

impl StrudelPattern {
    pub fn description(&self) -> String {
        format!(
            "{}, {}, {}:{}, atractores={}",
            self.config.criterion,
            self.config.segmentation,
            self.config.root,
            self.config.scale,
            self.attractors.len()
        )
    }
}

pub fn generate_strudel_pattern(
    results: &[(&str, &DiscreteAnalysisResult)],
    config: Option<MusicMappingConfig>,
) -> Result<StrudelPattern> {
    let config = config.unwrap_or_default();
    config.validate()?;

    let mut attractors = Vec::new();
    for (case_label, result) in results {
        for (attractor_index, discrete_attractor) in result.attractors.iter().enumerate() {
            let mut notes = Vec::new();
            let mut cells = Vec::new();
            for point in &discrete_attractor.points {
                if let Some(position) = normalized_position(point, result, &config.segmentation) {
                    notes.push(note_from_position(position, &config)?);
                    cells.push(point.cell_id);
                }
            }
            if !notes.is_empty() {
                attractors.push(MusicalAttractor {
                    case_label: (*case_label).to_string(),
                    attractor_index: attractor_index + 1,
                    period: discrete_attractor.attractor.period,
                    notes,
                    cells,
                });
            }
        }
    }

    let code = strudel_code(&attractors, &config);
    Ok(StrudelPattern {
        config,
        attractors,
        code,
    })
}

fn normalized_position(
    point: &DiscretePoint,
    result: &DiscreteAnalysisResult,
    segmentation: &str,
) -> Option<f64> {
    if segmentation == "sectors" {
        if let Some(indices) = &point.indices
            && let Some(theta_index) = indices.get("theta")
        {
            let steps = result.transformer.grid.subdivisions.get("theta")?.steps;
            return Some(index_fraction(*theta_index, steps));
        }
        let value = point.value;
        return Some((value.im.atan2(value.re).rem_euclid(std::f64::consts::TAU)) / std::f64::consts::TAU);
    }

    let indices = point.indices.as_ref()?;
    if segmentation == "positions" {
        let total_cells = result
            .transformer
            .grid
            .subdivisions
            .values()
            .fold(1usize, |acc, subdivision| acc * subdivision.steps);
        let cell_id = point.cell_id?;
        return Some(index_fraction(cell_id, total_cells));
    }

    let axes = result
        .transformer
        .grid
        .subdivisions
        .keys()
        .map(String::as_str)
        .collect::<Vec<_>>();
    let axis = band_axis(&axes);
    let index = *indices.get(axis)?;
    let steps = result.transformer.grid.subdivisions.get(axis)?.steps;
    Some(index_fraction(index, steps))
}

fn band_axis<'a>(axes: &'a [&'a str]) -> &'a str {
    for axis in ["y", "r", "x", "theta"] {
        if axes.contains(&axis) {
            return axis;
        }
    }
    axes[0]
}

fn index_fraction(index: usize, steps: usize) -> f64 {
    if steps == 0 {
        return 0.0;
    }
    let value = (index as f64 + 0.5) / steps as f64;
    value.clamp(0.0, 1.0 - 1e-12)
}

fn note_from_position(position: f64, config: &MusicMappingConfig) -> Result<String> {
    let scales = scale_map();
    let scale = scales
        .get(config.scale.as_str())
        .ok_or_else(|| anyhow::anyhow!("Unknown scale: {}", config.scale))?;
    let degrees_per_octave = scale.len();
    let total_degrees = degrees_per_octave * config.octaves;
    let degree = ((position * total_degrees as f64) as usize).min(total_degrees - 1);
    let octave_offset = degree / degrees_per_octave;
    let scale_degree = degree % degrees_per_octave;
    let midi = root_midi(&config.root)? + scale[scale_degree] + 12 * octave_offset as i32;
    let pitch_classes = ["c", "c#", "d", "d#", "e", "f", "f#", "g", "g#", "a", "a#", "b"];
    let pitch_class = pitch_classes[(midi.rem_euclid(12)) as usize];
    let octave = midi / 12 - 1;
    Ok(format!("{pitch_class}{octave}"))
}

fn root_midi(root: &str) -> Result<i32> {
    let root = root.trim();
    if root.len() < 2 {
        bail!("root must use a note such as C4, D#3 or Bb2");
    }
    let mut chars = root.chars();
    let first = chars.next().unwrap().to_ascii_lowercase();
    let mut name = String::from(first);
    // Optional accidental (# or b) right after the letter.
    if let Some(next) = root.chars().nth(1)
        && (next == '#' || next == 'b')
    {
        name.push(next.to_ascii_lowercase());
    }
    let octave_str = &root[name.len()..];
    let octave: i32 = octave_str
        .trim()
        .parse()
        .map_err(|_| anyhow::anyhow!("root must use a note such as C4, D#3 or Bb2"))?;
    let offsets = BTreeMap::from([
        ("c", 0),
        ("c#", 1),
        ("db", 1),
        ("d", 2),
        ("d#", 3),
        ("eb", 3),
        ("e", 4),
        ("f", 5),
        ("f#", 6),
        ("gb", 6),
        ("g", 7),
        ("g#", 8),
        ("ab", 8),
        ("a", 9),
        ("a#", 10),
        ("bb", 10),
        ("b", 11),
    ]);
    let offset = offsets
        .get(name.as_str())
        .ok_or_else(|| anyhow::anyhow!("root note is not valid"))?;
    Ok((octave + 1) * 12 + *offset)
}

fn scale_map() -> BTreeMap<&'static str, &'static [i32]> {
    BTreeMap::from(SCALES)
}

fn strudel_code(attractors: &[MusicalAttractor], config: &MusicMappingConfig) -> String {
    if attractors.is_empty() {
        return String::from("silence");
    }
    let groups: Vec<String> = attractors.iter().map(|a| a.notes.join(" ")).collect();
    let melody = groups.join(" ");
    match config.criterion.as_str() {
        "melody" => format!("note(\"{melody}\").sound(\"piano\").slow(2)"),
        "harmony" => {
            let chords = attractors
                .iter()
                .map(|a| format!("[{}]", a.notes.join(",")))
                .collect::<Vec<_>>()
                .join(" ");
            format!("note(\"{chords}\").sound(\"gm_epiano1\").slow(2)")
        }
        "rhythm" => {
            let total = melody.split_whitespace().count();
            let rhythm = (0..total)
                .map(|index| if index % 2 == 0 { "x" } else { "~" })
                .collect::<Vec<_>>()
                .join(" ");
            format!(
                "stack(note(\"{melody}\").sound(\"gm_acoustic_bass\"), s(\"bd\").struct(\"{rhythm}\"))"
            )
        }
        _ => format!(
            "stack(note(\"{melody}\").sound(\"piano\"), note(\"{melody}\").sound(\"sawtooth\").slow(2).gain(0.35))"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_midi_handles_natural_notes() {
        // C4 = MIDI 60
        assert_eq!(root_midi("C4").unwrap(), 60);
// A4 = MIDI 69
        assert_eq!(root_midi("A4").unwrap(), 69);
        // Bb2 = MIDI 46
        assert_eq!(root_midi("Bb2").unwrap(), 46);
    }

    #[test]
    fn root_midi_handles_sharps_and_flats() {
        // D#3 = MIDI 51
        assert_eq!(root_midi("D#3").unwrap(), 51);
        // Eb3 = MIDI 51
        assert_eq!(root_midi("Eb3").unwrap(), 51);
        // F#4 = MIDI 66
        assert_eq!(root_midi("F#4").unwrap(), 66);
        // Ab4 = MIDI 68
        assert_eq!(root_midi("Ab4").unwrap(), 68);
    }

    #[test]
    fn root_midi_handles_lowercase() {
        assert_eq!(root_midi("c4").unwrap(), 60);
        assert_eq!(root_midi("d#3").unwrap(), 51);
    }

    #[test]
    fn root_midi_rejects_invalid_input() {
        assert!(root_midi("").is_err());
        assert!(root_midi("C").is_err());
        assert!(root_midi("H4").is_err());
        assert!(root_midi("C#").is_err());
    }
}
