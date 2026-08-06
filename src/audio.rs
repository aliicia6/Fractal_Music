use anyhow::{Result, bail};
use rodio::{OutputStream, OutputStreamBuilder, Sink};
use rodio::buffer::SamplesBuffer;

/// Reproductor local de una parte práctica de la notación de Strudel.
/// No necesita red ni procesos externos: el audio se sintetiza con rodio.
pub struct AudioEngine {
    _stream: OutputStream,
    sink: Sink,
}

impl AudioEngine {
    pub fn new() -> Result<Self> {
        let stream = OutputStreamBuilder::open_default_stream()
            .map_err(|error| anyhow::anyhow!("no se pudo abrir el dispositivo de audio: {error}"))?;
        let sink = Sink::connect_new(stream.mixer());
        Ok(Self { _stream: stream, sink })
    }

    pub fn play(&mut self, code: &str) -> Result<()> {
        let events = parse_strudel_events(code);
        if events.is_empty() {
            bail!("no se encontraron notas o sonidos reproducibles");
        }
        self.sink.stop();
        self.sink = Sink::connect_new(self._stream.mixer());
        let sample_rate = 44_100_u32;
        let mut samples = Vec::new();
        for (frequencies, duration) in events {
            let frame_count = (duration * sample_rate as f32) as usize;
            for frame in 0..frame_count {
                let time = frame as f32 / sample_rate as f32;
                let amplitude = 0.16 / frequencies.len().max(1) as f32;
                let value = frequencies
                    .iter()
                    .map(|frequency| (std::f32::consts::TAU * frequency * time).sin())
                    .sum::<f32>()
                    * amplitude;
                samples.extend([value, value]);
            }
        }
        self.sink.append(SamplesBuffer::new(2, sample_rate, samples));
        Ok(())
    }

    pub fn stop(&self) {
        self.sink.stop();
    }
}

fn parse_strudel_events(code: &str) -> Vec<(Vec<f32>, f32)> {
    let mut events = Vec::new();
    for (function, duration) in [("note", 0.24_f32), ("sound", 0.16_f32), ("s", 0.16_f32)] {
        let mut offset = 0;
        while let Some(relative) = code[offset..].find(&format!("{function}(")) {
            let start = offset + relative + function.len() + 1;
            let Some(end) = code[start..].find(')') else { break };
            let content = &code[start..start + end];
            let content = content.trim_matches(|character| character == '"' || character == '\'');
            for token in content.split_whitespace().filter(|token| !token.is_empty()) {
                let token = token.trim_matches(|character| character == '<' || character == '>');
                let frequencies = if token.starts_with('[') && token.ends_with(']') {
                    token[1..token.len() - 1]
                        .split(|character: char| character == ',' || character.is_whitespace())
                        .filter_map(parse_token)
                        .collect::<Vec<_>>()
                } else {
                    parse_token(token).into_iter().collect()
                };
                if !frequencies.is_empty() {
                    events.push((frequencies, duration));
                }
            }
            offset = start + end + 1;
        }
    }
    events
}

pub fn describe_strudel(code: &str) -> Vec<String> {
    let mut descriptions = Vec::new();
    for function in ["note", "sound", "s"] {
        let mut offset = 0;
        while let Some(relative) = code[offset..].find(&format!("{function}(")) {
            let start = offset + relative + function.len() + 1;
            let Some(end) = code[start..].find(')') else { break };
            let content = code[start..start + end]
                .trim_matches(|character| character == '"' || character == '\'');
            if function == "note" && content.contains('[') {
                descriptions.push(format!(
                    "Acorde {content}: las notas entre corchetes se tocan simultáneamente."
                ));
            } else if function == "note" {
                descriptions.push(format!(
                    "Notas {content}: se tocan en secuencia, de izquierda a derecha."
                ));
            } else {
                descriptions.push(format!(
                    "Sonido {content}: se dispara como evento rítmico."
                ));
            }
            offset = start + end + 1;
        }
    }
    descriptions
}

fn parse_token(token: &str) -> Option<f32> {
    match token.to_ascii_lowercase().as_str() {
        "bd" | "bass" => return Some(90.0),
        "sd" | "snare" => return Some(180.0),
        "hh" | "hihat" => return Some(7000.0),
        "cp" | "clap" => return Some(420.0),
        "silence" | "~" => return None,
        _ => {}
    }
    let bytes = token.as_bytes();
    let base = match bytes.first()?.to_ascii_lowercase() {
        b'c' => 0,
        b'd' => 2,
        b'e' => 4,
        b'f' => 5,
        b'g' => 7,
        b'a' => 9,
        b'b' => 11,
        _ => return None,
    };
    let mut index = 1;
    let accidental = match bytes.get(index).copied() {
        Some(b'#') => { index += 1; 1 }
        Some(b'b') => { index += 1; -1 }
        _ => 0,
    };
    let octave = token[index..].parse::<i32>().ok()?;
    let midi = 12 * (octave + 1) + base + accidental;
    Some(440.0 * 2.0_f32.powf((midi as f32 - 69.0) / 12.0))
}
