use std::collections::{BTreeMap, HashSet};

use anyhow::{Result, bail};
use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotBounds, PlotPoint, Points, VLine};
use num_complex::Complex64;

use crate::{
    AnalysisConfig, AxisRangeConfig, BehaviorKind, DiscretizationConfig, DiscreteAnalysisResult,
    ExplorationCase, ExplorerConfig, MusicMappingConfig, RecurrenceConfig, build_session_data,
    default_concept_config, generate_strudel_pattern, logistic_to_quadratic, parse_session_data,
    quadratic_to_logistic, run_exploration, write_results_csv, write_session,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SequenceChoice {
    Cosine,
    Logistic,
    Quadratic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CoordinateChoice {
    X,
    Cartesian,
    Polar,
    Y,
    Radius,
    Angle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TableMode {
    Todo,
    Sucesion,
    Atractores,
}

#[derive(Clone)]
struct CaseRun {
    case: ExplorationCase,
    result: DiscreteAnalysisResult,
    is_active: bool,
}

pub struct VisualExplorerApp {
    sequence: SequenceChoice,
    coordinates: CoordinateChoice,
    table_mode: TableMode,
    max_terms: String,
    tolerance: String,
    max_period: String,
    transient_terms: String,
    table_rows: String,
    x0: String,
    r: String,
    c_real: String,
    c_imag: String,
    z0_real: String,
    z0_imag: String,
    x_range: String,
    y_range: String,
    r_range: String,
    theta_range: String,
    cases: Vec<ExplorationCase>,
    active_case_id: Option<String>,
    case_counter: usize,
    runs: Vec<CaseRun>,
    message: String,
    dirty: bool,
    music_criterion: String,
    music_segmentation: String,
    music_scale: String,
    strudel_code: String,
    bifurcation_points: Vec<[f64; 2]>,
    mandelbrot_points: Vec<[f64; 2]>,
}

impl Default for VisualExplorerApp {
    fn default() -> Self {
        let config = default_concept_config();
        let mut app = Self {
            sequence: SequenceChoice::Logistic,
            coordinates: CoordinateChoice::Cartesian,
            table_mode: TableMode::Todo,
            max_terms: String::from("1000"),
            tolerance: String::from("1e-7"),
            max_period: String::from("16"),
            transient_terms: String::from("100"),
            table_rows: String::from("9"),
            x0: String::from("0.2"),
            r: String::from("3.2"),
            c_real: String::from("-0.123"),
            c_imag: String::from("0.745"),
            z0_real: String::from("0"),
            z0_imag: String::from("0"),
            x_range: String::from("-2,2,16"),
            y_range: String::from("-2,2,16"),
            r_range: String::from("0,2,12"),
            theta_range: format!("0,{},24", std::f64::consts::TAU),
            cases: Vec::new(),
            active_case_id: None,
            case_counter: 0,
            runs: Vec::new(),
            message: String::new(),
            dirty: true,
            music_criterion: String::from("melody"),
            music_segmentation: String::from("bands"),
            music_scale: String::from("minor"),
            strudel_code: String::from("silence"),
            bifurcation_points: compute_bifurcation_points(),
            mandelbrot_points: compute_mandelbrot_points(),
        };
        app.apply_config(&config);
        app
    }
}

impl eframe::App for VisualExplorerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.dirty {
            if let Err(error) = self.recalculate() {
                self.message = format!("Error: {error}");
            }
            self.dirty = false;
        }

        egui::SidePanel::left("controls").resizable(true).show(ctx, |ui| {
            ui.heading("Explorador");
            ui.separator();
            ui.label("Sucesion");
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.sequence, SequenceChoice::Cosine, "cos(x)");
                ui.selectable_value(&mut self.sequence, SequenceChoice::Logistic, "logistica");
                ui.selectable_value(&mut self.sequence, SequenceChoice::Quadratic, "z^2+c");
            });
            ui.label("Coordenadas");
            ui.horizontal_wrapped(|ui| {
                ui.selectable_value(&mut self.coordinates, CoordinateChoice::X, "x");
                ui.selectable_value(&mut self.coordinates, CoordinateChoice::Cartesian, "cartesiano");
                ui.selectable_value(&mut self.coordinates, CoordinateChoice::Polar, "polar");
                ui.selectable_value(&mut self.coordinates, CoordinateChoice::Y, "y");
                ui.selectable_value(&mut self.coordinates, CoordinateChoice::Radius, "radio");
                ui.selectable_value(&mut self.coordinates, CoordinateChoice::Angle, "angulo");
            });
            ui.separator();
            ui.label("Parametros");
            text_row(ui, "x0", &mut self.x0);
            text_row(ui, "r", &mut self.r);
            text_row(ui, "c real", &mut self.c_real);
            text_row(ui, "c imag", &mut self.c_imag);
            text_row(ui, "z0 real", &mut self.z0_real);
            text_row(ui, "z0 imag", &mut self.z0_imag);
            ui.separator();
            ui.label("Analisis");
            text_row(ui, "terminos", &mut self.max_terms);
            text_row(ui, "tol", &mut self.tolerance);
            text_row(ui, "periodo", &mut self.max_period);
            text_row(ui, "transitorio", &mut self.transient_terms);
            text_row(ui, "filas tabla", &mut self.table_rows);
            ui.separator();
            ui.label("Dominios min,max,n");
            text_row(ui, "x", &mut self.x_range);
            text_row(ui, "y", &mut self.y_range);
            text_row(ui, "r", &mut self.r_range);
            text_row(ui, "theta", &mut self.theta_range);

            ui.separator();
            if ui.button("Recalcular").clicked() {
                self.message.clear();
                self.dirty = true;
            }
            if ui.button("Caso base").clicked() {
                self.apply_config(&default_concept_config());
                self.message = String::from("Caso base aplicado.");
                self.dirty = true;
            }
            if ui.button("Guardar caso").clicked() {
                match self.current_config() {
                    Ok(config) => {
                        if let Err(error) =
                            self.store_case(config.recurrence, "manual", BTreeMap::new(), false, true)
                        {
                            self.message = format!("Error al guardar: {error}");
                        } else {
                            self.message = String::from("Caso guardado.");
                        }
                        self.dirty = true;
                    }
                    Err(error) => self.message = format!("Error: {error}"),
                }
            }
            if ui.button("Eliminar caso activo").clicked() {
                if let Some(active) = &self.active_case_id {
                    self.cases.retain(|case| &case.identifier != active);
                    self.active_case_id = None;
                    self.message = String::from("Caso eliminado.");
                    self.dirty = true;
                }
            }
            if ui.button("Exportar JSON").clicked() {
                if let Some(path) = rfd::FileDialog::new().add_filter("json", &["json"]).save_file() {
                    match self.export_json(&path) {
                        Ok(()) => self.message = String::from("Sesion JSON exportada."),
                        Err(error) => self.message = format!("Error al exportar JSON: {error}"),
                    }
                }
            }
            if ui.button("Importar JSON").clicked() {
                if let Some(path) = rfd::FileDialog::new().add_filter("json", &["json"]).pick_file() {
                    match self.import_json(&path) {
                        Ok(()) => {
                            self.message = String::from("Sesion importada.");
                            self.dirty = true;
                        }
                        Err(error) => self.message = format!("Error al importar: {error}"),
                    }
                }
            }
            if ui.button("Exportar CSV").clicked() {
                if let Some(path) = rfd::FileDialog::new().add_filter("csv", &["csv"]).save_file() {
                    let data = self
                        .runs
                        .iter()
                        .map(|run| (run.case.clone(), run.result.clone()))
                        .collect::<Vec<_>>();
                    match write_results_csv(path, &data) {
                        Ok(()) => self.message = String::from("CSV exportado."),
                        Err(error) => self.message = format!("Error al exportar CSV: {error}"),
                    }
                }
            }
            if ui.button("Limpiar todo").clicked() {
                self.cases.clear();
                self.active_case_id = None;
                self.runs.clear();
                self.message = String::from("Sesion vacia.");
                self.dirty = false;
            }
        });

        egui::TopBottomPanel::bottom("bottom_panel")
            .resizable(true)
            .show(ctx, |ui| {
                ui.separator();
                ui.label(if self.message.is_empty() {
                    "Listo".to_string()
                } else {
                    self.message.clone()
                });
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("Tabla:");
                    ui.selectable_value(&mut self.table_mode, TableMode::Todo, "todo");
                    ui.selectable_value(&mut self.table_mode, TableMode::Sucesion, "sucesion");
                    ui.selectable_value(&mut self.table_mode, TableMode::Atractores, "atractores");
                });
                self.draw_table(ui);
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.columns(2, |columns| {
                columns[0].heading("Trayectorias y atractores");
                self.draw_main_plot(&mut columns[0]);
                columns[0].separator();
                self.draw_summary(&mut columns[0]);

                columns[1].heading("Mapa de parametros");
                self.draw_parameter_plot(&mut columns[1]);
                columns[1].separator();
                self.draw_cases_panel(&mut columns[1]);
                columns[1].separator();
                self.draw_music_panel(&mut columns[1]);
            });
        });

        ctx.request_repaint();
    }
}

impl VisualExplorerApp {
    fn draw_main_plot(&self, ui: &mut egui::Ui) {
        let Some(reference) = self.runs.first() else {
            ui.label("Sesion vacia");
            return;
        };
        let axes = reference.result.transformer.coordinate_system.axes();
        let plot = Plot::new("main_plot").legend(Legend::default());
        plot.show(ui, |plot_ui| {
            for run in &self.runs {
                let width = if run.is_active { 2.0_f32 } else { 1.0_f32 };
                if axes.len() == 1 {
                    let axis = axes[0];
                    let points = run
                        .result
                        .terms
                        .iter()
                        .filter(|term| term.point.inside_domain)
                        .map(|term| [term.term.index as f64, term.point.coordinates[axis]])
                        .collect::<Vec<_>>();
                    plot_ui.line(Line::new(run.case.label.clone(), points).width(width));
                } else {
                    let x_axis = axes[0];
                    let y_axis = axes[1];
                    let points = run
                        .result
                        .terms
                        .iter()
                        .filter(|term| term.point.inside_domain)
                        .map(|term| [term.point.coordinates[x_axis], term.point.coordinates[y_axis]])
                        .collect::<Vec<_>>();
                    plot_ui.line(Line::new(run.case.label.clone(), points).width(width));
                }
                for attractor in &run.result.attractors {
                    let attr_points = if axes.len() == 1 {
                        attractor
                            .points
                            .iter()
                            .enumerate()
                            .filter(|(_, point)| point.inside_domain)
                            .map(|(index, point)| [index as f64, point.coordinates[axes[0]]])
                            .collect::<Vec<_>>()
                    } else {
                        attractor
                            .points
                            .iter()
                            .filter(|point| point.inside_domain)
                            .map(|point| [point.coordinates[axes[0]], point.coordinates[axes[1]]])
                            .collect::<Vec<_>>()
                    };
                    plot_ui.points(Points::new("atractor", attr_points).radius(4.0_f32));
                }
            }
        });
    }

    fn draw_parameter_plot(&mut self, ui: &mut egui::Ui) {
        match self.sequence {
            SequenceChoice::Cosine => {
                ui.label("cos(x): sin mapa de parametro");
            }
            SequenceChoice::Logistic => {
                let mut clicked: Option<PlotPoint> = None;
                Plot::new("bifurcation_plot")
                    .allow_zoom(true)
                    .allow_drag(true)
                    .show(ui, |plot_ui| {
                        plot_ui.points(
                            Points::new("bif", self.bifurcation_points.clone())
                                .radius(0.6_f32)
                                .color(egui::Color32::BLACK),
                        );
                        if let Ok(r_value) = self.r.parse::<f64>() {
                            plot_ui.vline(VLine::new("r", r_value).color(egui::Color32::RED));
                        }
                        if plot_ui.response().clicked() {
                            clicked = plot_ui.pointer_coordinate();
                        }
                    });
                if let Some(point) = clicked {
                    let r_value = point.x.clamp(0.0, 4.0);
                    let x_value = point.y.clamp(0.0, 1.0);
                    self.r = format!("{r_value:.10}");
                    self.x0 = format!("{x_value:.10}");
                    let converted = logistic_to_quadratic(r_value, x_value);
                    self.c_real = format!("{:.10}", converted.c.re);
                    self.c_imag = format!("{:.10}", converted.c.im);
                    self.z0_real = format!("{:.10}", converted.z0.re);
                    self.z0_imag = format!("{:.10}", converted.z0.im);
                    let _ = self.store_logistic_pair(r_value, x_value, "bifurcation_click", true);
                    self.message = String::from("Punto logistico seleccionado.");
                    self.dirty = true;
                }
            }
            SequenceChoice::Quadratic => {
                let mut clicked: Option<PlotPoint> = None;
                Plot::new("mandelbrot_plot")
                    .allow_zoom(true)
                    .allow_drag(true)
                    .data_aspect(1.0)
                    .show(ui, |plot_ui| {
                        plot_ui.points(
                            Points::new("mandelbrot", self.mandelbrot_points.clone())
                                .radius(0.9_f32)
                                .color(egui::Color32::DARK_GRAY),
                        );
                        if let (Ok(c_re), Ok(c_im)) = (self.c_real.parse::<f64>(), self.c_imag.parse::<f64>()) {
                            plot_ui.points(
                                Points::new("selected", vec![[c_re, c_im]])
                                    .radius(4.0_f32)
                                    .color(egui::Color32::WHITE),
                            );
                        }
                        plot_ui.set_plot_bounds(PlotBounds::from_min_max([-2.0, -1.35], [1.0, 1.35]));
                        if plot_ui.response().clicked() {
                            clicked = plot_ui.pointer_coordinate();
                        }
                    });
                if let Some(point) = clicked {
                    let c_re = point.x.clamp(-2.0, 1.0);
                    let c_im = point.y.clamp(-1.35, 1.35);
                    self.c_real = format!("{c_re:.10}");
                    self.c_imag = format!("{c_im:.10}");
                    self.z0_real = String::from("0");
                    self.z0_imag = String::from("0");
                    let _ = self.store_case(
                        RecurrenceConfig {
                            name: String::from("quadratic_complex"),
                            parameters: BTreeMap::from([
                                (String::from("c_real"), c_re),
                                (String::from("c_imag"), c_im),
                                (String::from("z0_real"), 0.0),
                                (String::from("z0_imag"), 0.0),
                            ]),
                        },
                        "mandelbrot_click",
                        BTreeMap::new(),
                        true,
                        true,
                    );
                    if let Some(converted) = quadratic_to_logistic(Complex64::new(c_re, c_im), Complex64::new(0.0, 0.0), 1e-9)
                    {
                        let _ = self.store_case(
                            RecurrenceConfig {
                                name: String::from("logistic"),
                                parameters: BTreeMap::from([
                                    (String::from("r"), converted.r),
                                    (String::from("x0"), converted.x0),
                                ]),
                            },
                            "mandelbrot_real_axis",
                            BTreeMap::new(),
                            false,
                            false,
                        );
                    }
                    self.message = String::from("Parametro complejo seleccionado.");
                    self.dirty = true;
                }
            }
        }
    }

    fn draw_summary(&self, ui: &mut egui::Ui) {
        ui.label(format!(
            "Casos calculados: {} | sistema: {}",
            self.runs.len(),
            self.coordinates_name()
        ));
        for run in self.runs.iter().take(5) {
            let behavior = match run.result.sequence_result.behavior {
                BehaviorKind::FixedPoint => "fixed_point",
                BehaviorKind::Periodic => "periodic",
                BehaviorKind::Divergent => "divergent",
                BehaviorKind::ChaoticOrUnresolved => "chaotic_or_unresolved",
                BehaviorKind::InsufficientData => "insufficient_data",
                BehaviorKind::CalculationError => "calculation_error",
            };
            let attractors = if run.result.attractors.is_empty() {
                String::from("sin atractor detectado")
            } else {
                run.result
                    .attractors
                    .iter()
                    .map(|attractor| format!("p={}", attractor.attractor.period))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            ui.monospace(format!(
                "{}: {}; {}; n={}",
                run.case.label,
                behavior,
                attractors,
                run.result.sequence_result.terms.len()
            ));
        }
    }

    fn draw_cases_panel(&mut self, ui: &mut egui::Ui) {
        ui.label("Casos");
        let mut selected: Option<(String, RecurrenceConfig, String)> = None;
        egui::ScrollArea::vertical().max_height(120.0).show(ui, |ui| {
            for case in &self.cases {
                let is_selected = self.active_case_id.as_ref() == Some(&case.identifier);
                if ui.selectable_label(is_selected, &case.label).clicked() {
                    selected = Some((case.identifier.clone(), case.recurrence.clone(), case.label.clone()));
                }
            }
        });
        if let Some((identifier, recurrence, label)) = selected {
            self.active_case_id = Some(identifier);
            self.apply_recurrence(&recurrence);
            self.message = format!("Caso activo: {label}");
            self.dirty = true;
        }
    }

    fn draw_music_panel(&mut self, ui: &mut egui::Ui) {
        ui.label("Strudel");
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.music_criterion, String::from("melody"), "melodia");
            ui.selectable_value(&mut self.music_criterion, String::from("harmony"), "armonia");
            ui.selectable_value(&mut self.music_criterion, String::from("rhythm"), "ritmo");
            ui.selectable_value(&mut self.music_criterion, String::from("texture"), "textura");
        });
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.music_segmentation, String::from("bands"), "franjas");
            ui.selectable_value(&mut self.music_segmentation, String::from("positions"), "posiciones");
            ui.selectable_value(&mut self.music_segmentation, String::from("sectors"), "sectores");
        });
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.music_scale, String::from("major"), "mayor");
            ui.selectable_value(&mut self.music_scale, String::from("minor"), "menor");
            ui.selectable_value(&mut self.music_scale, String::from("pentatonic"), "pentatonica");
            ui.selectable_value(&mut self.music_scale, String::from("chromatic"), "cromatica");
        });
        if ui.button("Generar").clicked() {
            let pairs = self
                .runs
                .iter()
                .map(|run| (run.case.label.clone(), run.result.clone()))
                .collect::<Vec<_>>();
            if let Ok(pattern) = generate_strudel_pattern(
                &pairs,
                Some(MusicMappingConfig {
                    criterion: self.music_criterion.clone(),
                    segmentation: self.music_segmentation.clone(),
                    scale: self.music_scale.clone(),
                    root: String::from("C4"),
                    octaves: 2,
                }),
            ) {
                self.strudel_code = pattern.code;
            }
        }
        if ui.button("Copiar").clicked() {
            if let Ok(mut clipboard) = arboard::Clipboard::new() {
                let _ = clipboard.set_text(self.strudel_code.clone());
                self.message = String::from("Comando Strudel copiado.");
            }
        }
        ui.add(egui::TextEdit::multiline(&mut self.strudel_code).desired_rows(3));
    }

    fn draw_table(&self, ui: &mut egui::Ui) {
        let rows = self.table_rows.parse::<usize>().unwrap_or(9).max(1);
        let data = self.build_table_rows();
        egui::ScrollArea::vertical().max_height(200.0).show_rows(ui, 20.0, data.len(), |ui, range| {
            egui::Grid::new("data_grid").striped(true).show(ui, |ui| {
                ui.strong("caso");
                ui.strong("tipo");
                ui.strong("n");
                ui.strong("valor");
                ui.strong("coord");
                ui.strong("idx");
                ui.strong("centro");
                ui.strong("celda");
                ui.end_row();
                for row_index in range.take(rows) {
                    let row = &data[row_index];
                    for value in row {
                        ui.label(value);
                    }
                    ui.end_row();
                }
            });
        });
    }

    fn build_table_rows(&self) -> Vec<[String; 8]> {
        let mut rows = Vec::new();
        for run in &self.runs {
            if matches!(self.table_mode, TableMode::Todo | TableMode::Sucesion) {
                for item in &run.result.terms {
                    rows.push([
                        run.case.label.clone(),
                        String::from("S"),
                        item.term.index.to_string(),
                        format_number(item.point.value),
                        format_mapping_float(item.point.coordinates.iter().map(|(k, v)| (k, *v))),
                        format_mapping_usize(item.point.indices.as_ref()),
                        format_mapping_float_opt(item.point.discrete_coordinates.as_ref()),
                        item.point.cell_id.map_or_else(|| String::from("-"), |v| v.to_string()),
                    ]);
                }
            }
            if matches!(self.table_mode, TableMode::Todo | TableMode::Atractores) {
                for (attractor_index, attractor) in run.result.attractors.iter().enumerate() {
                    for (point_index, point) in attractor.points.iter().enumerate() {
                        rows.push([
                            run.case.label.clone(),
                            format!("A{}.{}", attractor_index + 1, point_index + 1),
                            String::from("-"),
                            format_number(point.value),
                            format_mapping_float(point.coordinates.iter().map(|(k, v)| (k, *v))),
                            format_mapping_usize(point.indices.as_ref()),
                            format_mapping_float_opt(point.discrete_coordinates.as_ref()),
                            point.cell_id.map_or_else(|| String::from("-"), |v| v.to_string()),
                        ]);
                    }
                }
            }
        }
        rows
    }

    fn current_config(&self) -> Result<ExplorerConfig> {
        let recurrence = match self.sequence {
            SequenceChoice::Cosine => RecurrenceConfig {
                name: String::from("cosine"),
                parameters: BTreeMap::from([(String::from("x0"), parse_float(&self.x0)?)]),
            },
            SequenceChoice::Logistic => RecurrenceConfig {
                name: String::from("logistic"),
                parameters: BTreeMap::from([
                    (String::from("r"), parse_float(&self.r)?),
                    (String::from("x0"), parse_float(&self.x0)?),
                ]),
            },
            SequenceChoice::Quadratic => RecurrenceConfig {
                name: String::from("quadratic_complex"),
                parameters: BTreeMap::from([
                    (String::from("c_real"), parse_float(&self.c_real)?),
                    (String::from("c_imag"), parse_float(&self.c_imag)?),
                    (String::from("z0_real"), parse_float(&self.z0_real)?),
                    (String::from("z0_imag"), parse_float(&self.z0_imag)?),
                ]),
            },
        };
        let selected_axes = match self.coordinates {
            CoordinateChoice::X => vec!["x"],
            CoordinateChoice::Cartesian => vec!["x", "y"],
            CoordinateChoice::Polar => vec!["r", "theta"],
            CoordinateChoice::Y => vec!["y"],
            CoordinateChoice::Radius => vec!["r"],
            CoordinateChoice::Angle => vec!["theta"],
        };
        let mut ranges = BTreeMap::new();
        for axis in selected_axes {
            let value = match axis {
                "x" => &self.x_range,
                "y" => &self.y_range,
                "r" => &self.r_range,
                "theta" => &self.theta_range,
                _ => bail!("eje no compatible"),
            };
            ranges.insert(axis.to_string(), parse_axis_range(value)?);
        }
        let coordinate_system = match self.coordinates {
            CoordinateChoice::X => "x",
            CoordinateChoice::Cartesian => "cartesian",
            CoordinateChoice::Polar => "polar",
            CoordinateChoice::Y => "y",
            CoordinateChoice::Radius => "radius",
            CoordinateChoice::Angle => "angle",
        };
        Ok(ExplorerConfig {
            recurrence,
            analysis: AnalysisConfig {
                max_terms: parse_int(&self.max_terms)? as usize,
                tolerance: parse_float(&self.tolerance)?,
                divergence_limit: 1e6,
                attractor_window: None,
                max_period: parse_int(&self.max_period)? as usize,
                min_repetitions: 5,
                transient_terms: parse_int(&self.transient_terms)? as usize,
                stability_ratio: 0.98,
                cluster_tolerance: None,
            },
            discretization: DiscretizationConfig {
                coordinate_system: coordinate_system.to_string(),
                ranges,
            },
        })
    }

    fn recalculate(&mut self) -> Result<()> {
        let config = self.current_config()?;
        let current_signature = recurrence_signature(&config.recurrence);
        let known_signatures = self
            .cases
            .iter()
            .map(|case| recurrence_signature(&case.recurrence))
            .collect::<HashSet<_>>();
        let mut cases_to_run = self.cases.clone();
        if !known_signatures.contains(&current_signature) {
            cases_to_run.insert(
                0,
                ExplorationCase {
                    identifier: String::from("__actual__"),
                    label: String::from("Actual"),
                    recurrence: config.recurrence.clone(),
                    source: String::from("working_copy"),
                    metadata: BTreeMap::new(),
                },
            );
        }
        let mut runs = Vec::new();
        for case in cases_to_run {
            let run_config = ExplorerConfig {
                recurrence: case.recurrence.clone(),
                analysis: config.analysis.clone(),
                discretization: config.discretization.clone(),
            };
            let result = run_exploration(&run_config)?;
            let is_active =
                self.active_case_id.as_ref() == Some(&case.identifier) || case.identifier == "__actual__";
            runs.push(CaseRun {
                case,
                result,
                is_active,
            });
        }
        self.runs = runs;
        let pairs = self
            .runs
            .iter()
            .map(|run| (run.case.label.clone(), run.result.clone()))
            .collect::<Vec<_>>();
        if let Ok(pattern) = generate_strudel_pattern(
            &pairs,
            Some(MusicMappingConfig {
                criterion: self.music_criterion.clone(),
                segmentation: self.music_segmentation.clone(),
                scale: self.music_scale.clone(),
                root: String::from("C4"),
                octaves: 2,
            }),
        ) {
            self.strudel_code = pattern.code;
        }
        Ok(())
    }

    fn apply_config(&mut self, config: &ExplorerConfig) {
        self.apply_recurrence(&config.recurrence);
        self.max_terms = config.analysis.max_terms.to_string();
        self.tolerance = config.analysis.tolerance.to_string();
        self.max_period = config.analysis.max_period.to_string();
        self.transient_terms = config.analysis.transient_terms.to_string();
        for (axis, axis_config) in &config.discretization.ranges {
            let value = format!("{},{},{}", axis_config.minimum, axis_config.maximum, axis_config.steps);
            match axis.as_str() {
                "x" => self.x_range = value,
                "y" => self.y_range = value,
                "r" => self.r_range = value,
                "theta" => self.theta_range = value,
                _ => {}
            }
        }
    }

    fn apply_recurrence(&mut self, recurrence: &RecurrenceConfig) {
        match recurrence.name.as_str() {
            "cosine" => {
                self.sequence = SequenceChoice::Cosine;
                self.x0 = format!("{:.10}", recurrence.parameters.get("x0").copied().unwrap_or(0.2));
            }
            "logistic" => {
                self.sequence = SequenceChoice::Logistic;
                let r_value = recurrence.parameters.get("r").copied().unwrap_or(3.2);
                let x_value = recurrence.parameters.get("x0").copied().unwrap_or(0.2);
                self.r = format!("{r_value:.10}");
                self.x0 = format!("{x_value:.10}");
                let converted = logistic_to_quadratic(r_value, x_value);
                self.c_real = format!("{:.10}", converted.c.re);
                self.c_imag = format!("{:.10}", converted.c.im);
                self.z0_real = format!("{:.10}", converted.z0.re);
                self.z0_imag = format!("{:.10}", converted.z0.im);
            }
            "quadratic_complex" => {
                self.sequence = SequenceChoice::Quadratic;
                self.c_real = format!(
                    "{:.10}",
                    recurrence.parameters.get("c_real").copied().unwrap_or(-0.123)
                );
                self.c_imag = format!(
                    "{:.10}",
                    recurrence.parameters.get("c_imag").copied().unwrap_or(0.745)
                );
                self.z0_real = format!(
                    "{:.10}",
                    recurrence.parameters.get("z0_real").copied().unwrap_or(0.0)
                );
                self.z0_imag = format!(
                    "{:.10}",
                    recurrence.parameters.get("z0_imag").copied().unwrap_or(0.0)
                );
            }
            _ => {}
        }
    }

    fn export_json(&self, path: &std::path::Path) -> Result<()> {
        let config = self.current_config()?;
        let data = build_session_data(
            &config,
            &self.cases,
            &self
                .runs
                .iter()
                .map(|run| (run.case.clone(), run.result.clone()))
                .collect::<Vec<_>>(),
            self.active_case_id.clone(),
            match self.table_mode {
                TableMode::Todo => "todo",
                TableMode::Sucesion => "sucesion",
                TableMode::Atractores => "atractores",
            },
            &BTreeMap::new(),
            &BTreeMap::from([
                (String::from("criterion"), self.music_criterion.clone()),
                (String::from("segmentation"), self.music_segmentation.clone()),
                (String::from("scale"), self.music_scale.clone()),
            ]),
        );
        write_session(path, &data)
    }

    fn import_json(&mut self, path: &std::path::Path) -> Result<()> {
        let raw = std::fs::read_to_string(path)?;
        let data = serde_json::from_str::<serde_json::Value>(&raw)?;
        let imported = parse_session_data(&data)?;
        self.cases = imported.cases;
        self.active_case_id = imported.active_case_id;
        self.case_counter = self.cases.len();
        self.apply_config(&imported.config);
        self.music_criterion = imported
            .music
            .get("criterion")
            .cloned()
            .unwrap_or_else(|| String::from("melody"));
        self.music_segmentation = imported
            .music
            .get("segmentation")
            .cloned()
            .unwrap_or_else(|| String::from("bands"));
        self.music_scale = imported
            .music
            .get("scale")
            .cloned()
            .unwrap_or_else(|| String::from("minor"));
        self.table_mode = match imported.table_mode.as_str() {
            "sucesion" => TableMode::Sucesion,
            "atractores" => TableMode::Atractores,
            _ => TableMode::Todo,
        };
        Ok(())
    }

    fn store_logistic_pair(&mut self, r_value: f64, x_value: f64, source: &str, force_new: bool) -> Result<()> {
        let logistic_case = self.store_case(
            RecurrenceConfig {
                name: String::from("logistic"),
                parameters: BTreeMap::from([
                    (String::from("r"), r_value),
                    (String::from("x0"), x_value),
                ]),
            },
            source,
            BTreeMap::new(),
            force_new,
            true,
        )?;
        let converted = logistic_to_quadratic(r_value, x_value);
        let _ = self.store_case(
            RecurrenceConfig {
                name: String::from("quadratic_complex"),
                parameters: BTreeMap::from([
                    (String::from("c_real"), converted.c.re),
                    (String::from("c_imag"), converted.c.im),
                    (String::from("z0_real"), converted.z0.re),
                    (String::from("z0_imag"), converted.z0.im),
                ]),
            },
            "logistic_isomorphism",
            BTreeMap::from([(String::from("linked_case"), logistic_case.identifier)]),
            false,
            false,
        )?;
        Ok(())
    }

    fn store_case(
        &mut self,
        recurrence: RecurrenceConfig,
        source: &str,
        metadata: BTreeMap<String, String>,
        force_new: bool,
        make_active: bool,
    ) -> Result<ExplorationCase> {
        let signature = recurrence_signature(&recurrence);
        if !force_new {
            if let Some(existing) = self
                .cases
                .iter()
                .find(|case| recurrence_signature(&case.recurrence) == signature)
                .cloned()
            {
                if make_active {
                    self.active_case_id = Some(existing.identifier.clone());
                }
                return Ok(existing);
            }
        }
        self.case_counter += 1;
        let identifier = format!("case-{:03}", self.case_counter);
        let label = match recurrence.name.as_str() {
            "cosine" => format!(
                "C{}: x0={:.4}",
                self.case_counter,
                recurrence.parameters.get("x0").copied().unwrap_or(0.0)
            ),
            "logistic" => format!(
                "L{}: r={:.5}",
                self.case_counter,
                recurrence.parameters.get("r").copied().unwrap_or(0.0)
            ),
            _ => format!(
                "Q{}: c={:.4}{:+.4}j",
                self.case_counter,
                recurrence.parameters.get("c_real").copied().unwrap_or(0.0),
                recurrence.parameters.get("c_imag").copied().unwrap_or(0.0)
            ),
        };
        let case = ExplorationCase {
            identifier: identifier.clone(),
            label,
            recurrence,
            source: source.to_string(),
            metadata,
        };
        self.cases.push(case.clone());
        if make_active {
            self.active_case_id = Some(identifier);
        }
        Ok(case)
    }

    fn coordinates_name(&self) -> &'static str {
        match self.coordinates {
            CoordinateChoice::X => "x",
            CoordinateChoice::Cartesian => "cartesiano",
            CoordinateChoice::Polar => "polar",
            CoordinateChoice::Y => "y",
            CoordinateChoice::Radius => "radio",
            CoordinateChoice::Angle => "angulo",
        }
    }
}

fn text_row(ui: &mut egui::Ui, label: &str, value: &mut String) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.text_edit_singleline(value);
    });
}

fn parse_float(text: &str) -> Result<f64> {
    Ok(text.trim().parse::<f64>()?)
}

fn parse_int(text: &str) -> Result<i64> {
    Ok(text.trim().parse::<f64>()? as i64)
}

fn parse_axis_range(text: &str) -> Result<AxisRangeConfig> {
    let values = text
        .split(',')
        .map(|item| item.trim())
        .collect::<Vec<_>>();
    if values.len() != 3 {
        bail!("Formato de rango invalido: {text}");
    }
    let minimum = values[0].parse::<f64>()?;
    let maximum = values[1].parse::<f64>()?;
    let steps = values[2].parse::<f64>()? as usize;
    Ok(AxisRangeConfig {
        minimum,
        maximum,
        steps,
    })
}

fn recurrence_signature(recurrence: &RecurrenceConfig) -> String {
    format!("{:?}{:?}", recurrence.name, recurrence.parameters)
}

fn format_number(value: Complex64) -> String {
    if value.im == 0.0 {
        format!("{:.5}", value.re)
    } else {
        format!("{:.5}{:+.5}j", value.re, value.im)
    }
}

fn format_mapping_float<'a>(iter: impl Iterator<Item = (&'a String, f64)>) -> String {
    iter.map(|(key, value)| format!("{key}={value:.5}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn format_mapping_float_opt(mapping: Option<&indexmap::IndexMap<String, f64>>) -> String {
    mapping.map_or_else(
        || String::from("-"),
        |map| format_mapping_float(map.iter().map(|(k, v)| (k, *v))),
    )
}

fn format_mapping_usize(mapping: Option<&indexmap::IndexMap<String, usize>>) -> String {
    mapping.map_or_else(
        || String::from("-"),
        |map| {
            map.iter()
                .map(|(key, value)| format!("{key}={value}"))
                .collect::<Vec<_>>()
                .join(", ")
        },
    )
}

fn compute_bifurcation_points() -> Vec<[f64; 2]> {
    let steps = 700;
    let mut points = Vec::new();
    for i in 0..steps {
        let r = i as f64 * 4.0 / (steps - 1) as f64;
        let mut x = 0.5;
        for iteration in 0..520 {
            x = r * x * (1.0 - x);
            if iteration > 350 {
                points.push([r, x]);
            }
        }
    }
    points
}

fn compute_mandelbrot_points() -> Vec<[f64; 2]> {
    let width = 220;
    let height = 170;
    let mut points = Vec::new();
    for ix in 0..width {
        for iy in 0..height {
            let x = -2.0 + 3.0 * ix as f64 / (width - 1) as f64;
            let y = -1.35 + 2.7 * iy as f64 / (height - 1) as f64;
            let c = Complex64::new(x, y);
            let mut z = Complex64::new(0.0, 0.0);
            let mut escaped = false;
            for _ in 0..64 {
                z = z * z + c;
                if z.norm() > 2.0 {
                    escaped = true;
                    break;
                }
            }
            if !escaped {
                points.push([x, y]);
            }
        }
    }
    points
}
