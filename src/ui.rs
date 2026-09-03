use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use eframe::egui;
use egui::{Color32, RichText};
use egui_plot::{HLine, Line, MarkerShape, Plot, PlotBounds, PlotPoint, PlotResponse, Points, Text, VLine};
use egui_wgpu::Callback as WgpuCallback;
use epaint::Shape;
use num_complex::Complex64;
use crate::audio::describe_strudel;

use crate::{
    AnalysisConfig, AxisRangeConfig, BehaviorKind, DiscretizationConfig, DiscreteAnalysisResult,
    ExplorationCase, ExplorerConfig, ImportedSession, MandelbrotCallback, MusicMappingConfig, RecurrenceConfig,
    build_coordinate_system, build_grid, build_session_data, default_concept_config, generate_strudel_pattern, logistic_to_quadratic,
    parse_session_data, quadratic_to_logistic, run_exploration, write_session, DiscreteTransformer,
};

const MAP_BOUNDS_LOGISTIC: (f64, f64, f64, f64) = (0.0, 4.0, 0.0, 1.0);
const MAP_BOUNDS_MANDELBROT: (f64, f64, f64, f64) = (-2.0, 1.0, -1.35, 1.35);
const ZOOM_SCALE: f64 = 1.55;
const DEFAULT_TOLERANCE: f64 = 1e-7;
const VISIBLE_TABLE_ROWS: usize = 4;
const TABLE_ROW_HEIGHT: f32 = 18.0;
const MAX_RENDERED_TRAJECTORY_POINTS: usize = 2_000;
const EMBEDDED_STRUDEL: &[u8] = include_bytes!("../resources/strudel/strudel.exe");

fn app_accent_color() -> Color32 {
    Color32::from_rgb(42, 104, 156)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Language {
    Spanish,
    English,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ThemeChoice {
    Dark,
    Light,
}

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
    mandelbrot_member: Option<bool>,
    trajectory_points: Vec<PlotPoint>,
    attractor_points: Vec<Vec<PlotPoint>>,
}

pub struct VisualExplorerApp {
    language: Language,
    theme: ThemeChoice,
    sequence: SequenceChoice,
    coordinates: CoordinateChoice,
    table_mode: TableMode,
    selected_table_mode: TableMode,
    max_terms: String,
    tolerance: String,
    max_period: String,
    transient_terms: String,
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
    case_colors: HashMap<String, Color32>,
    active_case_id: Option<String>,
    selected_case_ids: HashSet<String>,
    case_selection_anchor: Option<usize>,
    cases_panel_rect: Option<egui::Rect>,
    point_actions_rect: Option<egui::Rect>,
    case_counter: usize,
    runs: Vec<CaseRun>,
    cached_analysis: Option<AnalysisConfig>,
    cached_discretization: Option<DiscretizationConfig>,
    table_cache: HashMap<String, Vec<[String; 8]>>,
    message: String,
    message_until: Option<Instant>,
    custom_fullscreen: bool,
    fullscreen_configured: bool,
    point_actions: Option<(f64, f64)>,
    point_actions_position: Option<egui::Pos2>,
    case_context_menu_open: bool,
    julia_open: bool,
    julia_point: Option<(f64, f64)>,
    julia_limits: (f64, f64, f64, f64),
    dirty: bool,
    music_criterion: String,
    music_segmentation: String,
    music_scale: String,
    strudel_code: String,
    logo_texture: Option<egui::TextureHandle>,
    strudel_process: Option<std::process::Child>,
    strudel_window_mode_active: bool,
    /*
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
            self.message = String::from("Patrón de Strudel actualizado.");
        }
    }

    */
    #[allow(dead_code)]
    strudel_editor_open: bool,
    strudel_help_open: bool,
    strudel_details_open: bool,
    export_directory: String,
bifurcation_points: Vec<PlotPoint>,
    map_limits: BTreeMap<String, (f64, f64, f64, f64)>,
    last_sequence: SequenceChoice,
    case_list_start: usize,
    table_start: usize,
    target_format: wgpu::TextureFormat,
// Pan/drag state for Mandelbrot interactive panning (Alt+drag to pan).
mandelbrot_pan_start: Option<egui::Pos2>,
mandelbrot_pan_limits_start: Option<(f64, f64, f64, f64)>,
// Pan state for parameter plots (e.g., bifurcation/logistic)
    parameter_pan_start: Option<egui::Pos2>,
    parameter_pan_limits_start: Option<(f64, f64, f64, f64)>,
    // Vista del plano de trayectorias (cartesiano, polar y proyecciones).
    main_plot_limits: Option<(f64, f64, f64, f64)>,
    cached_main_auto_limits: Option<(f64, f64, f64, f64)>,
    main_plot_pan_start: Option<egui::Pos2>,
    main_plot_pan_limits_start: Option<(f64, f64, f64, f64)>,
}

impl VisualExplorerApp {
    /// Create a new app with the given wgpu target format (used for the Mandelbrot GPU renderer).
    pub fn new(target_format: wgpu::TextureFormat) -> Self {
        let config = default_concept_config();
        let mut app = Self {
            language: Language::Spanish,
            theme: ThemeChoice::Dark,
            sequence: SequenceChoice::Logistic,
            coordinates: CoordinateChoice::Cartesian,
            table_mode: TableMode::Sucesion,
            selected_table_mode: TableMode::Sucesion,
            max_terms: String::from("1000"),
            tolerance: String::from("1e-7"),
            max_period: String::from("16"),
            transient_terms: String::from("100"),
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
            case_colors: HashMap::new(),
            active_case_id: None,
            selected_case_ids: HashSet::new(),
            case_selection_anchor: None,
            cases_panel_rect: None,
            point_actions_rect: None,
            case_counter: 0,
            runs: Vec::new(),
            cached_analysis: None,
            cached_discretization: None,
            table_cache: HashMap::new(),
            message: String::new(),
            message_until: None,
            custom_fullscreen: true,
            fullscreen_configured: true,
            point_actions: None,
            point_actions_position: None,
            case_context_menu_open: false,
            julia_open: false,
            julia_point: None,
            julia_limits: (-2.0, 2.0, -2.0, 2.0),
            dirty: true,
            music_criterion: String::from("melody"),
            music_segmentation: String::from("bands"),
            music_scale: String::from("minor"),
            strudel_code: String::from("silence"),
            logo_texture: None,
            strudel_process: None,
            strudel_window_mode_active: false,
            strudel_editor_open: false,
            strudel_help_open: false,
            strudel_details_open: false,
            export_directory: default_export_directory(),
bifurcation_points: compute_bifurcation_points(),
            map_limits: BTreeMap::from([
                (String::from("logistica"), MAP_BOUNDS_LOGISTIC),
                (String::from("z^2+c"), MAP_BOUNDS_MANDELBROT),
            ]),
            last_sequence: SequenceChoice::Logistic,
            case_list_start: 0,
            table_start: 0,
            target_format,
            mandelbrot_pan_start: None,
            mandelbrot_pan_limits_start: None,
            parameter_pan_start: None,
            parameter_pan_limits_start: None,
            main_plot_limits: None,
            cached_main_auto_limits: None,
            main_plot_pan_start: None,
            main_plot_pan_limits_start: None,
        };
        app.apply_config(&config);
        // Abrir directamente en el mapa de Mandelbrot, usando los parámetros
        // cuadráticos que acaba de cargar la configuración conceptual.
        app.sequence = SequenceChoice::Quadratic;
        app.last_sequence = SequenceChoice::Quadratic;
        app.coordinates = CoordinateChoice::Cartesian;
        
        app
    }
}

impl Default for VisualExplorerApp {
    fn default() -> Self {
        Self::new(wgpu::TextureFormat::Rgba8Unorm)
    }
}

impl eframe::App for VisualExplorerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let strudel_closed = self
            .strudel_process
            .as_mut()
            .and_then(|process| process.try_wait().ok())
            .flatten()
            .is_some();
        if strudel_closed {
            self.strudel_process = None;
            if self.strudel_window_mode_active {
                self.restore_fullscreen(ctx);
                self.strudel_window_mode_active = false;
            }
        } else if self.strudel_process.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(50));
        }
        ctx.set_visuals(match self.theme {
            ThemeChoice::Dark => egui::Visuals::dark(),
            ThemeChoice::Light => egui::Visuals::light(),
        });
        ctx.style_mut(|style| {
            style.visuals.selection.bg_fill = app_accent_color();
            style.visuals.selection.stroke.color = Color32::WHITE;
        });
        if self.logo_texture.is_none() {
            let bytes = include_bytes!("../assets/mandelbrot_logo.png");
            if let Ok(image) = image::load_from_memory(bytes) {
                let image = image.to_rgba8();
                let size = [image.width() as usize, image.height() as usize];
                let color_image = egui::ColorImage::from_rgba_unmultiplied(size, image.as_raw());
                self.logo_texture = Some(ctx.load_texture(
                    "mandelbrot_app_logo",
                    color_image,
                    egui::TextureOptions::NEAREST,
                ));
            }
        }

        if ctx.input(|input| input.pointer.any_click()) {
            if let Some(position) = ctx.input(|input| input.pointer.interact_pos()) {
                let primary_click = ctx.input(|input| input.pointer.button_clicked(egui::PointerButton::Primary));
                let had_context_menu = self.case_context_menu_open;
                if primary_click
                    && !self
                        .point_actions_rect
                        .is_some_and(|rect| rect.contains(position))
                {
                    self.point_actions = None;
                    self.point_actions_position = None;
                    self.case_context_menu_open = false;
                }
                if primary_click
                    && had_context_menu
                    && self.cases_panel_rect.is_some_and(|rect| rect.contains(position))
                    && !self
                        .point_actions_rect
                        .is_some_and(|rect| rect.contains(position))
                {
                    self.selected_case_ids.clear();
                    self.case_selection_anchor = None;
                    self.active_case_id = None;
                }
                if !self
                    .cases_panel_rect
                    .is_some_and(|rect| rect.contains(position))
                {
                    self.selected_case_ids.clear();
                    self.case_selection_anchor = None;
                    self.active_case_id = None;
                }
            }
        }

        if ctx.input(|input| input.key_pressed(egui::Key::Escape)) {
            self.custom_fullscreen = false;
            self.fullscreen_configured = false;
            ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(false));
            ctx.send_viewport_cmd(egui::ViewportCommand::Decorations(true));
            ctx.send_viewport_cmd(egui::ViewportCommand::Maximized(true));
        }

        if let Some(until) = self.message_until {
            let now = Instant::now();
            if now >= until {
                self.message.clear();
                self.message_until = None;
            } else {
                ctx.request_repaint_after(until.saturating_duration_since(now));
            }
        }
        // Detectar el cambio de la secuencia para hacer la conversión automática.
        if self.sequence != self.last_sequence {
            if self.last_sequence == SequenceChoice::Logistic
                && self.sequence == SequenceChoice::Quadratic
            {
                if let (Ok(r_value), Ok(x_value)) = (self.r.parse::<f64>(), self.x0.parse::<f64>())
                {
                    let converted = logistic_to_quadratic(r_value, x_value);
                    self.c_real = format!("{:.10}", converted.c.re);
                    self.c_imag = format!("{:.10}", converted.c.im);
                    self.z0_real = format!("{:.10}", converted.z0.re);
                    self.z0_imag = format!("{:.10}", converted.z0.im);
                    self.message = String::from("Parámetros convertidos de logística a z^2+c.");
                    self.dirty = true;
                }
            } else if self.last_sequence == SequenceChoice::Quadratic
                && self.sequence == SequenceChoice::Logistic
            {
                if let (Ok(c_re), Ok(c_im)) = (self.c_real.parse::<f64>(), self.c_imag.parse::<f64>())
                {
                    if let Some(converted) = quadratic_to_logistic(
                        Complex64::new(c_re, c_im),
                        Complex64::new(
                            self.z0_real.parse::<f64>().unwrap_or(0.0),
                            self.z0_imag.parse::<f64>().unwrap_or(0.0),
                        ),
                        self.tolerance_value(),
                    ) {
                        self.r = format!("{:.10}", converted.r);
                        self.x0 = format!("{:.10}", converted.x0);
                        self.message = String::from("Parámetros convertidos de z^2+c a logística.");
                        self.dirty = true;
                    }
                }
            }
            self.last_sequence = self.sequence;
        }

        if self.dirty {
            if let Err(error) = self.recalculate() {
                self.message = format!("Error: {error}");
            }
            self.dirty = false;
        }

        let language = self.language;
        egui::TopBottomPanel::top("app_menu_bar")
            .exact_height(34.0)
            .show(ctx, |ui| {
                ui.horizontal_centered(|ui| {
                    draw_app_logo(ui, self.logo_texture.as_ref());

                    ui.menu_button(localized(language, "Archivo", "File"), |ui| {
                        if ui.button(localized(language, "Importar JSON…", "Import JSON…")).clicked() {
                            self.open_import_json();
                            ui.close();
                        }
                        if ui.button(localized(language, "Exportar JSON…", "Export JSON…")).clicked() {
                            self.open_export_json();
                            ui.close();
                        }
                        if ui.button(localized(language, "Exportar tablas CSV…", "Export tables CSV…")).clicked() {
                            self.open_export_tables_csv();
                            ui.close();
                        }
                    });

                    ui.menu_button(localized(language, "Configuración", "Settings"), |ui| {
                        ui.label(localized(language, "Idioma", "Language"));
                        ui.selectable_value(&mut self.language, Language::Spanish, "Español");
                        ui.selectable_value(&mut self.language, Language::English, "English");
                        ui.separator();
                        ui.label(localized(language, "Tema", "Theme"));
                        ui.selectable_value(&mut self.theme, ThemeChoice::Dark, localized(language, "Oscuro", "Dark"));
                        ui.selectable_value(&mut self.theme, ThemeChoice::Light, localized(language, "Claro", "Light"));
                        ui.separator();
                        ui.label(localized(language, "Carpeta de exportación", "Export folder"));
                        ui.horizontal(|ui| {
                            ui.add(egui::TextEdit::singleline(&mut self.export_directory).desired_width(220.0));
                            if ui.button(localized(language, "Elegir…", "Choose…")).clicked() {
                                if let Some(path) = rfd::FileDialog::new().pick_folder() {
                                    self.export_directory = path.to_string_lossy().into_owned();
                                }
                            }
                        });
                    });

                    ui.menu_button(localized(language, "Editar", "Edit"), |ui| {
                        ui.menu_button(localized(language, "Análisis", "Analysis"), |ui| {
                            let enter_max_terms = text_row(ui, "términos", &mut self.max_terms);
                            let enter_tolerance = text_row(ui, "tolerancia", &mut self.tolerance);
                            let enter_max_period = text_row(ui, "período", &mut self.max_period);
                            let enter_transient_terms = text_row(ui, "transitorio", &mut self.transient_terms);
                            if enter_max_terms || enter_tolerance || enter_max_period || enter_transient_terms {
                                self.table_cache.clear();
                                self.dirty = true;
                            }
                        });
                        ui.menu_button(localized(language, "Dominios", "Domains"), |ui| {
                            ui.label(RichText::new(localized(language, "Formato: min,max,n", "Format: min,max,n")).weak().small());
                            let enter_x_range = text_row(ui, "x", &mut self.x_range);
                            let enter_y_range = text_row(ui, "y", &mut self.y_range);
                            let enter_r_range = text_row(ui, "r", &mut self.r_range);
                            let enter_theta_range = text_row(ui, "theta", &mut self.theta_range);
                            if enter_x_range || enter_y_range || enter_r_range || enter_theta_range {
                                self.table_cache.clear();
                                self.dirty = true;
                            }
                        });
                        if ui.button(localized(language, "Recalcular análisis", "Recalculate analysis")).clicked() {
                            self.table_cache.clear();
                            self.dirty = true;
                            ui.close();
                        }
                        if ui.button(localized(language, "Restablecer dominios", "Reset domains")).clicked() {
                            self.x_range = String::from("-2,2,16");
                            self.y_range = String::from("-2,2,16");
                            self.r_range = String::from("0,2,12");
                            self.theta_range = format!("0,{},24", std::f64::consts::TAU);
                            self.dirty = true;
                            ui.close();
                        }
                    });

                    ui.menu_button(localized(language, "Ver", "View"), |ui| {
                        if ui.button(localized(language, "Pantalla completa", "Fullscreen")).clicked() {
                            self.custom_fullscreen = true;
                            self.fullscreen_configured = true;
                            ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(false));
                            ctx.send_viewport_cmd(egui::ViewportCommand::Decorations(false));
                            if let Some(monitor_size) = ctx.input(|input| input.viewport().monitor_size) {
                                ctx.send_viewport_cmd(egui::ViewportCommand::OuterPosition(egui::pos2(0.0, 0.0)));
                                ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(monitor_size));
                            }
                            ctx.request_repaint();
                            ui.close();
                        }
                        if ui.button(localized(language, "Ventana sin bordes", "Borderless window")).clicked() {
                            self.custom_fullscreen = true;
                            self.fullscreen_configured = false;
                            ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(false));
                            ctx.send_viewport_cmd(egui::ViewportCommand::Decorations(false));
                            ctx.send_viewport_cmd(egui::ViewportCommand::Maximized(true));
                            ctx.request_repaint();
                            ui.close();
                        }
                        if ui.button(localized(language, "Ventana normal", "Normal window")).clicked() {
                            self.custom_fullscreen = false;
                            self.fullscreen_configured = false;
                            ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(false));
                            ctx.send_viewport_cmd(egui::ViewportCommand::Decorations(true));
                            ctx.send_viewport_cmd(egui::ViewportCommand::Maximized(true));
                            ctx.request_repaint();
                            ui.close();
                        }
                    });

                    ui.menu_button(localized(language, "Ayuda", "Help"), |ui| {
                        ui.heading(localized(language, "Cómo funciona", "How it works"));
                        ui.label(localized(
                            language,
                            "Configura los parámetros y pulsa «Calcular». Cada caso aparece con un color propio en mapas y tablas.",
                            "Set the parameters and press “Calculate”. Each case gets its own color in maps and tables.",
                        ));
                        ui.separator();
                        ui.label(localized(
                            language,
                            "Alt + arrastrar: mover el mapa.\nCtrl + rueda: acercar o alejar.\nPulsa un caso para seleccionarlo.",
                            "Alt + drag: pan the map.\nCtrl + wheel: zoom in or out.\nSelect a case to inspect it.",
                        ));
                        ui.separator();
                        ui.label(localized(
                            language,
                            "Los JSON guardan la sesión completa y las tablas se exportan a CSV para Excel.",
                            "JSON files save the complete session and tables can be exported to CSV for Excel.",
                        ));
                        ui.separator();
                        ui.strong(localized(language, "Gu\u{00ed}a completa", "Complete guide"));
                        ui.label(localized(
                            language,
                            "Flujo: elige una sucesi\u{00f3}n, escribe sus par\u{00e1}metros y pulsa Enter. Pulsa Calcular para crear o actualizar el caso. Selecciona una fila para resaltarla en todos los mapas y ver sus datos.",
                            "Workflow: choose a sequence, enter its parameters, and press Enter. Press Calculate to create or update the case. Select a row to highlight it in every map and inspect its data.",
                        ));
                        ui.label(localized(
                            language,
                            "Mapas: Ctrl + rueda ampl\u{00ed}a o reduce; Alt + arrastrar desplaza. En Mandelbrot, el clic izquierdo a\u{00f1}ade c. El men\u{00fa} contextual de la tabla permite borrar casos y abrir el conjunto de Julia.",
                            "Maps: Ctrl + wheel zooms; Alt + drag pans. In Mandelbrot, left click adds c. The table context menu can delete cases and open the Julia set.",
                        ));
                        ui.label(localized(
                            language,
                            "An\u{00e1}lisis: t\u{00e9}rminos indica la cantidad calculada; tolerancia decide cu\u{00e1}ndo dos valores son iguales y se usa tambi\u{00e9}n para Mandelbrot; transitorio descarta valores iniciales; per\u{00ed}odo limita los ciclos buscados.",
                            "Analysis: terms sets the number calculated; tolerance decides when values are equal and is also used for Mandelbrot; transient discards initial values; period limits searched cycles.",
                        ));
                        ui.label(localized(
                            language,
                            "Dominios usan min,m\u{00e1}x,n. x e y son cartesianos; r y theta son polares. La cuadr\u{00ed}cula determina la discretizaci\u{00f3}n de trayectorias y atractores.",
                            "Domains use min,max,n. x and y are Cartesian; r and theta are polar. The grid determines trajectory and attractor discretization.",
                        ));
                        ui.label(localized(
                            language,
                            "Archivo permite importar JSON, exportar la sesi\u{00f3}n completa y exportar todas las tablas a CSV. Configuraci\u{00f3}n cambia idioma, tema claro u oscuro y carpeta de exportaci\u{00f3}n. Ver cambia el modo de ventana.",
                            "File imports JSON, exports the complete session, and exports all tables to CSV. Settings changes language and export folder. View changes the window mode.",
                        ));
                        ui.label(localized(
                            language,
                            "M\u{00fa}sica: criterio elige melod\u{00ed}a, armon\u{00ed}a, ritmo o textura; segmentaci\u{00f3}n usa bandas, posiciones o sectores; escala define las notas. Copiar prepara el patr\u{00f3}n y Abrir Strudel inicia el componente offline.",
                            "Music: criterion chooses melody, harmony, rhythm, or texture; segmentation uses bands, positions, or sectors; scale defines notes. Copy prepares the pattern and Open Strudel starts the offline component.",
                        ));
                    });

                    if self.custom_fullscreen {
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.add_sized(egui::vec2(26.0, 24.0), egui::Button::new("X").fill(app_accent_color())).on_hover_text(localized(language, "Cerrar", "Close")).clicked() {
                                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                            }
                            if ui.add_sized(egui::vec2(26.0, 24.0), egui::Button::new("□")).on_hover_text(localized(language, "Restaurar ventana", "Restore window")).clicked() {
                                self.custom_fullscreen = false;
                                self.fullscreen_configured = false;
                                ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(false));
                                ctx.send_viewport_cmd(egui::ViewportCommand::Decorations(true));
                                ctx.send_viewport_cmd(egui::ViewportCommand::Maximized(true));
                            }
                            if ui.add_sized(egui::vec2(26.0, 24.0), egui::Button::new("—")).on_hover_text(localized(language, "Minimizar", "Minimize")).clicked() {
                                ctx.send_viewport_cmd(egui::ViewportCommand::Minimized(true));
                            }
                        });
                    }
                });
            });

        egui::SidePanel::left("controls").resizable(true).show(ctx, |ui| {
            let language = self.language;
            ui.heading(localized(language, "Explorador", "Explorer"));
            ui.separator();
            ui.label(localized(language, "Sucesión", "Sequence"));
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.sequence, SequenceChoice::Cosine, "cos(x)");
                ui.selectable_value(&mut self.sequence, SequenceChoice::Logistic, "logística");
                ui.selectable_value(&mut self.sequence, SequenceChoice::Quadratic, "z^2+c");
            });
            ui.label(localized(language, "Coordenadas", "Coordinates"));
            ui.horizontal_wrapped(|ui| {
                if ui.selectable_value(&mut self.coordinates, CoordinateChoice::X, "x").clicked() {
                    self.main_plot_limits = None;
                    self.dirty = true;
                }
                if ui.selectable_value(
                    &mut self.coordinates,
                    CoordinateChoice::Cartesian,
                    "cartesiano",
                ).clicked() {
                    self.main_plot_limits = None;
                    self.dirty = true;
                }
                if ui.selectable_value(&mut self.coordinates, CoordinateChoice::Polar, "polar").clicked() {
                    self.main_plot_limits = None;
                    self.dirty = true;
                }
                if ui.selectable_value(&mut self.coordinates, CoordinateChoice::Y, "y").clicked() {
                    self.main_plot_limits = None;
                    self.dirty = true;
                }
                if ui.selectable_value(&mut self.coordinates, CoordinateChoice::Radius, "radio").clicked() {
                    self.main_plot_limits = None;
                    self.dirty = true;
                }
                if ui.selectable_value(&mut self.coordinates, CoordinateChoice::Angle, "ángulo").clicked() {
                    self.main_plot_limits = None;
                    self.dirty = true;
                }
            });
            ui.separator();
            ui.heading(localized(language, "Parámetros", "Parameters"));
            ui.separator();
            if self.sequence == SequenceChoice::Cosine {
                ui.label(RichText::new("x_{n+1} = cos(x_n)").strong());
            }
            let enter_x0 = text_row(ui, "x0", &mut self.x0);
            let enter_r = text_row(ui, "r", &mut self.r);
            let enter_c_real = text_row(ui, "c real", &mut self.c_real);
            let enter_c_imag = text_row(ui, "c imag", &mut self.c_imag);
            let enter_z0_real = text_row(ui, "z0 real", &mut self.z0_real);
            let enter_z0_imag = text_row(ui, "z0 imag", &mut self.z0_imag);
            if enter_r || enter_x0 {
                self.sync_from_logistic_parameters();
                self.dirty = true;
            } else if enter_c_real || enter_c_imag || enter_z0_real || enter_z0_imag {
                self.sync_from_quadratic_parameters();
                self.dirty = true;
            }
            if ui.button(localized(language, "Calcular", "Calculate")).clicked() {
                match self.calculate_current_case() {
                    Ok(()) => self.message = String::from("Caso calculado y guardado."),
                    Err(error) => self.message = format!("Error calculando: {error}"),
                }
            }

            if !self.message.is_empty() {
                ui.label(self.message.clone());
            }
            ui.separator();
            self.draw_music_panel(ctx, ui);
            ui.separator();
        });

        let language = self.language;
        egui::TopBottomPanel::bottom("bottom_panel")
            .exact_height(280.0)
            .show(ctx, |ui| {
                // Mostrar tabla (izquierda) y lista de casos (derecha) lado a lado
                ui.columns(2, |columns| {
                    columns[0].heading(localized(language, "Todos los puntos", "All points"));
                    columns[0].horizontal(|ui| {
                        ui.label(localized(language, "Mostrar:", "Show:"));
                        ui.selectable_value(&mut self.table_mode, TableMode::Sucesion, "sucesión");
                        ui.selectable_value(
                            &mut self.table_mode,
                            TableMode::Atractores,
                            "atractores",
                        );
                    });
                    let all_mode = self.table_mode;
                    self.draw_table(&mut columns[0], all_mode, false, "all_points_table");
                    columns[0].separator();
                    columns[0].heading(localized(
                        language,
                        "Punto seleccionado",
                        "Selected point",
                    ));
                    columns[0].horizontal(|ui| {
                        ui.label(localized(language, "Mostrar:", "Show:"));
                        ui.selectable_value(
                            &mut self.selected_table_mode,
                            TableMode::Sucesion,
                            "sucesión",
                        );
                        ui.selectable_value(
                            &mut self.selected_table_mode,
                            TableMode::Atractores,
                            "atractores",
                        );
                    });
                    if self.active_case_id.is_some() {
                        let selected_mode = self.selected_table_mode;
                        self.draw_table(
                            &mut columns[0],
                            selected_mode,
                            true,
                            "selected_point_table",
                        );
                    } else {
                        columns[0].label(localized(
                            language,
                            "No hay ningún punto seleccionado.",
                            "No point selected.",
                        ));
                    }
                    columns[1].horizontal(|ui| {
                        ui.heading(localized(language, "Casos", "Cases"));
                        if ui.button(localized(language, "Eliminar todo", "Delete all")).clicked() {
                            self.clear_all_cases();
                        }
                    });
                    // Ensure the cases panel stays to the right and doesn't overlap the table
                    // Make both columns compact to reduce overlap and visual weight
                    columns[0].set_min_width(220.0);
                    columns[1].set_min_width(180.0);
                    self.draw_cases_panel(&mut columns[1]);
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.columns(2, |columns| {
                columns[0].heading(localized(language, "Trayectorias y atractores", "Trajectories and attractors"));
                columns[0].separator();
                self.draw_main_plot(&mut columns[0]);

                columns[1].heading(localized(language, "Mapa de parámetros", "Parameter map"));
                columns[1].separator();
                self.draw_parameter_plot(&mut columns[1]);
            });
        });

        self.draw_julia_window(ctx, language);

    }
}

impl VisualExplorerApp {
    fn draw_point_actions_below_cases(&mut self, ui: &mut egui::Ui, language: Language) {
        self.point_actions_rect = None;
        if !self.case_context_menu_open {
            return;
        }
        let selected_count = self.selected_case_ids.len();
        let point_action = if selected_count > 1 {
            None
        } else {
            self.point_actions
        };
        if point_action.is_none() && selected_count == 0 {
            return;
        }

        let menu_response = egui::Frame::popup(ui.style()).show(ui, |ui| {
            if let Some((c_re, c_im)) = point_action {
                ui.label(format!("c = {:.8}{:+.8}j", c_re, c_im));
                if ui.button(localized(language, "Mostrar conjunto de Julia", "Show Julia set")).clicked() {
                    self.julia_point = Some((c_re, c_im));
                    self.julia_limits = (-2.0, 2.0, -2.0, 2.0);
                    self.julia_open = true;
                    self.point_actions = None;
                    self.point_actions_position = None;
                    self.case_context_menu_open = false;
                }
                if ui.button(localized(language, "Copiar coordenadas", "Copy coordinates")).clicked() {
                    if let Ok(mut clipboard) = arboard::Clipboard::new() {
                        let _ = clipboard.set_text(format!("{c_re:.12}{c_im:+.12}j"));
                    }
                    self.point_actions = None;
                    self.point_actions_position = None;
                    self.case_context_menu_open = false;
                }
                if selected_count == 1
                    && ui.button(localized(language, "Borrar el caso", "Delete case")).clicked()
                {
                    self.delete_selected_cases();
                }
            } else {
                ui.label(format!("{} {}", selected_count, localized(language, "casos seleccionados", "selected cases")));
                if ui.button(localized(language, "Copiar coordenadas", "Copy coordinates")).clicked() {
                    let coordinates = self
                        .cases
                        .iter()
                        .filter(|case| self.selected_case_ids.contains(&case.identifier))
                        .filter_map(|case| {
                            let re = case.recurrence.parameters.get("c_real")?;
                            let im = case.recurrence.parameters.get("c_imag")?;
                            Some(format!("{}: {re:.12}{im:+.12}j", case.label))
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    if let Ok(mut clipboard) = arboard::Clipboard::new() {
                        let _ = clipboard.set_text(coordinates);
                    }
                }
                if selected_count >= 1 {
                    let delete_label = if selected_count == 1 {
                        localized(language, "Borrar el caso", "Delete case")
                    } else {
                        localized(
                            language,
                            "Borrar casos seleccionados",
                            "Delete selected cases",
                        )
                    };
                    if ui.button(delete_label).clicked() {
                        self.delete_selected_cases();
                    }
                }
            }
            if ui.button(localized(language, "Cerrar", "Close")).clicked() {
                self.point_actions = None;
                self.point_actions_position = None;
                self.case_context_menu_open = false;
                self.selected_case_ids.clear();
                self.case_selection_anchor = None;
                self.active_case_id = None;
            }
        });
        self.point_actions_rect = Some(menu_response.response.rect);
    }

    fn draw_julia_window(&mut self, ctx: &egui::Context, language: Language) {
        let Some((c_re, c_im)) = self.julia_point else {
            return;
        };
        let mut open = self.julia_open;
        let julia_size = egui::vec2(720.0, 720.0);
        let centered_pos = ctx.available_rect().center() - julia_size / 2.0;
        egui::Window::new("")
            .open(&mut open)
            .default_pos(centered_pos)
            .default_size(julia_size)
            .resizable(true)
            .collapsible(false)
            .show(ctx, |ui| {
                ui.heading(localized(language, "Conjunto de Julia", "Julia set"));
                ui.separator();
                ui.label(format!("c = {:.8}{:+.8}j", c_re, c_im));
                if ui.button(localized(language, "Restablecer vista", "Reset view")).clicked() {
                    self.julia_limits = (-2.0, 2.0, -2.0, 2.0);
                }
                let square_size = ui.available_width().min(ui.available_height()).max(1.0);
                let limits = self.julia_limits;
                let response = egui_plot::Plot::new("julia_plot")
                    .width(square_size)
                    .height(square_size)
                    .view_aspect(1.0)
                    .allow_zoom(true)
                    .allow_drag(true)
                    .allow_scroll(true)
                    .show_axes(true)
                    .show(ui, |plot_ui| {
                        plot_ui.set_plot_bounds(PlotBounds::from_min_max(
                            [limits.0, limits.2],
                            [limits.1, limits.3],
                        ));
                    });
                let julia_bounds = response.transform.bounds();
                let julia_min = julia_bounds.min();
                let julia_max = julia_bounds.max();
                self.julia_limits = (
                    julia_min[0],
                    julia_max[0],
                    julia_min[1],
                    julia_max[1],
                );
                let frame = *response.transform.frame();
                let callback = WgpuCallback::new_paint_callback(
                    frame,
                    MandelbrotCallback::new_julia(
                        [limits.0 as f32, limits.2 as f32],
                        [limits.1 as f32, limits.3 as f32],
                        [c_re as f32, c_im as f32],
                        frame,
                        self.target_format,
                    ),
                );
                ui.painter().add(Shape::Callback(callback));
            });
        self.julia_open = open;
    }

    fn draw_main_plot(&mut self, ui: &mut egui::Ui) {
        let Some(reference) = self.runs.first() else {
            ui.label("Sesión vacía");
            return;
        };
        ui.horizontal(|ui| {
            if ui.button("Restablecer vista").clicked() {
                self.main_plot_limits = None;
                self.main_plot_pan_start = None;
                self.main_plot_pan_limits_start = None;
            }
            ui.label("(Alt+arrastrar para desplazar, Ctrl+rueda para zoom)");
        });
        let axes = reference.result.transformer.coordinate_system.axes();
        let is_polar = axes == ["r", "theta"];
        let limits = if let Some(limits) = self.main_plot_limits {
            limits
        } else if let Some(limits) = self.cached_main_auto_limits {
            limits
        } else {
            let limits = self.default_main_plot_limits(axes, is_polar);
            self.cached_main_auto_limits = Some(limits);
            limits
        };
        // Cuadrícula uniforme y tenue en todos los planos, igual que en el
        // mapa logístico, para que no tape las trayectorias ni los puntos.
        let grid_color = Color32::from_rgba_unmultiplied(180, 180, 180, 45);
        // Redondear a píxeles completos evita que el layout alterne entre dos
        // tamaños fraccionarios y produzca un pequeño temblor visual.
        let square_size = ui
            .available_width()
            .min(ui.available_height())
            .max(1.0)
            .floor();
        let mut plot = Plot::new("main_plot")
            .width(square_size)
            .height(square_size)
            .view_aspect(1.0)
            .allow_boxed_zoom(false)
            .allow_zoom(false)
            .allow_drag(false)
            .allow_scroll(false)
            .show_axes(!is_polar)
            .label_formatter(move |name, value| {
                if is_polar {
                    format!(
                        "{name}\n(r = {:.3}, theta = {:.3} rad)",
                        value.x, value.y
                    )
                } else {
                    format!("{name}\n({:.3}, {:.3})", value.x, value.y)
                }
            });

        if is_polar {
            plot = plot.data_aspect(1.0);
        } else if axes.len() == 2 {
            plot = plot.data_aspect(1.0);
        }

        if is_polar {
            // En polar, dibujamos la trayectoria en el plano (x,y) = (r*cos, r*sin)
            // para que el "plano polar" sea visible con círculos radiales.
            let response = plot.show(ui, |plot_ui| {
                plot_ui.set_plot_bounds(PlotBounds::from_min_max(
                    [limits.0, limits.2],
                    [limits.1, limits.3],
                ));
                let r_subdivision = &reference.result.transformer.grid.subdivisions["r"];
                let theta_subdivision = &reference.result.transformer.grid.subdivisions["theta"];
                let r_limit = r_subdivision.range.maximum;

                // Retícula polar basada en las divisiones reales de r y theta.
                for radius in r_subdivision.edges().into_iter().filter(|radius| *radius > 0.0) {
                    let circle = (0..=120)
                        .map(|i| {
                            let a = i as f64 * std::f64::consts::TAU / 120.0;
                            [radius * a.cos(), radius * a.sin()]
                        })
                        .collect::<Vec<_>>();
                    plot_ui.line(
                        Line::new(format!("r={radius:.3}"), circle)
                            .color(Color32::from_gray(220))
                            .width(0.5_f32),
                    );
                }
                for theta in theta_subdivision
                    .edges()
                    .into_iter()
                    .take(theta_subdivision.steps)
                {
                    let label_radius = r_limit * 1.08;
                    plot_ui.line(
                        Line::new(
                            format!("theta={theta:.3}"),
                            vec![[0.0, 0.0], [r_limit * theta.cos(), r_limit * theta.sin()]],
                        )
                        .color(Color32::from_gray(200))
                        .width(0.5_f32),
                    );
                    let degrees = theta.to_degrees().rem_euclid(360.0);
                    plot_ui.text(
                        Text::new(
                            format!("angulo_{theta:.6}"),
                            PlotPoint::new(
                                label_radius * theta.cos(),
                                label_radius * theta.sin(),
                            ),
                            format!("{degrees:.0}°\n{theta:.2} rad"),
                        )
                        .color(Color32::from_gray(190)),
                    );
                }
                for (index, run) in self.runs.iter().enumerate() {
                    let color = self.color_for_case(&run.case.identifier);
                    let width = if run.is_active { 2.0_f32 } else { 1.0_f32 };
                    let points = &run.trajectory_points;
                    if points.is_empty() {
                        continue;
                    }
                    plot_ui.line(
                        Line::new(run.case.label.clone(), points.as_slice())
                            .width(width)
                            .color(color),
                    );
                    plot_ui.points(
                        Points::new(format!("puntos_{index}"), points.as_slice())
                            .radius(if run.is_active { 2.5_f32 } else { 1.5_f32 })
                            .color(color),
                    );
                    for (attractor_index, attr_points) in run.attractor_points.iter().enumerate() {
                        if !attr_points.is_empty() {
                            plot_ui.points(
                                Points::new(format!("atractor_{index}_{attractor_index}"), attr_points.as_slice())
                                    .shape(MarkerShape::Diamond)
                                    .radius(4.0_f32)
                                    .color(color),
                            );
                        }
                    }
                }
            });
            self.handle_main_plot_interaction(ui, &response, limits);
            return;
        }

        let response = plot.show(ui, |plot_ui| {
            plot_ui.set_plot_bounds(PlotBounds::from_min_max(
                [limits.0, limits.2],
                [limits.1, limits.3],
            ));
            if axes.len() == 1 {
                let axis = axes[0];
                let subdivision = &reference.result.transformer.grid.subdivisions[axis];
                for edge in subdivision.edges() {
                        plot_ui.hline(
                            HLine::new(format!("{axis}={edge:.3}"), edge)
                            .color(grid_color)
                            .width(0.5_f32),
                    );
                }
            } else {
                let x_axis = axes[0];
                let y_axis = axes[1];
                let x_subdivision = &reference.result.transformer.grid.subdivisions[x_axis];
                let y_subdivision = &reference.result.transformer.grid.subdivisions[y_axis];
                for edge in x_subdivision.edges() {
                    plot_ui.vline(
                        VLine::new(format!("{x_axis}={edge:.3}"), edge)
                            .color(grid_color)
                            .width(0.5_f32),
                    );
                }
                for edge in y_subdivision.edges() {
                    plot_ui.hline(
                        HLine::new(format!("{y_axis}={edge:.3}"), edge)
                            .color(grid_color)
                            .width(0.5_f32),
                    );
                }
            }
            for (index, run) in self.runs.iter().enumerate() {
                let width = if run.is_active { 2.0_f32 } else { 1.0_f32 };
                let color = self.color_for_case(&run.case.identifier);
                if axes.len() == 1 {
                    let points = &run.trajectory_points;
                    plot_ui.line(
                        Line::new(run.case.label.clone(), points.as_slice())
                            .width(width)
                            .color(color),
                    );
                    plot_ui.points(
                        Points::new(format!("puntos_{index}"), points.as_slice())
                            .radius(if run.is_active { 2.5_f32 } else { 1.5_f32 })
                            .color(color),
                    );
                } else {
                    let points = &run.trajectory_points;
                    plot_ui.line(
                        Line::new(run.case.label.clone(), points.as_slice())
                            .width(width)
                            .color(color),
                    );
                    plot_ui.points(
                        Points::new(format!("puntos_{index}"), points.as_slice())
                            .radius(if run.is_active { 2.5_f32 } else { 1.5_f32 })
                            .color(color),
                    );
                }
                // Atractores como diamantes.
                for (attractor_index, attr_points) in run.attractor_points.iter().enumerate() {
                    if !attr_points.is_empty() {
                        plot_ui.points(
                            Points::new(format!("atractor_{index}_{attractor_index}"), attr_points.as_slice())
                                .shape(MarkerShape::Diamond)
                                .radius(4.0_f32)
                                .color(color),
                        );
                    }
                }
                // Punto inicial como círculo con borde.
            }
        });
        self.handle_main_plot_interaction(ui, &response, limits);
    }

    fn draw_parameter_plot(&mut self, ui: &mut egui::Ui) {
        match self.sequence {
            SequenceChoice::Cosine => {
                ui.label("cos(x): sin mapa de parámetro");
            }
            SequenceChoice::Logistic => {
                let mut clicked: Option<PlotPoint> = None;
                let limits = self
                    .map_limits
                    .get("logistica")
                    .copied()
                    .unwrap_or(MAP_BOUNDS_LOGISTIC);
                
                // Controls for the Logistic map: reset + hints
                ui.horizontal(|ui| {
                    if ui.button("Restablecer vista").clicked() {
                        self.map_limits.insert(String::from("logistica"), MAP_BOUNDS_LOGISTIC);
                        self.parameter_pan_start = None;
                        self.parameter_pan_limits_start = None;
                        self.dirty = true;
                    }
                    ui.label("(Alt+arrastrar para desplazar, Ctrl+rueda para zoom)");
                });
                let square_size = ui.available_width().min(ui.available_height()).max(1.0);
                let response = Plot::new("bifurcation_plot")
                    .width(square_size)
                    .height(square_size)
                    .view_aspect(1.0)
                    .allow_zoom(false)
                    .allow_drag(false)
                    .allow_scroll(false)
                    .allow_boxed_zoom(false)
                    .show_grid(false)
                    .show(ui, |plot_ui| {
                        // Fijar la vista antes de dibujar cualquier elemento evita
                        // que los nuevos puntos modifiquen la escala de la cuadrícula.
                        plot_ui.set_plot_bounds(PlotBounds::from_min_max(
                            [limits.0, limits.2],
                            [limits.1, limits.3],
                        ));
                        let grid_color = Color32::from_rgba_unmultiplied(180, 180, 180, 45);
                        // Mantener una cuadrícula densa, como la original:
                        // subdivisiones de 0.2 en r y de 0.05 en x₀.
                        for r_tick in 0..=20 {
                            let r_value = r_tick as f64 / 5.0;
                            plot_ui.vline(
                                VLine::new(format!("grid_r_{r_tick}"), r_value)
                                    .color(grid_color),
                            );
                        }
                        for x_tick in 0..=20 {
                            let x_value = x_tick as f64 / 20.0;
                            plot_ui.hline(
                                HLine::new(format!("grid_x_{x_tick}"), x_value)
                                    .color(grid_color),
                            );
                        }

                        plot_ui.points(
                            Points::new("bif", self.bifurcation_points.as_slice())
                                .radius(0.6_f32)
                                .color(Color32::from_rgba_unmultiplied(255, 255, 255, 110)),
                        );

                        // Un caso cuadrático real también puede representarse en el plano logístico.
                        for (idx, run) in self.runs.iter().enumerate() {
                            if let Some((r_value, x_value)) = self.logistic_coordinates_for_run(run) {
                                let base_color = self.color_for_case(&run.case.identifier);
                                let color = Color32::from_rgba_unmultiplied(
                                    base_color.r(),
                                    base_color.g(),
                                    base_color.b(),
                                    255,
                                );
                                plot_ui.points(
                                    if self.active_case_id.as_ref() == Some(&run.case.identifier) {
                                        Points::new(
                                            format!("selected_case_log_{idx}"),
                                            vec![[r_value, x_value]],
                                        )
                                        .radius(7.0_f32)
                                        .filled(false)
                                        .color(Color32::from_rgba_unmultiplied(255, 255, 255, 210))
                                    } else {
                                        Points::new(format!("case_log_{idx}"), vec![[r_value, x_value]])
                                            .radius(4.0_f32)
                                            .filled(true)
                                            .color(color)
                                    },
                                );
                                if self.active_case_id.as_ref() == Some(&run.case.identifier) {
                                    plot_ui.points(
                                        Points::new(
                                            format!("case_log_{idx}"),
                                            vec![[r_value, x_value]],
                                        )
                                        .radius(4.0_f32)
                                        .filled(true)
                                        .color(color),
                                    );
                                }
                            }
                        }
                        if plot_ui.response().clicked() {
                            let _ = plot_ui.pointer_coordinate();
                        }
                    });
                // Manejar pan (Alt+arrastrar) y el clic/zoom de forma manual con la rueda (Ctrl+rueda para zoom).
                let input = ui.input(|i| i.clone());

                // Inicio de pan si se presiona dentro del rect del plot mientras Alt está pulsado
                if input.modifiers.alt {
                    if input.pointer.any_pressed() {
                        if let Some(origin) = input.pointer.press_origin() {
                            if response.response.rect.contains(origin) {
                                self.parameter_pan_start = Some(origin);
                                self.parameter_pan_limits_start = Some(limits);
                            }
                        }
                    }
                }

                // Si estamos en modo pan y el puntero se mueve, actualizar limites según delta en valores.
                if let (Some(start_pos), Some(start_limits)) = (self.parameter_pan_start, self.parameter_pan_limits_start) {
                    if input.pointer.any_down() {
                        if let Some(current_pos) = input.pointer.hover_pos() {
                            let start_val = response.transform.value_from_position(start_pos);
                            let current_val = response.transform.value_from_position(current_pos);
                            let dx = current_val.x - start_val.x;
                            let dy = current_val.y - start_val.y;
                            let new_limits = (
                                start_limits.0 - dx,
                                start_limits.1 - dx,
                                start_limits.2 - dy,
                                start_limits.3 - dy,
                            );
                            self.map_limits.insert(String::from("logistica"), new_limits);
                            self.dirty = true;
                        }
                    } else {
                        // Soltado: limpiar estado de pan
                        self.parameter_pan_start = None;
                        self.parameter_pan_limits_start = None;
                    }
                }

                // Zoom con rueda: requerir Ctrl/Command para evitar zoom accidental
                if let Some(pointer) = response.response.hover_pos() {
                    let value = response.transform.value_from_position(pointer);
                    let scroll = input.raw_scroll_delta.y;
                    if scroll != 0.0 && (input.modifiers.ctrl || input.modifiers.command) {
                        let step = if scroll > 0.0 { 1 } else { -1 };
                        let scale = if step > 0 { 1.0 / ZOOM_SCALE } else { ZOOM_SCALE };
                        self.map_limits.insert(
                            String::from("logistica"),
                            self.zoom_limits(limits, (value.x, value.y), scale, MAP_BOUNDS_LOGISTIC),
                        );
                        self.dirty = true;
                    }
                }
                if response.response.clicked() {
                    if let Some(hp) = response.response.hover_pos() {
                        clicked = Some(response.transform.value_from_position(hp).into());
                        // Diagnostic log: hover pixel and data coordinates
                        let data = response.transform.value_from_position(hp);
                        let _pixel_pos = response.transform.position_from_point(&PlotPoint::new(data.x, data.y));
                        // Click recorded for logistic; stored into cases. (logs removed)
                    }
                }
                if let Some(point) = clicked {
                    let r_value = point.x.clamp(0.0, 4.0);
                    let x_value = point.y.clamp(0.0, 1.0);
                    let _ = self.store_logistic_pair(r_value, x_value, "bifurcation_click", true);
                    self.message = String::from("Punto logístico seleccionado.");
                    // Recalculate immediately to show the new point
                    self.dirty = true;
                    if let Err(e) = self.recalculate() {
                        self.message = format!("Error recalculando: {e}");
                    }
                }
            }
            SequenceChoice::Quadratic => {
                let mut clicked: Option<PlotPoint> = None;
                let raw_limits = self
                    .map_limits
                    .get("z^2+c")
                    .copied()
                    .unwrap_or(MAP_BOUNDS_MANDELBROT);
                // Make the initial bounds square around the center to avoid visual stretching.
                let x_center = (raw_limits.0 + raw_limits.1) / 2.0;
                let y_center = (raw_limits.2 + raw_limits.3) / 2.0;
                let x_span = raw_limits.1 - raw_limits.0;
                let y_span = raw_limits.3 - raw_limits.2;
                let span = x_span.max(y_span);
                let limits = (
                    x_center - span / 2.0,
                    x_center + span / 2.0,
                    y_center - span / 2.0,
                    y_center + span / 2.0,
                );
                // Controls for the Mandelbrot map: reset + hints
                ui.horizontal(|ui| {
                    if ui.button("Restablecer vista").clicked() {
                        self.map_limits.insert(String::from("z^2+c"), MAP_BOUNDS_MANDELBROT);
                        self.mandelbrot_pan_start = None;
                        self.mandelbrot_pan_limits_start = None;
                        self.dirty = true;
                    }
                    ui.label("(Alt+arrastrar para desplazar, Ctrl+rueda para zoom)");
                });

                let square_size = ui.available_width().min(ui.available_height()).max(1.0);
                let response = Plot::new("mandelbrot_plot")
                    .width(square_size)
                    .height(square_size)
                    .view_aspect(1.0)
                    .allow_zoom(false)
                    .allow_drag(false)
                    .allow_scroll(false)
                    .allow_boxed_zoom(false)
                    .data_aspect(1.0)
                    .show(ui, |plot_ui| {
                        plot_ui.set_plot_bounds(PlotBounds::from_min_max(
                            [limits.0, limits.2],
                            [limits.1, limits.3],
                        ));
                        if plot_ui.response().clicked() {
                            let _ = plot_ui.pointer_coordinate();
                        }
                    });

                // Dibujar el conjunto de Mandelbrot con el renderer GPU, alineado
                // con el marco del Plot (el rectangulo de datos, sin los ejes).
                let frame = *response.transform.frame();
                // No overlays in shader: draw selected marker with painter to match plot colors.
                let overlays: Vec<[f32; 2]> = Vec::new();
                let callback = WgpuCallback::new_paint_callback(
                    frame,
                    MandelbrotCallback::new(
                        [limits.0 as f32, limits.2 as f32],
                        [limits.1 as f32, limits.3 as f32],
                        overlays,
                        frame,
                        self.target_format,
                    ),
                );
                ui.painter().add(Shape::Callback(callback));

                // Mostrar solo los puntos creados explícitamente mediante clic en el Mandelbrot.
                // Los casos auxiliares creados desde la logística no son marcadores del plano.
                for run in &self.runs {
                    if let Some((c_re, c_im)) = self.mandelbrot_coordinate_for_run(run) {
                        let pos = response
                            .transform
                            .position_from_point(&PlotPoint::new(c_re, c_im));
                        let color = self.color_for_case(&run.case.identifier);
                        if run.mandelbrot_member.unwrap_or(true) {
                            ui.painter().circle_filled(pos, 4.0, color);
                            ui.painter().circle_stroke(
                                pos,
                                4.0,
                                egui::Stroke::new(1.0_f32, Color32::BLACK),
                            );
                        } else {
                            let outside_color = Color32::from_rgb(255, 90, 45);
                            ui.painter().circle_stroke(
                                pos,
                                5.0,
                                egui::Stroke::new(1.8_f32, outside_color),
                            );
                            ui.painter().line_segment(
                                [pos + egui::vec2(-4.0, -4.0), pos + egui::vec2(4.0, 4.0)],
                                egui::Stroke::new(1.2_f32, outside_color),
                            );
                            ui.painter().line_segment(
                                [pos + egui::vec2(-4.0, 4.0), pos + egui::vec2(4.0, -4.0)],
                                egui::Stroke::new(1.2_f32, outside_color),
                            );
                        }
                        if self.active_case_id.as_ref() == Some(&run.case.identifier) {
                            ui.painter().circle_stroke(
                                pos,
                                7.0,
                                egui::Stroke::new(
                                    1.5_f32,
                                    Color32::from_rgba_unmultiplied(255, 255, 255, 210),
                                ),
                            );
                        }
                    }
                }

                // Manejar pan (Alt+arrastrar) y el clic/zoom de forma manual con la rueda.
                let input = ui.input(|i| i.clone());

                // Inicio de pan si se presiona dentro del rect del plot mientras Alt está pulsado
                if input.modifiers.alt {
                    if input.pointer.any_pressed() {
                        if let Some(origin) = input.pointer.press_origin() {
                            if response.response.rect.contains(origin) {
                                self.mandelbrot_pan_start = Some(origin);
                                self.mandelbrot_pan_limits_start = Some(limits);
                            }
                        }
                    }
                }

                // Si estamos en modo pan y el puntero se mueve, actualizar limites según delta en valores.
                if let (Some(start_pos), Some(start_limits)) = (self.mandelbrot_pan_start, self.mandelbrot_pan_limits_start) {
                    if input.pointer.any_down() {
                        if let Some(current_pos) = input.pointer.hover_pos() {
                            let start_val = response.transform.value_from_position(start_pos);
                            let current_val = response.transform.value_from_position(current_pos);
                            let dx = current_val.x - start_val.x;
                            let dy = current_val.y - start_val.y;
                            let new_limits = (
                                start_limits.0 - dx,
                                start_limits.1 - dx,
                                start_limits.2 - dy,
                                start_limits.3 - dy,
                            );
                            self.map_limits.insert(String::from("z^2+c"), new_limits);
                            self.dirty = true;
                        }
                    } else {
                        // Soltado: limpiar estado de pan
                        self.mandelbrot_pan_start = None;
                        self.mandelbrot_pan_limits_start = None;
                    }
                }

                if let Some(pointer) = response.response.hover_pos() {
                    let value = response.transform.value_from_position(pointer);
                    let scroll = input.raw_scroll_delta.y;
                    // Requerir Ctrl (o Command en macOS) para hacer zoom con la rueda.
                    if scroll != 0.0 && (input.modifiers.ctrl || input.modifiers.command) {
                        let step = if scroll > 0.0 { 1 } else { -1 };
                        let scale = if step > 0 { 1.0 / ZOOM_SCALE } else { ZOOM_SCALE };
                        let new_limits = self.zoom_limits(limits, (value.x, value.y), scale, MAP_BOUNDS_MANDELBROT);
                        // Mantener los limites cuadrados tras el zoom.
                        let x_center = (new_limits.0 + new_limits.1) / 2.0;
                        let y_center = (new_limits.2 + new_limits.3) / 2.0;
                        let x_span = new_limits.1 - new_limits.0;
                        let y_span = new_limits.3 - new_limits.2;
                        let span = x_span.max(y_span);
                        let square_limits = (
                            x_center - span / 2.0,
                            x_center + span / 2.0,
                            y_center - span / 2.0,
                            y_center + span / 2.0,
                        );
                        self.map_limits.insert(String::from("z^2+c"), square_limits);
                        self.dirty = true;
                    }
                }
                if response.response.clicked() {
                    if let Some(hp) = response.response.hover_pos() {
                        // Usar el mismo marco y la misma orientación que el
                        // shader del Mandelbrot, evitando desajustes del Plot.
                        let frame = *response.transform.frame();
                        let rel_x = ((hp.x - frame.min.x) / frame.width()).clamp(0.0, 1.0);
                        let rel_y = ((hp.y - frame.min.y) / frame.height()).clamp(0.0, 1.0);
                        clicked = Some(
                            PlotPoint::new(
                                limits.0 + rel_x as f64 * (limits.1 - limits.0),
                                limits.3 - rel_y as f64 * (limits.3 - limits.2),
                            )
                            .into(),
                        );
                        // Click recorded for mandelbrot; stored into cases. (logs removed)
                    }
                }
                if let Some(point) = clicked {
                    let c_re = point.x.clamp(-2.0, 1.0);
                    let raw_c_im = point.y.clamp(-1.35, 1.35);
                    // Un clic sobre el eje real puede caer unas décimas de
                    // píxel por encima o por debajo. Ajustarlo evita perder
                    // falsos incompatibles con la logística.
                    let frame_height = response.transform.frame().height().max(1.0) as f64;
                    let pixel_tolerance = (limits.3 - limits.2) / frame_height * 1.5;
                    let c_im = if raw_c_im.abs() <= pixel_tolerance {
                        0.0
                    } else {
                        raw_c_im
                    };
                    if !is_mandelbrot_member(c_re, c_im) {
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
                            "mandelbrot_click_outside",
                            BTreeMap::from([(
                                String::from("mandelbrot_member"),
                                String::from("false"),
                            )]),
                            true,
                            true,
                        );
                        self.message = String::from(
                            "Punto añadido: está fuera del conjunto de Mandelbrot.",
                        );
                        self.message = String::from(
                            "Ese punto está fuera del conjunto de Mandelbrot y no se ha añadido.",
                        );
                        self.dirty = true;
                        if let Err(e) = self.recalculate() {
                            self.message = format!("Error recalculando: {e}");
                        }
                        self.message = String::from(
                            "Punto añadido: está fuera del conjunto de Mandelbrot.",
                        );
                    } else {
                        let complex_case = self.store_case(
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
                        if quadratic_to_logistic(
                            Complex64::new(c_re, c_im),
                            Complex64::new(0.0, 0.0),
                            self.tolerance_value(),
                        ).is_some() {
                            if complex_case.is_ok() {
                                self.message = String::from(
                                    "Punto complejo seleccionado; su equivalente se muestra en la tabla.",
                                );
                            }
                        } else {
                            self.message = String::from(
                                "Punto complejo seleccionado; no tiene equivalente logístico real.",
                            );
                        }
                        self.set_transient_message("Punto complejo seleccionado.");
                        // Recalculate immediately to show the new point.
                        self.dirty = true;
                        if let Err(e) = self.recalculate() {
                            self.message = format!("Error recalculando: {e}");
                        }
                    }
                }
            }
        }
    }

    fn logistic_coordinates_for_run(&self, run: &CaseRun) -> Option<(f64, f64)> {
        match run.case.recurrence.name.as_str() {
            "logistic" => Some((
                *run.case.recurrence.parameters.get("r")?,
                *run.case.recurrence.parameters.get("x0")?,
            )),
            "quadratic_complex" => quadratic_to_logistic(
                Complex64::new(
                    *run.case.recurrence.parameters.get("c_real")?,
                    *run.case.recurrence.parameters.get("c_imag")?,
                ),
                Complex64::new(
                    *run.case.recurrence.parameters.get("z0_real")?,
                    *run.case.recurrence.parameters.get("z0_imag")?,
                ),
                self.tolerance_value(),
            )
            .and_then(|converted| {
                // Evitar que un redondeo mínimo deje el punto justo fuera
                // del marco visible de la bifurcación.
                if (-self.tolerance_value()
                    ..=4.0 + self.tolerance_value())
                    .contains(&converted.r)
                    && (-self.tolerance_value()
                        ..=1.0 + self.tolerance_value())
                        .contains(&converted.x0)
                {
                    Some((converted.r.clamp(0.0, 4.0), converted.x0.clamp(0.0, 1.0)))
                } else {
                    None
                }
            }),
            _ => None,
        }
    }

    fn mandelbrot_coordinate_for_run(&self, run: &CaseRun) -> Option<(f64, f64)> {
        match run.case.recurrence.name.as_str() {
            "quadratic_complex" => Some((
                *run.case.recurrence.parameters.get("c_real")?,
                *run.case.recurrence.parameters.get("c_imag")?,
            )),
            "logistic" => {
                let converted = logistic_to_quadratic(
                    *run.case.recurrence.parameters.get("r")?,
                    *run.case.recurrence.parameters.get("x0")?,
                );
                Some((converted.c.re, converted.c.im))
            }
            _ => None,
        }
    }

    #[allow(dead_code)]
    fn draw_summary(&self, ui: &mut egui::Ui) {
        ui.label(format!(
            "Casos calculados: {} | sistema: {}",
            self.runs.len(),
            self.coordinates_name()
        ));
        for run in self.runs.iter().take(5) {
            let color = self.color_for_case(&run.case.identifier);
            let behavior = match run.result.sequence_result.behavior {
                BehaviorKind::FixedPoint => "punto fijo",
                BehaviorKind::Periodic => "periódica",
                BehaviorKind::Divergent => "divergente",
                BehaviorKind::ChaoticOrUnresolved => "caótica/no resuelta",
                BehaviorKind::InsufficientData => "datos insuficientes",
                BehaviorKind::CalculationError => "error de calculo",
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
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new("●").color(color));
                ui.monospace(format!(
                    "{}: {}; {}; n={}",
                    run.case.label,
                    behavior,
                    attractors,
                    run.result.sequence_result.terms.len()
                ));
                let parameters = run
                    .case
                    .recurrence
                    .parameters
                    .iter()
                    .map(|(name, value)| format!("{name}={value:.5}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                if !parameters.is_empty() {
                    ui.label(RichText::new(format!("[{parameters}]")).weak());
                }
            });
        }
    }

    fn draw_cases_panel(&mut self, ui: &mut egui::Ui) {
        self.cases_panel_rect = Some(ui.max_rect());
        let mut selected: Option<(usize, String, RecurrenceConfig, String, bool, bool)> = None;
        // Reserva espacio para las acciones debajo de la lista, de modo que el
        // menú nunca se dibuje encima de un punto del mapa.
        let actions_height = if self.point_actions.is_some() || !self.selected_case_ids.is_empty() {
            92.0
        } else {
            0.0
        };
        let cases_height = (ui.available_height() - actions_height).max(1.0);
        egui::ScrollArea::vertical()
            .id_salt("cases_scroll")
                    .max_height(cases_height)
                    .min_scrolled_height(cases_height)
                    .show(ui, |ui| {
                for (index, case) in self
                    .cases
                    .iter()
                    .enumerate()
                {
                    let is_selected = self.selected_case_ids.contains(&case.identifier);
                    let color = self.color_for_case(&case.identifier);
                    let parameters = case
                        .recurrence
                        .parameters
                        .iter()
                        .map(|(name, value)| format!("{name}={value:.5}"))
                        .collect::<Vec<_>>()
                        .join(", ");
                    let equivalent = if case.recurrence.name == "quadratic_complex" {
                        quadratic_to_logistic(
                            Complex64::new(
                                case.recurrence.parameters.get("c_real").copied().unwrap_or(0.0),
                                case.recurrence.parameters.get("c_imag").copied().unwrap_or(0.0),
                            ),
                            Complex64::new(
                                case.recurrence.parameters.get("z0_real").copied().unwrap_or(0.0),
                                case.recurrence.parameters.get("z0_imag").copied().unwrap_or(0.0),
                            ),
                            self.tolerance_value(),
                        )
                        .map(|converted| {
                            format!(
                                " (equiv. logístico: r={:.5}, x₀={:.5})",
                                converted.r, converted.x0
                            )
                        })
                    } else {
                        None
                    };
                    let outside_mandelbrot = case.recurrence.name == "quadratic_complex"
                        && !self
                            .runs
                            .iter()
                            .find(|run| run.case.identifier == case.identifier)
                            .and_then(|run| run.mandelbrot_member)
                            .unwrap_or(true);
                    ui.horizontal_wrapped(|ui| {
                        ui.label(RichText::new("●").color(color));
                        let case_response = ui.selectable_label(is_selected, &case.label);
                        if case_response.clicked_by(egui::PointerButton::Primary) {
                            // Un clic izquierdo en la tabla cierra el menú
                            // contextual anterior antes de seleccionar la fila.
                            let close_context = self.point_actions.is_some();
                            self.point_actions = None;
                            self.point_actions_position = None;
                            if close_context {
                                self.selected_case_ids.clear();
                                self.case_selection_anchor = None;
                                self.active_case_id = None;
                            } else {
                                let modifiers = ui.input(|input| input.modifiers);
                                selected = Some((
                                    index,
                                    case.identifier.clone(),
                                    case.recurrence.clone(),
                                    case.label.clone(),
                                    modifiers.ctrl || modifiers.command,
                                    modifiers.shift,
                                ));
                            }
                        }
                        if case_response.clicked_by(egui::PointerButton::Secondary)
                            && !case_response.clicked_by(egui::PointerButton::Primary)
                        {
                            self.case_context_menu_open = true;
                            if !self.selected_case_ids.contains(&case.identifier) {
                                self.selected_case_ids.clear();
                                self.case_selection_anchor = Some(index);
                                self.selected_case_ids.insert(case.identifier.clone());
                                self.active_case_id = Some(case.identifier.clone());
                            }
                            if case.recurrence.name == "quadratic_complex" {
                                let c_re = case
                                    .recurrence
                                    .parameters
                                    .get("c_real")
                                    .copied()
                                    .unwrap_or(0.0);
                                let c_im = case
                                    .recurrence
                                    .parameters
                                    .get("c_imag")
                                    .copied()
                                    .unwrap_or(0.0);
                                self.point_actions = Some((c_re, c_im));
                                self.point_actions_position = None;
                            }
                        }
                        if !parameters.is_empty() {
                            ui.label(
                                RichText::new(format!(
                                    "[{parameters}]{}",
                                    equivalent.as_deref().unwrap_or("")
                                ))
                                .weak(),
                            );
                        }
                        if outside_mandelbrot {
                            ui.label(
                                RichText::new("⚠ fuera de Mandelbrot")
                                    .color(Color32::from_rgb(255, 100, 55)),
                            );
                        }
                    });
                }
            });
        if let Some((index, identifier, recurrence, label, ctrl, shift)) = selected {
            if shift {
                let anchor = self.case_selection_anchor.unwrap_or(index);
                let start = anchor.min(index);
                let end = anchor.max(index);
                self.selected_case_ids = self.cases[start..=end]
                    .iter()
                    .map(|case| case.identifier.clone())
                    .collect();
                self.case_selection_anchor = Some(index);
            } else if ctrl {
                if !self.selected_case_ids.insert(identifier.clone()) {
                    self.selected_case_ids.remove(&identifier);
                }
                self.case_selection_anchor = Some(index);
            } else {
                self.selected_case_ids.clear();
                self.selected_case_ids.insert(identifier.clone());
                self.case_selection_anchor = Some(index);
            }
            self.active_case_id = self
                .selected_case_ids
                .contains(&identifier)
                .then_some(identifier.clone())
                .or_else(|| self.selected_case_ids.iter().next().cloned());
            for run in &mut self.runs {
                run.is_active = self.active_case_id.as_ref() == Some(&run.case.identifier);
            }
            if self.active_case_id.as_ref() == Some(&identifier) {
                self.apply_recurrence(&recurrence);
            }
            self.table_cache
                .retain(|key, _| !key.starts_with("selected_point_table:"));
            self.message = format!("Caso activo: {label}");
        }
        self.draw_point_actions_below_cases(ui, self.language);
    }

    fn draw_music_panel(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        let language = self.language;
        let music_before = (
            self.music_criterion.clone(),
            self.music_segmentation.clone(),
            self.music_scale.clone(),
        );
        ui.heading("Strudel");
        ui.separator();
        ui.label(localized(language, "Criterio", "Criterion"));
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(&mut self.music_criterion, String::from("melody"), "melodía");
            ui.selectable_value(
                &mut self.music_criterion,
                String::from("harmony"),
                "armonía",
            );
            ui.selectable_value(&mut self.music_criterion, String::from("rhythm"), "ritmo");
            ui.selectable_value(
                &mut self.music_criterion,
                String::from("texture"),
                "textura",
            );
        });
        ui.label(localized(language, "Mapa", "Map"));
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(
                &mut self.music_segmentation,
                String::from("bands"),
                "franjas",
            );
            ui.selectable_value(
                &mut self.music_segmentation,
                String::from("positions"),
                "posiciones",
            );
            ui.selectable_value(
                &mut self.music_segmentation,
                String::from("sectors"),
                "sectores",
            );
        });
        ui.label(localized(language, "Escala", "Scale"));
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(&mut self.music_scale, String::from("major"), "mayor");
            ui.selectable_value(&mut self.music_scale, String::from("minor"), "menor");
            ui.selectable_value(
                &mut self.music_scale,
                String::from("pentatonic"),
                "pentatónica",
            );
            ui.selectable_value(
                &mut self.music_scale,
                String::from("chromatic"),
                "cromática",
            );
        });
        ui.horizontal(|ui| {
            if false && ui.button(localized(language, "Generar", "Generate")).clicked() {
                let pairs = self
                    .runs
                    .iter()
                    .map(|run| (run.case.label.as_str(), &run.result))
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
                    self.message = String::from("Patrón de Strudel generado.");
                }
            }
        });
        let music_after = (
            self.music_criterion.clone(),
            self.music_segmentation.clone(),
            self.music_scale.clone(),
        );
        if music_before != music_after {
            self.refresh_strudel_code();
        }
        if ui.button(localized(language, "Copiar", "Copy")).clicked() {
            self.copy_strudel_code();
        }
        if ui.button(localized(language, "Abrir Strudel", "Open Strudel")).clicked() {
            self.refresh_strudel_code();
            if self.fullscreen_configured {
                self.enter_strudel_window_mode(ctx);
            }
            self.open_full_strudel();
        }
        let line_count = self.strudel_code.lines().count().max(1);
        let editor_width = (ui.available_width() - 42.0).max(220.0);
        egui::Frame::dark_canvas(ui.style()).show(ui, |ui| {
            ui.horizontal_top(|ui| {
                ui.vertical(|ui| {
                    for line in 1..=line_count {
                        ui.monospace(RichText::new(format!("{line:>3}")).weak());
                    }
                });
                let code_changed = ui.add_sized(
                    egui::vec2(editor_width, 16.0 * 18.0),
                    egui::TextEdit::multiline(&mut self.strudel_code)
                        .font(egui::TextStyle::Monospace)
                        .desired_rows(16)
                        .desired_width(editor_width),
                ).changed();
                if code_changed {
                }
            });
        });
        if self.strudel_help_open {
            egui::Frame::group(ui.style()).show(ui, |ui| {
                ui.strong(localized(language, "Comandos de Strudel", "Strudel commands"));
                ui.monospace("note(\"c3 e3 g3\")     notas en secuencia\nnote(\"[c3,e3,g3]\")  acorde simultáneo\nsound(\"bd sd hh\")     sonidos rítmicos\n.s(\"sine\")            instrumento\n.hush()               silencio");
            });
        }
        if self.strudel_details_open {
            egui::Frame::group(ui.style()).show(ui, |ui| {
                ui.strong(localized(language, "Cómo se interpreta", "How it plays"));
                let details = describe_strudel(&self.strudel_code);
                if details.is_empty() {
                    ui.label(localized(language, "No se han detectado comandos musicales.", "No musical commands detected."));
                } else {
                    for detail in details {
                        ui.label(detail);
                    }
                }
            });
        }
    }

    fn refresh_strudel_code(&mut self) {
        let pairs = self
            .runs
            .iter()
            .map(|run| (run.case.label.as_str(), &run.result))
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
            self.strudel_code = Self::generated_strudel_code(pattern.code);
            self.message = String::from("Patron de Strudel actualizado.");
        }
    }

    fn generated_strudel_code(code: String) -> String {
        code
    }

    fn enter_strudel_window_mode(&mut self, ctx: &egui::Context) {
        self.strudel_window_mode_active = true;
        self.custom_fullscreen = true;
        ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(false));
        ctx.send_viewport_cmd(egui::ViewportCommand::Decorations(false));
        ctx.send_viewport_cmd(egui::ViewportCommand::Maximized(true));
        ctx.request_repaint();
    }

    fn restore_fullscreen(&mut self, ctx: &egui::Context) {
        self.custom_fullscreen = true;
        ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(false));
        ctx.send_viewport_cmd(egui::ViewportCommand::Decorations(false));
        ctx.send_viewport_cmd(egui::ViewportCommand::Maximized(false));
        if let Some(monitor_size) = ctx.input(|input| input.viewport().monitor_size) {
            ctx.send_viewport_cmd(egui::ViewportCommand::OuterPosition(egui::pos2(0.0, 0.0)));
            ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(monitor_size));
        }
        ctx.request_repaint();
    }

    fn open_full_strudel(&mut self) {
        if let Some(process) = &mut self.strudel_process {
            match process.try_wait() {
                Ok(None) => return,
                Ok(Some(_)) => self.strudel_process = None,
                Err(_) => return,
            }
        }
        let temp_root = std::env::temp_dir().join("fractal_music_strudel");
        let path = temp_root.join(if cfg!(target_os = "windows") {
            "strudel.exe"
        } else {
            "strudel"
        });
        let profile_dir = temp_root.join("profile");
        if let Err(error) =
            fs::create_dir_all(&temp_root).and_then(|_| fs::write(&path, EMBEDDED_STRUDEL))
        {
            self.message = format!("No se pudo extraer Strudel: {error}");
            return;
        }
        if let Err(error) = fs::create_dir_all(&profile_dir) {
            self.message = format!("No se pudo preparar el perfil de Strudel: {error}");
            return;
        }

        let mut command = std::process::Command::new(&path);
        command.current_dir(&temp_root);
        command.env("WEBVIEW2_USER_DATA_FOLDER", &profile_dir);
        if let Some(runtime_dir) = portable_webview2_directory() {
            if !runtime_dir.join("msedgewebview2.exe").is_file() {
                self.message = format!(
                    "No se encontró el runtime portable de WebView2 en {}.",
                    runtime_dir.display()
                );
                return;
            }
            command.env("WEBVIEW2_BROWSER_EXECUTABLE_FOLDER", runtime_dir);
        }
        match command.spawn() {
            Ok(process) => self.strudel_process = Some(process),
            Err(error) => self.message = format!("No se pudo abrir Strudel: {error}"),
        }
    }

    fn copy_strudel_code(&mut self) {
        match arboard::Clipboard::new()
            .and_then(|mut clipboard| clipboard.set_text(self.strudel_code.clone()))
        {
            Ok(()) => self.message = String::from("CÃ³digo de Strudel copiado."),
            Err(error) => self.message = format!("No se pudo copiar el cÃ³digo: {error}"),
        }
    }

    #[allow(dead_code)]
    fn draw_strudel_editor(&mut self, ctx: &egui::Context, language: Language) {
        if !self.strudel_editor_open {
            return;
        }
        let mut open = self.strudel_editor_open;
        egui::Window::new(localized(language, "Editor de Strudel", "Strudel editor"))
            .open(&mut open)
            .default_size(egui::vec2(760.0, 520.0))
            .resizable(true)
            .show(ctx, |ui| {
                ui.heading(localized(language, "Editor de Strudel", "Strudel editor"));
                ui.separator();
                ui.label(localized(
                    language,
                    "Editor offline: escribe o modifica tu patrón directamente.",
                    "Offline editor: write or modify your pattern directly.",
                ));
                ui.add(
                    egui::TextEdit::multiline(&mut self.strudel_code)
                        .font(egui::TextStyle::Monospace)
                        .desired_rows(18)
                        .desired_width(f32::INFINITY),
                );
                ui.horizontal(|ui| {
                    if ui.button(localized(language, "Guardar archivo", "Save file")).clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("Strudel", &["strudel", "js", "txt"])
                            .set_file_name("patron.strudel")
                            .save_file()
                        {
                            match std::fs::write(&path, &self.strudel_code) {
                                Ok(()) => {
                                    self.message = format!("Archivo guardado: {}", path.display());
                                }
                                Err(error) => {
                                    self.message = format!("Error guardando Strudel: {error}");
                                }
                            }
                        }
                    }
                    if ui.button(localized(language, "Copiar", "Copy")).clicked() {
                        if let Ok(mut clipboard) = arboard::Clipboard::new() {
                            let _ = clipboard.set_text(self.strudel_code.clone());
                        }
                    }
                });
            });
        self.strudel_editor_open = open;
    }

    fn draw_table(
        &mut self,
        ui: &mut egui::Ui,
        mode: TableMode,
        selected_only: bool,
        table_id: &str,
    ) {
        let cache_key = format!(
            "{table_id}:{mode:?}:{}",
            self.active_case_id.as_deref().unwrap_or("-")
        );
        if !self.table_cache.contains_key(&cache_key) {
            let data = self.build_table_rows(mode, selected_only);
            self.table_cache.insert(cache_key.clone(), data);
        }
        let data = self
            .table_cache
            .get(&cache_key)
            .expect("table cache entry was just inserted");
        let visible_rows = VISIBLE_TABLE_ROWS;
        // Dynamically adjust table height based on mode
        // Las dos tablas usan exactamente la misma altura para que los
        // sectores permanezcan alineados aunque tengan distinto contenido.
        let row_spacing = ui.spacing().item_spacing.y;
        let table_height = (visible_rows as f32 * (TABLE_ROW_HEIGHT + row_spacing)
            - row_spacing)
            .max(TABLE_ROW_HEIGHT);

        egui::Grid::new(format!("{table_id}_header")).show(ui, |ui| {
            ui.strong("caso");
            ui.strong("tipo");
            ui.strong("n");
            ui.strong("valor");
            ui.strong("coord");
            ui.strong("idx");
            ui.strong("centro");
            ui.strong("celda");
            ui.end_row();
        });
        if data.is_empty() {
            let message = match mode {
                TableMode::Sucesion => "No hay sucesiones para mostrar.",
                TableMode::Atractores => "No hay atractores para mostrar.",
                TableMode::Todo => "No hay datos para mostrar.",
            };
            ui.allocate_ui_with_layout(
                egui::vec2(ui.available_width(), table_height),
                egui::Layout::top_down(egui::Align::Min),
                |ui| ui.label(RichText::new(message).weak()),
            );
            return;
        }
        egui::ScrollArea::vertical()
            .id_salt(format!("{table_id}:{}", data.len()))
            .max_height(table_height)
            .min_scrolled_height(table_height)
            .max_width(ui.available_width())
            .hscroll(true)
            .auto_shrink([false, false])
            .show_rows(ui, TABLE_ROW_HEIGHT, data.len(), |ui, row_range| {
                egui::Grid::new(format!("{table_id}_body")).striped(true).show(ui, |ui| {
                    for row in &data[row_range] {
                        for value in row {
                            ui.label(value);
                        }
                        ui.end_row();
                    }
                });
            });
    }

    fn build_table_rows(&self, mode: TableMode, selected_only: bool) -> Vec<[String; 8]> {
        let mut rows = Vec::new();
        for run in &self.runs {
            // En la vista de sucesiones se muestra únicamente el caso activo
            // seleccionado desde la tabla de casos.
            if selected_only {
                if let Some(active_case_id) = &self.active_case_id {
                    if active_case_id != &run.case.identifier {
                        continue;
                    }
                }
            }
            if matches!(mode, TableMode::Todo | TableMode::Sucesion) {
                for item in &run.result.terms {
                    rows.push([
                        run.case.label.clone(),
                        String::from("S"),
                        item.term.index.to_string(),
                        format_number(item.point.value),
                        format_mapping_float(item.point.coordinates.iter().map(|(k, v)| (k, *v))),
                        format_mapping_usize(item.point.indices.as_ref()),
                        format_mapping_float_opt(item.point.discrete_coordinates.as_ref()),
                        item.point
                            .cell_id
                            .map_or_else(|| String::from("-"), |v| v.to_string()),
                    ]);
                }
            }
            if matches!(mode, TableMode::Todo | TableMode::Atractores) {
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
                            point
                                .cell_id
                                .map_or_else(|| String::from("-"), |v| v.to_string()),
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
                tolerance: self.tolerance_value(),
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
        self.rebuild_case_colors();
        self.cached_main_auto_limits = None;
        let reuse_cached_results = self.cached_analysis.as_ref() == Some(&config.analysis)
            && self.cached_discretization.as_ref() == Some(&config.discretization);
        let mut previous_runs = std::mem::take(&mut self.runs)
            .into_iter()
            .map(|run| (run.case.identifier.clone(), run))
            .collect::<HashMap<_, _>>();
        let mut results_changed = !reuse_cached_results;
        let mut runs = Vec::new();
        for case in &self.cases {
            let mut cached_render_points = None;
            let mut cached_mandelbrot_member = None;
            let result = if reuse_cached_results {
                if let Some(previous) = previous_runs.remove(&case.identifier) {
                    if previous.case.recurrence == case.recurrence {
                        cached_mandelbrot_member = Some(previous.mandelbrot_member);
                        cached_render_points = Some((
                            previous.trajectory_points,
                            previous.attractor_points,
                        ));
                        previous.result
                    } else {
                        results_changed = true;
                        let run_config = ExplorerConfig {
                            recurrence: case.recurrence.clone(),
                            analysis: config.analysis.clone(),
                            discretization: config.discretization.clone(),
                        };
                        run_exploration(&run_config)?
                    }
                } else {
                    results_changed = true;
                    let run_config = ExplorerConfig {
                        recurrence: case.recurrence.clone(),
                        analysis: config.analysis.clone(),
                        discretization: config.discretization.clone(),
                    };
                    run_exploration(&run_config)?
                }
            } else {
                let run_config = ExplorerConfig {
                    recurrence: case.recurrence.clone(),
                    analysis: config.analysis.clone(),
                    discretization: config.discretization.clone(),
                };
                run_exploration(&run_config)?
            };
            let is_active = self.active_case_id.as_ref() == Some(&case.identifier);
            let (trajectory_points, attractor_points) = cached_render_points
                .unwrap_or_else(|| build_render_points(&result));
            runs.push(CaseRun {
                mandelbrot_member: cached_mandelbrot_member
                    .unwrap_or_else(|| mandelbrot_membership_for_case(case)),
                trajectory_points,
                attractor_points,
                case: case.clone(),
                result,
                is_active,
            });
        }
        self.runs = runs;
        if results_changed {
            self.table_cache.clear();
            let pairs = self
                .runs
                .iter()
                .map(|run| (run.case.label.as_str(), &run.result))
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
                self.strudel_code = Self::generated_strudel_code(pattern.code);
            }
        }
        self.cached_analysis = Some(config.analysis);
        self.cached_discretization = Some(config.discretization);
        Ok(())
    }

    fn calculate_current_case(&mut self) -> Result<()> {
        let config = self.current_config()?;
        match config.recurrence.name.as_str() {
            "logistic" => {
                let r_value = config.recurrence.parameters["r"];
                let x_value = config.recurrence.parameters["x0"];
                self.store_logistic_pair(r_value, x_value, "manual_calculation", false)?;
            }
            "quadratic_complex" => {
                self.store_case(
                    config.recurrence.clone(),
                    "manual_calculation",
                    BTreeMap::new(),
                    false,
                    true,
                )?;
            }
            _ => {
                self.store_case(
                    config.recurrence,
                    "manual_calculation",
                    BTreeMap::new(),
                    false,
                    true,
                )?;
            }
        }
        self.dirty = true;
        self.recalculate()?;
        self.autosave_session()?;
        Ok(())
    }

    fn sync_from_logistic_parameters(&mut self) {
        if let (Ok(r_value), Ok(x_value)) = (self.r.parse::<f64>(), self.x0.parse::<f64>()) {
            let converted = logistic_to_quadratic(r_value, x_value);
            self.c_real = format!("{:.10}", converted.c.re);
            self.c_imag = format!("{:.10}", converted.c.im);
            self.z0_real = format!("{:.10}", converted.z0.re);
            self.z0_imag = format!("{:.10}", converted.z0.im);
        }
    }

    fn sync_from_quadratic_parameters(&mut self) {
        if let (Ok(c_re), Ok(c_im), Ok(z0_re), Ok(z0_im)) = (
            self.c_real.parse::<f64>(),
            self.c_imag.parse::<f64>(),
            self.z0_real.parse::<f64>(),
            self.z0_imag.parse::<f64>(),
        ) {
            if let Some(converted) = quadratic_to_logistic(
                Complex64::new(c_re, c_im),
                Complex64::new(z0_re, z0_im),
                self.tolerance_value(),
            ) {
                self.r = format!("{:.10}", converted.r);
                self.x0 = format!("{:.10}", converted.x0);
            }
        }
    }

    fn apply_config(&mut self, config: &ExplorerConfig) {
        self.coordinates = match config.discretization.coordinate_system.as_str() {
            "x" => CoordinateChoice::X,
            "cartesian" => CoordinateChoice::Cartesian,
            "polar" => CoordinateChoice::Polar,
            "y" => CoordinateChoice::Y,
            "radius" => CoordinateChoice::Radius,
            "angle" => CoordinateChoice::Angle,
            _ => CoordinateChoice::Cartesian,
        };
        self.apply_recurrence(&config.recurrence);
        self.max_terms = config.analysis.max_terms.to_string();
        self.tolerance = config.analysis.tolerance.to_string();
        self.max_period = config.analysis.max_period.to_string();
        self.transient_terms = config.analysis.transient_terms.to_string();
        for (axis, axis_config) in &config.discretization.ranges {
            let value = format!(
                "{},{},{}",
                axis_config.minimum, axis_config.maximum, axis_config.steps
            );
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
                self.x0 =
                    format!("{:.10}", recurrence.parameters.get("x0").copied().unwrap_or(0.2));
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
        self.last_sequence = self.sequence;
    }

    fn open_import_json(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("json", &["json"])
            .pick_file()
        {
            match self.import_json(&path) {
                Ok(()) => self.message = String::from("Sesión importada."),
                Err(error) => self.message = format!("Error al importar: {error}"),
            }
        }
    }

    fn open_export_json(&mut self) {
        let _ = std::fs::create_dir_all(&self.export_directory);
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("json", &["json"])
            .set_directory(&self.export_directory)
            .save_file()
        {
            match self.export_json(&path) {
                Ok(()) => self.message = String::from("Sesión JSON exportada."),
                Err(error) => self.message = format!("Error al exportar JSON: {error}"),
            }
        }
    }

    fn open_export_tables_csv(&mut self) {
        let _ = std::fs::create_dir_all(&self.export_directory);
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("CSV compatible con Excel", &["csv"])
            .set_directory(&self.export_directory)
            .save_file()
        {
            match self.export_tables_csv(&path) {
                Ok(()) => self.message = String::from("Tablas CSV exportadas."),
                Err(error) => self.message = format!("Error al exportar tablas: {error}"),
            }
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
            &self.map_limits,
            &BTreeMap::from([
                (String::from("criterion"), self.music_criterion.clone()),
                (String::from("segmentation"), self.music_segmentation.clone()),
                (String::from("scale"), self.music_scale.clone()),
            ]),
        );
        write_session(path, &data)
    }

    fn export_tables_csv(&self, path: &std::path::Path) -> Result<()> {
        let mut writer = csv::WriterBuilder::new()
            .delimiter(b';')
            .from_path(path)?;
        writer.write_record([
            "caso", "identificador", "tabla", "tipo", "n", "valor",
            "coordenadas", "indices", "centro", "celda", "seleccionado",
            "estado_mandelbrot",
        ])?;

        for run in &self.runs {
            let selected = (self.active_case_id.as_ref() == Some(&run.case.identifier)).to_string();
            let mandelbrot_status = if run.case.recurrence.name == "quadratic_complex" {
                let c_re = run.case.recurrence.parameters.get("c_real").copied().unwrap_or(0.0);
                let c_im = run.case.recurrence.parameters.get("c_imag").copied().unwrap_or(0.0);
                if run
                    .mandelbrot_member
                    .unwrap_or_else(|| is_mandelbrot_member(c_re, c_im))
                {
                    "dentro"
                } else {
                    "fuera"
                }
            } else {
                "no_aplica"
            };

            for item in &run.result.terms {
                let n = item.term.index.to_string();
                let value = format_number(item.point.value);
                let coordinates = format_mapping_float(item.point.coordinates.iter().map(|(k, v)| (k, *v)));
                let indices = format_mapping_usize(item.point.indices.as_ref());
                let center = format_mapping_float_opt(item.point.discrete_coordinates.as_ref());
                let cell = item.point.cell_id.map_or_else(|| String::from("-"), |v| v.to_string());
                writer.write_record([
                    run.case.label.as_str(), run.case.identifier.as_str(),
                    "todos / sucesiones", "S", n.as_str(), value.as_str(),
                    coordinates.as_str(), indices.as_str(), center.as_str(), cell.as_str(),
                    selected.as_str(), mandelbrot_status,
                ])?;
            }
            for (attractor_index, attractor) in run.result.attractors.iter().enumerate() {
                for (point_index, point) in attractor.points.iter().enumerate() {
                    let kind = format!("A{}.{}", attractor_index + 1, point_index + 1);
                    let value = format_number(point.value);
                    let coordinates = format_mapping_float(point.coordinates.iter().map(|(k, v)| (k, *v)));
                    let indices = format_mapping_usize(point.indices.as_ref());
                    let center = format_mapping_float_opt(point.discrete_coordinates.as_ref());
                    let cell = point.cell_id.map_or_else(|| String::from("-"), |v| v.to_string());
                    writer.write_record([
                        run.case.label.as_str(), run.case.identifier.as_str(),
                        "todos / atractores", kind.as_str(), "-", value.as_str(),
                        coordinates.as_str(), indices.as_str(), center.as_str(), cell.as_str(),
                        selected.as_str(), mandelbrot_status,
                    ])?;
                }
            }
        }
        writer.flush()?;
        Ok(())
    }

    fn restore_imported_results(&mut self, imported: &ImportedSession) -> Result<()> {
        if imported.cached_results.is_empty() {
            return Ok(());
        }
        let transformer = DiscreteTransformer::new(
            build_coordinate_system(&imported.config.discretization.coordinate_system)?,
            build_grid(&imported.config.discretization)?,
        )?;
        let mut restored_runs = Vec::with_capacity(self.cases.len());
        for case in &self.cases {
            let Some(sequence_result) = imported.cached_results.get(&case.identifier) else {
                continue;
            };
            let run_config = ExplorerConfig {
                recurrence: case.recurrence.clone(),
                analysis: imported.config.analysis.clone(),
                discretization: imported.config.discretization.clone(),
            };
            let result = DiscreteAnalysisResult::from_analysis(
                sequence_result.clone(),
                transformer.clone(),
                Some(run_config),
            )?;
            let (trajectory_points, attractor_points) = build_render_points(&result);
            restored_runs.push(CaseRun {
                case: case.clone(),
                is_active: self.active_case_id.as_ref() == Some(&case.identifier),
                mandelbrot_member: mandelbrot_membership_for_case(case),
                trajectory_points,
                attractor_points,
                result,
            });
        }
        self.runs = restored_runs;
        self.cached_analysis = Some(imported.config.analysis.clone());
        self.cached_discretization = Some(imported.config.discretization.clone());
        Ok(())
    }

    fn import_json(&mut self, path: &std::path::Path) -> Result<()> {
        let raw = std::fs::read_to_string(path)?;
        let data = serde_json::from_str::<serde_json::Value>(&raw)?;
        let imported = parse_session_data(&data)?;
        self.cases = imported.cases.clone();
        self.runs.clear();
        self.table_cache.clear();
        self.cached_analysis = None;
        self.cached_discretization = None;
        self.active_case_id = imported.active_case_id.clone();
        self.case_counter = self.cases.len();
        self.apply_config(&imported.config);
        self.restore_imported_results(&imported)?;
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
        // Fusionar los limites guardados con los limites por defecto.
        for (key, value) in imported.map_limits {
            self.map_limits.insert(key, value);
        }
        // Solo se recalculan los casos sin resultado almacenado en el JSON.
        self.recalculate()?;
        self.refresh_strudel_code();
        self.dirty = false;
        Ok(())
    }

    fn autosave_session(&self) -> Result<()> {
        // Build current config and session data, then write to autosave file in cwd.
        if let Ok(config) = self.current_config() {
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
                &self.map_limits,
                &BTreeMap::from([
                    (String::from("criterion"), self.music_criterion.clone()),
                    (String::from("segmentation"), self.music_segmentation.clone()),
                    (String::from("scale"), self.music_scale.clone()),
                ]),
            );
            let path = std::path::Path::new("session_autosave.json");
            write_session(path, &data)?;
        }
        Ok(())
    }

    fn store_logistic_pair(
        &mut self,
        r_value: f64,
        x_value: f64,
        source: &str,
        force_new: bool,
    ) -> Result<()> {
        self.store_case(
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
        Ok(())
    }

    fn delete_selected_cases(&mut self) {
        if self.selected_case_ids.is_empty() {
            return;
        }
        let selected = &self.selected_case_ids;
        self.cases.retain(|case| !selected.contains(&case.identifier));
        self.runs.retain(|run| !selected.contains(&run.case.identifier));
        self.active_case_id = None;
        self.selected_case_ids.clear();
        self.case_selection_anchor = None;
        self.point_actions = None;
        self.point_actions_position = None;
        self.case_context_menu_open = false;
        self.table_cache.clear();
        self.cached_analysis = None;
        self.cached_discretization = None;
        self.refresh_strudel_code();
        let _ = self.autosave_session();
        self.message = String::from("Casos seleccionados eliminados.");
        self.dirty = false;
    }

    fn clear_all_cases(&mut self) {
        self.cases.clear();
        self.active_case_id = None;
        self.selected_case_ids.clear();
        self.case_selection_anchor = None;
        self.runs.clear();
        self.cached_analysis = None;
        self.cached_discretization = None;
        self.table_cache.clear();
        self.case_counter = 0;
        self.case_list_start = 0;
        self.table_start = 0;
        self.point_actions = None;
        self.point_actions_position = None;
        self.case_context_menu_open = false;
        self.julia_open = false;
        self.julia_point = None;
        let _ = self.autosave_session();
        self.message = String::from("Datos eliminados.");
        self.dirty = true;
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
        self.case_list_start = self.cases.len().saturating_sub(5);
        // Persist the newly added case
        let _ = self.autosave_session();
        if make_active {
            self.active_case_id = Some(identifier);
        }
        Ok(case)
    }

    #[allow(dead_code)]
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

    fn zoom_limits(
        &self,
        limits: (f64, f64, f64, f64),
        focus: (f64, f64),
        scale: f64,
        bounds: (f64, f64, f64, f64),
    ) -> (f64, f64, f64, f64) {
        let (x_min, x_max, y_min, y_max) = limits;
        let (bound_x_min, bound_x_max, bound_y_min, bound_y_max) = bounds;
        let new_x_span = (x_max - x_min) * scale
            .max(1e-7)
            .min(bound_x_max - bound_x_min);
        let new_y_span = (y_max - y_min) * scale
            .max(1e-7)
            .min(bound_y_max - bound_y_min);
        let (new_x_min, new_x_max) = self.center_limited_range(
            focus.0,
            new_x_span,
            bound_x_min,
            bound_x_max,
        );
        let (new_y_min, new_y_max) = self.center_limited_range(
            focus.1,
            new_y_span,
            bound_y_min,
            bound_y_max,
        );
        (new_x_min, new_x_max, new_y_min, new_y_max)
    }

    fn default_main_plot_limits(
        &self,
        axes: &[&str],
        is_polar: bool,
    ) -> (f64, f64, f64, f64) {
        if is_polar {
            let r_limit = self
                .r_range
                .split(',')
                .nth(1)
                .and_then(|value| value.trim().parse::<f64>().ok())
                .filter(|value| *value > 0.0)
                .unwrap_or(2.0);
            let display_limit = r_limit * 1.18;
            return (
                -display_limit,
                display_limit,
                -display_limit,
                display_limit,
            );
        }
        let mut points = Vec::new();
        for run in &self.runs {
            for term in &run.result.terms {
                if !term.point.inside_domain {
                    continue;
                }
                let point = if is_polar {
                    let r = term.point.coordinates["r"];
                    let theta = term.point.coordinates["theta"];
                    [r * theta.cos(), r * theta.sin()]
                } else if axes.len() == 1 {
                    [term.term.index as f64, term.point.coordinates[axes[0]]]
                } else {
                    [
                        term.point.coordinates[axes[0]],
                        term.point.coordinates[axes[1]],
                    ]
                };
                points.push(point);
            }
        }
        if points.is_empty() {
            return (-1.0, 1.0, -1.0, 1.0);
        }
        let (mut x_min, mut x_max) = (points[0][0], points[0][0]);
        let (mut y_min, mut y_max) = (points[0][1], points[0][1]);
        for point in points.iter().skip(1) {
            x_min = x_min.min(point[0]);
            x_max = x_max.max(point[0]);
            y_min = y_min.min(point[1]);
            y_max = y_max.max(point[1]);
        }
        let x_margin = ((x_max - x_min) * 0.08).max(0.05);
        let y_margin = ((y_max - y_min) * 0.08).max(0.05);
        (x_min - x_margin, x_max + x_margin, y_min - y_margin, y_max + y_margin)
    }

    fn handle_main_plot_interaction(
        &mut self,
        ui: &egui::Ui,
        response: &PlotResponse<()>,
        limits: (f64, f64, f64, f64),
    ) {
        let input = ui.input(|input| input.clone());

        // Desplazamiento explícito: solo Alt+arrastre inicia el pan.
        if input.modifiers.alt && input.pointer.any_pressed() {
            if let Some(origin) = input.pointer.press_origin() {
                if response.response.rect.contains(origin) {
                    self.main_plot_pan_start = Some(origin);
                    self.main_plot_pan_limits_start = Some(limits);
                }
            }
        }
        if let (Some(start_pos), Some(start_limits)) =
            (self.main_plot_pan_start, self.main_plot_pan_limits_start)
        {
            if input.pointer.any_down() {
                if let Some(current_pos) = input.pointer.hover_pos() {
                    let start_value = response.transform.value_from_position(start_pos);
                    let current_value = response.transform.value_from_position(current_pos);
                    let dx = current_value.x - start_value.x;
                    let dy = current_value.y - start_value.y;
                    self.main_plot_limits = Some((
                        start_limits.0 - dx,
                        start_limits.1 - dx,
                        start_limits.2 - dy,
                        start_limits.3 - dy,
                    ));
                }
            } else {
                self.main_plot_pan_start = None;
                self.main_plot_pan_limits_start = None;
            }
        }

        // Zoom únicamente con Ctrl+rueda; Shift y el arrastre normal no intervienen.
        if let Some(pointer) = response.response.hover_pos() {
            let scroll = input.raw_scroll_delta.y;
            if scroll != 0.0 && (input.modifiers.ctrl || input.modifiers.command) {
                let focus = response.transform.value_from_position(pointer);
                let scale = if scroll > 0.0 { 1.0 / ZOOM_SCALE } else { ZOOM_SCALE };
                self.main_plot_limits = Some((
                    focus.x - (focus.x - limits.0) * scale,
                    focus.x + (limits.1 - focus.x) * scale,
                    focus.y - (focus.y - limits.2) * scale,
                    focus.y + (limits.3 - focus.y) * scale,
                ));
            }
        }
    }

    fn center_limited_range(&self, focus: f64, span: f64, lower: f64, upper: f64) -> (f64, f64) {
        let center = focus.max(lower + span / 2.0).min(upper - span / 2.0);
        (center - span / 2.0, center + span / 2.0)
    }

    fn color_for_case(&self, identifier: &str) -> Color32 {
        self.case_colors
            .get(identifier)
            .copied()
            .unwrap_or(Color32::from_rgb(80, 190, 220))
    }

    #[allow(dead_code)]
    fn color_for_case_legacy(&self, identifier: &str) -> Color32 {
        let Some(index) = self
            .cases
            .iter()
            .position(|case| case.identifier == identifier)
        else {
            return Color32::from_rgb(80, 190, 220);
        };

        // La paleta se genera con tonos muy separados y saturados, en vez de
        // reciclar una paleta corta. Así los casos visibles conservan colores
        // distintos y ninguno se confunde con el fondo, el blanco o el gris.
        let mut color = generated_case_color(index);
        let step = self.cases.len().max(1);
        for attempt in 1..=self.cases.len() {
            let already_used = self
                .cases
                .iter()
                .enumerate()
                .any(|(other_index, case)| {
                    case.identifier != identifier && generated_case_color(other_index) == color
                });
            if !already_used {
                break;
            }
            color = generated_case_color(index + attempt * step);
        }
        color
    }

    fn rebuild_case_colors(&mut self) {
        self.case_colors.clear();
        let mut used = HashSet::with_capacity(self.cases.len());
        for (index, case) in self.cases.iter().enumerate() {
            let mut candidate_index = index;
            let color = loop {
                let candidate = generated_case_color(candidate_index);
                if used.insert(candidate) {
                    break candidate;
                }
                candidate_index = candidate_index.saturating_add(self.cases.len().max(1));
            };
            self.case_colors.insert(case.identifier.clone(), color);
        }
    }

    fn tolerance_value(&self) -> f64 {
        self.tolerance
            .parse::<f64>()
            .ok()
            .filter(|value| value.is_finite() && *value > 0.0)
            .unwrap_or(DEFAULT_TOLERANCE)
    }

    fn set_transient_message(&mut self, message: &str) {
        self.message = message.to_string();
        self.message_until = Some(Instant::now() + Duration::from_millis(1500));
    }
}

fn generated_case_color(index: usize) -> Color32 {
    let hue = ((index as f64 * 0.618_033_988_749_895) % 1.0) as f32;
    let saturation = 0.78_f32;
    let value = 0.92_f32;
    let sector = hue * 6.0;
    let sector_index = sector.floor() as i32;
    let fraction = sector - sector.floor();
    let p = value * (1.0 - saturation);
    let q = value * (1.0 - saturation * fraction);
    let t = value * (1.0 - saturation * (1.0 - fraction));
    let (red, green, blue) = match sector_index {
        0 => (value, t, p),
        1 => (q, value, p),
        2 => (p, value, t),
        3 => (p, q, value),
        4 => (t, p, value),
        _ => (value, p, q),
    };
    Color32::from_rgb(
        (red * 255.0).round() as u8,
        (green * 255.0).round() as u8,
        (blue * 255.0).round() as u8,
    )
}

fn mandelbrot_membership_for_case(case: &ExplorationCase) -> Option<bool> {
    if case.recurrence.name != "quadratic_complex" {
        return None;
    }
    Some(is_mandelbrot_member(
        case.recurrence.parameters.get("c_real").copied().unwrap_or(0.0),
        case.recurrence.parameters.get("c_imag").copied().unwrap_or(0.0),
    ))
}

fn is_mandelbrot_member(c_re: f64, c_im: f64) -> bool {
    // Comprobaciones rápidas para el cardioide principal y el bulbo de la
    // izquierda; además evitan iterar innecesariamente en los puntos claros.
    let q = (c_re - 0.25).powi(2) + c_im.powi(2);
    if q * (q + c_re - 0.25) <= 0.25 * c_im.powi(2) {
        return true;
    }
    if (c_re + 1.0).powi(2) + c_im.powi(2) <= 0.0625 {
        return true;
    }

    let mut z_re = 0.0;
    let mut z_im = 0.0;
    // Debe coincidir con el límite usado por el shader del mapa (256),
    // para que la detección al pulsar nunca contradiga lo que se ve.
    for _ in 0..256 {
        let next_re = z_re * z_re - z_im * z_im + c_re;
        let next_im = 2.0 * z_re * z_im + c_im;
        z_re = next_re;
        z_im = next_im;
        if z_re * z_re + z_im * z_im > 4.0 {
            return false;
        }
    }
    true
}

fn localized<'a>(language: Language, spanish: &'a str, english: &'a str) -> &'a str {
    match language {
        Language::Spanish => spanish,
        Language::English => english,
    }
}

fn draw_app_logo(ui: &mut egui::Ui, texture: Option<&egui::TextureHandle>) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(26.0, 26.0), egui::Sense::hover());
    let painter = ui.painter();
    if let Some(texture) = texture {
        painter.image(
            texture.id(),
            rect,
            egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
            Color32::WHITE,
        );
    } else {
        painter.rect_filled(rect, 5.0, Color32::from_rgb(12, 30, 58));
    }
}

fn default_export_directory() -> String {
    std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(std::path::Path::to_path_buf))
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_default())
        .join("exports")
        .to_string_lossy()
        .into_owned()
}

fn portable_webview2_directory() -> Option<PathBuf> {
    std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(|parent| parent.join("webview2")))
}

fn text_row(ui: &mut egui::Ui, label: &str, value: &mut String) -> bool {
    ui.horizontal(|ui| {
        ui.label(label);
        let response = ui.text_edit_singleline(value);
        response.lost_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter))
    })
    .inner
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

fn thin_plot_plotpoints<I>(points: I, maximum: usize) -> Vec<PlotPoint>
where
    I: Iterator<Item = PlotPoint> + Clone,
{
    let point_count = points.clone().count();
    if point_count <= maximum || maximum < 2 {
        return points.collect();
    }
    let last = point_count - 1;
    let mut selected = Vec::with_capacity(maximum);
    let mut next_index = 0;
    for (index, point) in points.enumerate() {
        if index == next_index {
            selected.push(point);
            if selected.len() == maximum {
                break;
            }
            next_index = selected.len() * last / (maximum - 1);
        }
    }
    selected
}

fn build_render_points(
    result: &DiscreteAnalysisResult,
) -> (Vec<PlotPoint>, Vec<Vec<PlotPoint>>) {
    let axes = result.transformer.coordinate_system.axes();
    let is_polar = axes == ["r", "theta"];
    let trajectory = result
        .terms
        .iter()
        .filter(|term| term.point.inside_domain)
        .map(|term| {
            if is_polar {
                let r = term.point.coordinates["r"];
                let theta = term.point.coordinates["theta"];
                PlotPoint::new(r * theta.cos(), r * theta.sin())
            } else if axes.len() == 1 {
                PlotPoint::new(term.term.index as f64, term.point.coordinates[axes[0]])
            } else {
                PlotPoint::new(
                    term.point.coordinates[axes[0]],
                    term.point.coordinates[axes[1]],
                )
            }
        })
        ;
    let trajectory = thin_plot_plotpoints(trajectory, MAX_RENDERED_TRAJECTORY_POINTS);

    let attractors = result
        .attractors
        .iter()
        .map(|attractor| {
            let points = attractor
                .points
                .iter()
                .enumerate()
                .filter(|(_, point)| point.inside_domain)
                .map(|(index, point)| {
                    if is_polar {
                        let r = point.coordinates["r"];
                        let theta = point.coordinates["theta"];
                        PlotPoint::new(r * theta.cos(), r * theta.sin())
                    } else if axes.len() == 1 {
                        PlotPoint::new(
                            (result.terms.len().saturating_sub(attractor.points.len()) + index)
                                as f64,
                            point.coordinates[axes[0]],
                        )
                    } else {
                        PlotPoint::new(
                            point.coordinates[axes[0]],
                            point.coordinates[axes[1]],
                        )
                    }
                })
                ;
            thin_plot_plotpoints(points, MAX_RENDERED_TRAJECTORY_POINTS)
        })
        .collect();

    (trajectory, attractors)
}

fn compute_bifurcation_points() -> Vec<PlotPoint> {
    let steps = 700;
    let mut points = Vec::new();
    for i in 0..steps {
        let r = i as f64 * 4.0 / (steps - 1) as f64;
        let mut x = 0.5;
        for iteration in 0..520 {
            x = r * x * (1.0 - x);
            if iteration > 350 {
                points.push(PlotPoint::new(r, x));
            }
        }
    }
    points
}
