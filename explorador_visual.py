"""Interactive explorer for real and complex recurrence sequences.

Run with::

    python explorador_visual.py

The explorer keeps a collection of cases.  A click in the logistic map stores
both the logistic orbit and its affine-conjugate quadratic complex orbit.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, RadioButtons, TextBox

from fractal_sequences import (
    AnalysisConfig,
    AxisRangeConfig,
    DiscretizationConfig,
    DiscreteAnalysisResult,
    ExplorerConfig,
    ExplorationCase,
    MusicMappingConfig,
    RecurrenceConfig,
    build_session_data,
    default_concept_config,
    logistic_to_quadratic,
    generate_strudel_pattern,
    quadratic_to_logistic,
    read_session,
    run_exploration,
    write_results_csv,
    write_session,
)


@dataclass(frozen=True)
class AxisConfig:
    minimum: float
    maximum: float
    steps: int


@dataclass(frozen=True)
class CaseRun:
    """One stored case plus its current analysis and discrete projection."""

    case: ExplorationCase
    result: DiscreteAnalysisResult
    is_active: bool = False


SEQUENCES = ("cos(x)", "logistica", "z^2+c")
COORDINATES = ("x", "cartesiano", "polar", "y", "radio", "angulo")
TABLE_MODES = ("todo", "sucesion", "atractores")
MUSIC_CRITERIA = ("melodia", "armonia", "ritmo", "textura")
MUSIC_SEGMENTATIONS = ("franjas", "posiciones", "sectores")
MUSIC_SCALES = ("mayor", "menor", "pentatonica", "cromatica")
MUSIC_CRITERION_TO_CONFIG = {
    "melodia": "melody",
    "armonia": "harmony",
    "ritmo": "rhythm",
    "textura": "texture",
}
MUSIC_SEGMENTATION_TO_CONFIG = {
    "franjas": "bands",
    "posiciones": "positions",
    "sectores": "sectors",
}
MUSIC_SCALE_TO_CONFIG = {
    "mayor": "major",
    "menor": "minor",
    "pentatonica": "pentatonic",
    "cromatica": "chromatic",
}
SEQUENCE_TO_CONFIG = {
    "cos(x)": "cosine",
    "logistica": "logistic",
    "z^2+c": "quadratic_complex",
}
CONFIG_TO_SEQUENCE = {value: key for key, value in SEQUENCE_TO_CONFIG.items()}
COORDINATE_TO_CONFIG = {
    "x": "x",
    "cartesiano": "cartesian",
    "polar": "polar",
    "y": "y",
    "radio": "radius",
    "angulo": "angle",
}
CONFIG_TO_COORDINATE = {value: key for key, value in COORDINATE_TO_CONFIG.items()}
COORDINATE_AXES = {
    "x": ("x",),
    "cartesiano": ("x", "y"),
    "polar": ("r", "theta"),
    "y": ("y",),
    "radio": ("r",),
    "angulo": ("theta",),
}


class VisualExplorer:
    """Matplotlib desktop application built on the sequence framework."""

    _MAP_BOUNDS = {
        "logistica": (0.0, 4.0, 0.0, 1.0),
        "z^2+c": (-2.0, 1.0, -1.35, 1.35),
    }

    def __init__(self) -> None:
        self.fig = plt.figure(figsize=(16, 10))
        manager = getattr(self.fig.canvas, "manager", None)
        if manager is not None:
            manager.set_window_title("Explorador de sucesiones")

        self.plot_ax = None
        self.parameter_ax = self.fig.add_axes([0.655, 0.45, 0.325, 0.50])
        self.summary_ax = self.fig.add_axes([0.32, 0.285, 0.305, 0.135])
        self.strudel_ax = self.fig.add_axes([0.655, 0.325, 0.325, 0.030])
        self.music_info_ax = self.fig.add_axes([0.655, 0.285, 0.325, 0.030])
        self.table_ax = self.fig.add_axes([0.32, 0.055, 0.66, 0.205])
        self.table_status_ax = self.fig.add_axes([0.32, 0.025, 0.66, 0.020])
        self.case_list_ax = self.fig.add_axes([0.025, 0.020, 0.120, 0.082])
        for axis in (
            self.summary_ax,
            self.music_info_ax,
            self.table_ax,
            self.table_status_ax,
            self.case_list_ax,
        ):
            axis.axis("off")

        self.widgets: list[object] = []
        self.textboxes: dict[str, TextBox] = {}
        self.sequence_selector: RadioButtons | None = None
        self.coordinate_selector: RadioButtons | None = None
        self.table_mode_selector: RadioButtons | None = None
        self.music_criterion_selector: RadioButtons | None = None
        self.music_segmentation_selector: RadioButtons | None = None
        self.music_scale_selector: RadioButtons | None = None
        self.strudel_box: TextBox | None = None

        self.cases: list[ExplorationCase] = []
        self.active_case_id: str | None = None
        self.case_runs: list[CaseRun] = []
        self._visible_case_ids: list[str] = []
        self._case_list_start = 0
        self._case_counter = 0
        self._suspend_callbacks = False
        self._is_redrawing = False
        self._is_blank = False
        self._message = ""
        self.music_pattern = None
        self._result_cache: dict[str, DiscreteAnalysisResult] = {}
        self._bifurcation_cache: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]] = {}
        self._mandelbrot_cache: dict[
            tuple[float, float, float, float],
            tuple[np.ndarray, tuple[float, float, float, float]],
        ] = {}
        self._map_limits = dict(self._MAP_BOUNDS)
        self.table_rows_cache: list[list[str]] = []
        self.visible_table_rows = 8
        self.table_start = 0
        self.table_max_start = 0

        self._build_controls()
        self._last_sequence = self._selected_sequence()
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.fig.canvas.mpl_connect("button_press_event", self._on_button_press)
        self.redraw()

    def _build_controls(self) -> None:
        self.fig.text(0.025, 0.965, "Explorador", fontsize=12, weight="bold")

        self.fig.text(0.025, 0.935, "Sucesion", fontsize=8, weight="bold")
        sequence_ax = self.fig.add_axes([0.025, 0.835, 0.120, 0.090])
        self.sequence_selector = RadioButtons(sequence_ax, SEQUENCES, active=1)
        self.sequence_selector.on_clicked(self._on_sequence_selected)
        self.widgets.append(self.sequence_selector)

        self.fig.text(0.165, 0.935, "Coordenadas", fontsize=8, weight="bold")
        coordinate_ax = self.fig.add_axes([0.165, 0.795, 0.125, 0.130])
        self.coordinate_selector = RadioButtons(coordinate_ax, COORDINATES, active=1)
        self.coordinate_selector.on_clicked(self._on_coordinate_selected)
        self.widgets.append(self.coordinate_selector)

        self.fig.text(0.025, 0.770, "Parametros", fontsize=8, weight="bold")
        self._add_textbox("x0", "x0", "0.2", 0.725, x=0.025)
        self._add_textbox("r", "r", "3.2", 0.725, x=0.165)
        self._add_textbox("c_real", "c real", "-0.123", 0.675, x=0.025)
        self._add_textbox("c_imag", "c imag", "0.745", 0.675, x=0.165)
        self._add_textbox("z0_real", "z0 real", "0.0", 0.625, x=0.025)
        self._add_textbox("z0_imag", "z0 imag", "0.0", 0.625, x=0.165)

        self.fig.text(0.025, 0.595, "Analisis", fontsize=8, weight="bold")
        self._add_textbox("max_terms", "terminos", "1000", 0.550, x=0.025)
        self._add_textbox("tolerance", "tol", "1e-7", 0.550, x=0.165)
        self._add_textbox("max_period", "periodo", "16", 0.500, x=0.025)
        self._add_textbox("transient_terms", "transitorio", "100", 0.500, x=0.165)
        self._add_textbox("table_rows", "filas", "9", 0.450, x=0.025)

        self.fig.text(0.025, 0.420, "Dominios: min,max,n", fontsize=8, weight="bold")
        self._add_textbox("x_range", "x", "-2,2,16", 0.375, x=0.025)
        self._add_textbox("y_range", "y", "-2,2,16", 0.375, x=0.165)
        self._add_textbox("r_range", "r", "0,2,12", 0.325, x=0.025)
        self._add_textbox(
            "theta_range",
            "theta",
            f"0,{2 * math.pi:.8f},24",
            0.325,
            x=0.165,
        )

        self._add_button("Recalcular", 0.025, 0.265, self._on_recalculate)
        self._add_button("Caso base", 0.165, 0.265, self._on_reset)
        self._add_button("Guardar caso", 0.025, 0.215, self._on_save_case)
        self._add_button("Eliminar", 0.165, 0.215, self._on_remove_case)
        self._add_button("Importar JSON", 0.025, 0.165, self._on_import)
        self._add_button("Exportar JSON", 0.165, 0.165, self._on_export_json)

        clear_ax = self.fig.add_axes([0.025, 0.135, 0.125, 0.025])
        clear_button = Button(clear_ax, "Limpiar todo")
        clear_button.on_clicked(self._on_clear_session)
        self.widgets.append(clear_button)

        csv_ax = self.fig.add_axes([0.165, 0.135, 0.125, 0.025])
        csv_button = Button(csv_ax, "Exportar CSV")
        csv_button.on_clicked(self._on_export_csv)
        self.widgets.append(csv_button)

        self.fig.text(0.025, 0.110, "Casos", fontsize=8, weight="bold")
        self.fig.text(0.165, 0.110, "Tabla", fontsize=8, weight="bold")
        table_mode_ax = self.fig.add_axes([0.165, 0.020, 0.125, 0.082])
        self.table_mode_selector = RadioButtons(table_mode_ax, TABLE_MODES, active=0)
        self.table_mode_selector.on_clicked(self._request_redraw)
        self.widgets.append(self.table_mode_selector)

        self.fig.text(0.655, 0.435, "Strudel", fontsize=8, weight="bold")
        self._add_music_button("Generar", 0.655, 0.395, self._on_generate_music)
        self._add_music_button("Copiar", 0.655, 0.365, self._on_copy_strudel)
        self.music_criterion_selector = self._add_music_selector(
            MUSIC_CRITERIA,
            0.715,
            0.365,
            0.075,
            "Criterio",
        )
        self.music_segmentation_selector = self._add_music_selector(
            MUSIC_SEGMENTATIONS,
            0.795,
            0.365,
            0.075,
            "Mapa",
        )
        self.music_scale_selector = self._add_music_selector(
            MUSIC_SCALES,
            0.875,
            0.365,
            0.105,
            "Escala",
            active=1,
        )
        self.strudel_box = TextBox(self.strudel_ax, "", initial="silence")
        self.widgets.append(self.strudel_box)

    def _add_textbox(self, key: str, label: str, initial: str, y: float, *, x: float) -> None:
        self.fig.text(x, y + 0.032, label, fontsize=7)
        box_ax = self.fig.add_axes([x, y, 0.125, 0.027])
        textbox = TextBox(box_ax, "", initial=initial)
        textbox.on_submit(self._request_redraw)
        self.textboxes[key] = textbox
        self.widgets.append(textbox)

    def _add_button(self, label: str, x: float, y: float, callback) -> None:
        button_ax = self.fig.add_axes([x, y, 0.125, 0.032])
        button = Button(button_ax, label)
        button.on_clicked(callback)
        self.widgets.append(button)

    def _add_music_button(self, label: str, x: float, y: float, callback) -> None:
        button_ax = self.fig.add_axes([x, y, 0.052, 0.025])
        button = Button(button_ax, label)
        button.label.set_fontsize(6)
        button.on_clicked(callback)
        self.widgets.append(button)

    def _add_music_selector(
        self,
        labels: tuple[str, ...],
        x: float,
        y: float,
        width: float,
        title: str,
        *,
        active: int = 0,
    ) -> RadioButtons:
        self.fig.text(x, 0.435, title, fontsize=6, weight="bold")
        selector_ax = self.fig.add_axes([x, y, width, 0.065])
        selector = RadioButtons(selector_ax, labels, active=active)
        for text in selector.labels:
            text.set_fontsize(5.5)
        selector.on_clicked(self._on_music_option_changed)
        self.widgets.append(selector)
        return selector

    def redraw(self, *_unused) -> None:
        """Recalculate only after the complete UI state is available."""

        if self._suspend_callbacks or self._is_redrawing:
            return

        self._is_redrawing = True
        try:
            if self._is_blank:
                self._draw_blank_state()
            else:
                current_config = self._current_config()
                self.case_runs = self._build_case_runs(current_config)
                reference_result = self.case_runs[0].result
                self._draw_plot(
                    reference_result.transformer.coordinate_system.axes,
                    reference_result.transformer.grid,
                    self.case_runs,
                )
                self._draw_parameter_selector(self.case_runs)
                self._draw_summary(self.case_runs)
                self._draw_table(self.case_runs)
                self._draw_case_list()
                self._draw_music(self.case_runs)
        except Exception as exc:
            self.case_runs = []
            self._message = f"Error: {exc}"
            self._draw_error(str(exc))
        finally:
            self._is_redrawing = False

        self.fig.canvas.draw_idle()

    def _request_redraw(self, *_unused) -> None:
        if not self._suspend_callbacks:
            self._is_blank = False
            self.redraw()

    @contextmanager
    def _paused_callbacks(self):
        previous = self._suspend_callbacks
        self._suspend_callbacks = True
        try:
            yield
        finally:
            self._suspend_callbacks = previous

    def _current_config(self) -> ExplorerConfig:
        sequence = self._selected_sequence()
        coordinate_label = self._selected_coordinates()
        recurrence_name = SEQUENCE_TO_CONFIG[sequence]

        if recurrence_name == "cosine":
            recurrence_params = {"x0": self._float_value("x0")}
        elif recurrence_name == "logistic":
            recurrence_params = {
                "x0": self._float_value("x0"),
                "r": self._float_value("r"),
            }
        else:
            recurrence_params = {
                "c_real": self._float_value("c_real"),
                "c_imag": self._float_value("c_imag"),
                "z0_real": self._float_value("z0_real"),
                "z0_imag": self._float_value("z0_imag"),
            }

        selected_axes = COORDINATE_AXES[coordinate_label]
        range_keys = {
            "x": "x_range",
            "y": "y_range",
            "r": "r_range",
            "theta": "theta_range",
        }
        ranges = {
            axis: self._axis_range_config(range_keys[axis])
            for axis in selected_axes
        }

        return ExplorerConfig(
            recurrence=RecurrenceConfig(name=recurrence_name, parameters=recurrence_params),
            analysis=AnalysisConfig(
                max_terms=self._int_value("max_terms"),
                tolerance=self._float_value("tolerance"),
                max_period=self._int_value("max_period"),
                transient_terms=self._int_value("transient_terms"),
                stability_ratio=0.98,
            ),
            discretization=DiscretizationConfig(
                coordinate_system=COORDINATE_TO_CONFIG[coordinate_label],
                ranges=ranges,
            ),
        )

    def _build_case_runs(self, current_config: ExplorerConfig) -> list[CaseRun]:
        current_signature = self._recurrence_signature(current_config.recurrence)
        stored_signatures = {
            self._recurrence_signature(case.recurrence) for case in self.cases
        }
        cases_to_run = list(self.cases)
        if current_signature not in stored_signatures:
            cases_to_run.insert(
                0,
                ExplorationCase(
                    identifier="__actual__",
                    label="Actual",
                    recurrence=current_config.recurrence,
                    source="working_copy",
                ),
            )

        runs: list[CaseRun] = []
        failures: list[str] = []
        for case in cases_to_run:
            config = ExplorerConfig(
                recurrence=case.recurrence,
                analysis=current_config.analysis,
                discretization=current_config.discretization,
            )
            try:
                result = self._run_cached(config)
            except Exception as exc:
                failures.append(f"{case.label}: {exc}")
                continue
            runs.append(
                CaseRun(
                    case=case,
                    result=result,
                    is_active=(
                        case.identifier == self.active_case_id
                        or case.identifier == "__actual__"
                    ),
                )
            )

        if not runs:
            detail = "; ".join(failures) or "No hay casos calculables"
            raise ValueError(detail)
        if failures:
            self._message = "Algunos casos no se pudieron calcular: " + "; ".join(failures)
        return runs

    def _run_cached(self, config: ExplorerConfig) -> DiscreteAnalysisResult:
        key = repr(config)
        cached = self._result_cache.get(key)
        if cached is not None:
            return cached
        result = run_exploration(config)
        if len(self._result_cache) >= 128:
            self._result_cache.clear()
        self._result_cache[key] = result
        return result

    def _draw_plot(self, axes, grid, runs: list[CaseRun]) -> None:
        projection = "polar" if axes == ("r", "theta") else None
        if self.plot_ax is not None:
            self.plot_ax.remove()
        self.plot_ax = self.fig.add_axes([0.32, 0.45, 0.305, 0.50], projection=projection)

        if axes == ("r", "theta"):
            self._draw_polar(grid, runs)
        elif len(axes) == 2:
            self._draw_cartesian(axes, grid, runs)
        else:
            self._draw_one_dimensional(axes[0], grid, runs)

    def _draw_cartesian(self, axes, grid, runs: list[CaseRun]) -> None:
        x_axis, y_axis = axes
        x_subdivision = grid.subdivisions[x_axis]
        y_subdivision = grid.subdivisions[y_axis]
        for edge in x_subdivision.edges():
            self.plot_ax.axvline(edge, color="0.88", linewidth=0.55, zorder=0)
        for edge in y_subdivision.edges():
            self.plot_ax.axhline(edge, color="0.88", linewidth=0.55, zorder=0)

        colors = plt.get_cmap("tab10")
        for index, run in enumerate(runs):
            color = colors(index % 10)
            inside = [item for item in run.result.terms if item.point.inside_domain]
            if inside:
                xs = [item.point.coordinates[x_axis] for item in inside]
                ys = [item.point.coordinates[y_axis] for item in inside]
                self.plot_ax.plot(
                    xs,
                    ys,
                    color=color,
                    linewidth=1.15 if run.is_active else 0.75,
                    alpha=0.82 if run.is_active else 0.42,
                    zorder=2,
                )
                self.plot_ax.scatter(
                    xs,
                    ys,
                    color=color,
                    s=13 if run.is_active else 8,
                    alpha=0.80 if run.is_active else 0.45,
                    linewidths=0,
                    zorder=3,
                    label=run.case.label if len(runs) <= 6 else None,
                )
                self.plot_ax.scatter(
                    [xs[0]],
                    [ys[0]],
                    marker="o",
                    color=color,
                    edgecolor="black",
                    s=45,
                    linewidth=0.6,
                    zorder=5,
                )
            self._draw_attractors(run, x_axis, y_axis, color)

        self.plot_ax.set_xlim(x_subdivision.range.minimum, x_subdivision.range.maximum)
        self.plot_ax.set_ylim(y_subdivision.range.minimum, y_subdivision.range.maximum)
        self.plot_ax.set_xlabel(x_axis)
        self.plot_ax.set_ylabel(y_axis)
        self.plot_ax.set_title("Trayectorias y atractores discretizados")
        self.plot_ax.set_aspect("equal", adjustable="box")
        if len(runs) <= 6 and runs:
            self.plot_ax.legend(loc="best", fontsize=7)

    def _draw_polar(self, grid, runs: list[CaseRun]) -> None:
        r_subdivision = grid.subdivisions["r"]
        theta_subdivision = grid.subdivisions["theta"]
        self.plot_ax.set_ylim(r_subdivision.range.minimum, r_subdivision.range.maximum)
        self.plot_ax.set_theta_zero_location("E")
        self.plot_ax.set_theta_direction(1)
        self.plot_ax.grid(True, alpha=0.35)
        self.plot_ax.set_rticks(r_subdivision.edges()[1:])
        self.plot_ax.set_xticks(theta_subdivision.edges()[:-1])

        colors = plt.get_cmap("tab10")
        for index, run in enumerate(runs):
            color = colors(index % 10)
            inside = [item for item in run.result.terms if item.point.inside_domain]
            if inside:
                theta = [item.point.coordinates["theta"] for item in inside]
                radius = [item.point.coordinates["r"] for item in inside]
                self.plot_ax.plot(
                    theta,
                    radius,
                    color=color,
                    linewidth=1.15 if run.is_active else 0.75,
                    alpha=0.82 if run.is_active else 0.42,
                    zorder=2,
                )
                self.plot_ax.scatter(
                    theta,
                    radius,
                    color=color,
                    s=13 if run.is_active else 8,
                    alpha=0.80 if run.is_active else 0.45,
                    linewidths=0,
                    zorder=3,
                )
                self.plot_ax.scatter(
                    [theta[0]],
                    [radius[0]],
                    marker="o",
                    color=color,
                    edgecolor="black",
                    s=45,
                    linewidth=0.6,
                    zorder=5,
                )
            self._draw_attractors(run, "theta", "r", color)

        self.plot_ax.set_title("Trayectorias polares y atractores")

    def _draw_one_dimensional(self, axis, grid, runs: list[CaseRun]) -> None:
        subdivision = grid.subdivisions[axis]
        for edge in subdivision.edges():
            self.plot_ax.axhline(edge, color="0.88", linewidth=0.55, zorder=0)

        colors = plt.get_cmap("tab10")
        for index, run in enumerate(runs):
            color = colors(index % 10)
            inside = [item for item in run.result.terms if item.point.inside_domain]
            if inside:
                ns = [item.term.index for item in inside]
                values = [item.point.coordinates[axis] for item in inside]
                self.plot_ax.plot(
                    ns,
                    values,
                    color=color,
                    linewidth=1.15 if run.is_active else 0.75,
                    alpha=0.82 if run.is_active else 0.42,
                    zorder=2,
                    label=run.case.label if len(runs) <= 6 else None,
                )
                self.plot_ax.scatter(
                    ns,
                    values,
                    color=color,
                    s=12 if run.is_active else 7,
                    alpha=0.80 if run.is_active else 0.45,
                    linewidths=0,
                    zorder=3,
                )
                self.plot_ax.scatter(
                    [ns[0]],
                    [values[0]],
                    marker="o",
                    color=color,
                    edgecolor="black",
                    s=45,
                    linewidth=0.6,
                    zorder=5,
                )

            attractor_values = [
                point.coordinates[axis]
                for attractor in run.result.attractors
                for point in attractor.points
                if point.inside_domain
            ]
            if attractor_values:
                start = max(0, len(run.result.terms) - len(attractor_values))
                self.plot_ax.scatter(
                    range(start, start + len(attractor_values)),
                    attractor_values,
                    marker="D",
                    color=color,
                    edgecolor="black",
                    s=55,
                    linewidth=0.6,
                    zorder=6,
                )

        self.plot_ax.set_ylim(subdivision.range.minimum, subdivision.range.maximum)
        self.plot_ax.set_xlabel("n")
        self.plot_ax.set_ylabel(axis)
        self.plot_ax.set_title(f"Proyeccion sobre {axis}: puntos y atractores")
        if len(runs) <= 6 and runs:
            self.plot_ax.legend(loc="best", fontsize=7)

    def _draw_attractors(self, run: CaseRun, x_axis: str, y_axis: str, color) -> None:
        xs = []
        ys = []
        for attractor in run.result.attractors:
            for point in attractor.points:
                if point.inside_domain:
                    xs.append(point.coordinates[x_axis])
                    ys.append(point.coordinates[y_axis])
        if xs:
            self.plot_ax.scatter(
                xs,
                ys,
                marker="D",
                s=62,
                color=color,
                edgecolor="black",
                linewidth=0.65,
                zorder=6,
            )

    def _draw_parameter_selector(self, runs: list[CaseRun]) -> None:
        self.parameter_ax.clear()
        selected = self._selected_sequence()
        if selected == "logistica":
            self._draw_bifurcation_selector(runs)
        elif selected == "z^2+c":
            self._draw_mandelbrot_selector(runs)
        else:
            self.parameter_ax.axis("off")
            self.parameter_ax.text(
                0.5,
                0.55,
                "cos(x)",
                ha="center",
                va="center",
                fontsize=14,
                weight="bold",
                color="0.30",
            )
            self.parameter_ax.text(
                0.5,
                0.42,
                "El punto inicial y el atractor se muestran en el plano.",
                ha="center",
                va="center",
                fontsize=8,
                color="0.40",
            )

    def _draw_bifurcation_selector(self, runs: list[CaseRun]) -> None:
        limits = self._map_limits["logistica"]
        r_values, x_values = self._bifurcation_data(limits[0], limits[1])
        self.parameter_ax.scatter(
            r_values,
            x_values,
            s=0.11,
            color="black",
            alpha=0.34,
            linewidths=0,
            rasterized=True,
        )

        colors = plt.get_cmap("tab10")
        for index, run in enumerate(runs):
            color = colors(index % 10)
            recurrence = run.case.recurrence
            marker = "o"
            point = None
            if recurrence.name == "logistic":
                point = (
                    recurrence.parameters.get("r", 0.0),
                    recurrence.parameters.get("x0", 0.0),
                )
            elif recurrence.name == "quadratic_complex":
                converted = quadratic_to_logistic(
                    complex(
                        recurrence.parameters.get("c_real", 0.0),
                        recurrence.parameters.get("c_imag", 0.0),
                    ),
                    complex(
                        recurrence.parameters.get("z0_real", 0.0),
                        recurrence.parameters.get("z0_imag", 0.0),
                    ),
                )
                if converted is not None:
                    point = (converted.r, converted.x0)
                    marker = "x"
            if point is not None:
                self.parameter_ax.scatter(
                    [point[0]],
                    [point[1]],
                    marker=marker,
                    s=52,
                    color=color,
                    edgecolor="white" if marker == "o" else None,
                    linewidth=0.8,
                    zorder=6,
                )

        selected_r = self._optional_float_value("r")
        selected_x = self._optional_float_value("x0")
        if selected_r is not None:
            self.parameter_ax.axvline(
                selected_r,
                color="#d62728",
                linewidth=0.9,
                alpha=0.75,
            )
        if selected_x is not None:
            self.parameter_ax.axhline(
                selected_x,
                color="#d62728",
                linewidth=0.7,
                alpha=0.55,
            )
        self.parameter_ax.set_xlim(limits[0], limits[1])
        self.parameter_ax.set_ylim(limits[2], limits[3])
        self.parameter_ax.set_title("Diagrama de bifurcacion (rueda para ampliar)", fontsize=10)
        self.parameter_ax.set_xlabel("r")
        self.parameter_ax.set_ylabel("x")

    def _draw_mandelbrot_selector(self, runs: list[CaseRun]) -> None:
        limits = self._map_limits["z^2+c"]
        image, extent = self._mandelbrot_image(limits)
        self.parameter_ax.imshow(
            image,
            extent=extent,
            origin="lower",
            cmap="magma",
            aspect="equal",
            interpolation="nearest",
        )

        colors = plt.get_cmap("tab10")
        for index, run in enumerate(runs):
            color = colors(index % 10)
            recurrence = run.case.recurrence
            marker = "o"
            c_value = None
            if recurrence.name == "quadratic_complex":
                c_value = complex(
                    recurrence.parameters.get("c_real", 0.0),
                    recurrence.parameters.get("c_imag", 0.0),
                )
            elif recurrence.name == "logistic":
                converted = logistic_to_quadratic(
                    recurrence.parameters.get("r", 0.0),
                    recurrence.parameters.get("x0", 0.0),
                )
                c_value = converted.c
                marker = "x"
            if c_value is not None:
                self.parameter_ax.scatter(
                    [c_value.real],
                    [c_value.imag],
                    marker=marker,
                    s=58,
                    color=color,
                    edgecolor="white" if marker == "o" else None,
                    linewidth=0.8,
                    zorder=6,
                )

        current_real = self._optional_float_value("c_real")
        current_imag = self._optional_float_value("c_imag")
        if current_real is not None and current_imag is not None:
            self.parameter_ax.scatter(
                [current_real],
                [current_imag],
                marker="+",
                s=115,
                color="white",
                linewidth=1.1,
                zorder=7,
            )
        self.parameter_ax.set_xlim(limits[0], limits[1])
        self.parameter_ax.set_ylim(limits[2], limits[3])
        self.parameter_ax.set_title("Conjunto de Mandelbrot (rueda para ampliar)", fontsize=10)
        self.parameter_ax.set_xlabel("Re(c)")
        self.parameter_ax.set_ylabel("Im(c)")

    def _bifurcation_data(self, minimum: float, maximum: float) -> tuple[np.ndarray, np.ndarray]:
        key = (round(minimum, 12), round(maximum, 12))
        cached = self._bifurcation_cache.get(key)
        if cached is not None:
            return cached

        r_values = np.linspace(minimum, maximum, 1500)
        x_values = np.full_like(r_values, 0.5)
        plotted_r = []
        plotted_x = []
        for iteration in range(720):
            x_values = r_values * x_values * (1.0 - x_values)
            if iteration >= 420:
                plotted_r.append(r_values.copy())
                plotted_x.append(x_values.copy())

        result = (np.concatenate(plotted_r), np.concatenate(plotted_x))
        if len(self._bifurcation_cache) >= 10:
            self._bifurcation_cache.clear()
        self._bifurcation_cache[key] = result
        return result

    def _mandelbrot_image(
        self,
        limits: tuple[float, float, float, float],
    ) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        key = tuple(round(value, 12) for value in limits)
        cached = self._mandelbrot_cache.get(key)
        if cached is not None:
            return cached

        x_min, x_max, y_min, y_max = limits
        x_span = x_max - x_min
        y_span = y_max - y_min
        width = 760
        height = max(420, min(700, int(width * y_span / x_span)))
        zoom_level = max(0.0, math.log2(3.0 / x_span))
        iterations = min(280, 150 + int(zoom_level * 24))

        real = np.linspace(x_min, x_max, width)
        imag = np.linspace(y_min, y_max, height)
        real_grid, imag_grid = np.meshgrid(real, imag)
        c_values = real_grid + 1j * imag_grid
        z_values = np.zeros_like(c_values)
        active = np.ones(c_values.shape, dtype=bool)
        escape_counts = np.full(c_values.shape, iterations, dtype=float)

        for iteration in range(iterations):
            z_values[active] = z_values[active] * z_values[active] + c_values[active]
            escaped = active & (np.abs(z_values) > 2.0)
            escape_counts[escaped] = iteration
            active[escaped] = False
            if not active.any():
                break

        result = (escape_counts, limits)
        if len(self._mandelbrot_cache) >= 8:
            self._mandelbrot_cache.clear()
        self._mandelbrot_cache[key] = result
        return result

    def _on_button_press(self, event) -> None:
        if event.inaxes is self.parameter_ax:
            self._on_parameter_click(event)
        elif event.inaxes is self.case_list_ax:
            self._on_case_list_click(event)

    def _on_parameter_click(self, event) -> None:
        if event.xdata is None or event.ydata is None or not self._is_primary_click(event):
            return
        if self._toolbar_is_active():
            return

        self._is_blank = False
        selected = self._selected_sequence()
        if selected == "logistica":
            r_value = min(4.0, max(0.0, float(event.xdata)))
            x_value = min(1.0, max(0.0, float(event.ydata)))
            converted = logistic_to_quadratic(r_value, x_value)
            with self._paused_callbacks():
                self._set_text("r", r_value)
                self._set_text("x0", x_value)
                self._set_text("c_real", converted.c.real)
                self._set_text("c_imag", converted.c.imag)
                self._set_text("z0_real", converted.z0.real)
                self._set_text("z0_imag", converted.z0.imag)
            self._store_logistic_pair(
                r_value,
                x_value,
                source="bifurcation_click",
                force_new=True,
            )
            self._message = "Punto logistico y su orbita compleja vinculada guardados."
            self.redraw()
        elif selected == "z^2+c":
            bounds = self._MAP_BOUNDS["z^2+c"]
            real_value = min(bounds[1], max(bounds[0], float(event.xdata)))
            imag_value = min(bounds[3], max(bounds[2], float(event.ydata)))
            with self._paused_callbacks():
                self._set_text("c_real", real_value)
                self._set_text("c_imag", imag_value)
                self._set_text("z0_real", 0.0)
                self._set_text("z0_imag", 0.0)
            complex_case = self._store_case(
                RecurrenceConfig(
                    name="quadratic_complex",
                    parameters={
                        "c_real": real_value,
                        "c_imag": imag_value,
                        "z0_real": 0.0,
                        "z0_imag": 0.0,
                    },
                ),
                source="mandelbrot_click",
                force_new=True,
                make_active=True,
            )
            converted = quadratic_to_logistic(complex(real_value, imag_value))
            if converted is not None:
                self._store_case(
                    RecurrenceConfig(
                        name="logistic",
                        parameters={"r": converted.r, "x0": converted.x0},
                    ),
                    source="mandelbrot_real_axis",
                    metadata={"linked_case": complex_case.identifier},
                    force_new=False,
                    make_active=False,
                )
            self._message = "Parametro complejo guardado y calculado."
            self.redraw()

    def _on_scroll(self, event) -> None:
        if event.inaxes in {self.table_ax, self.table_status_ax}:
            self._on_table_wheel(event)
            return
        if event.inaxes is self.case_list_ax:
            self._on_case_list_wheel(event)
            return
        if event.inaxes is self.parameter_ax:
            self._zoom_parameter_map(event)

    def _zoom_parameter_map(self, event) -> None:
        if event.xdata is None or event.ydata is None:
            return
        selected = self._selected_sequence()
        if selected not in self._MAP_BOUNDS:
            return
        step = getattr(event, "step", 0)
        if step == 0:
            return
        scale = 1.0 / 1.55 if step > 0 else 1.55
        limits = self._map_limits[selected]
        bounds = self._MAP_BOUNDS[selected]
        self._map_limits[selected] = self._zoom_limits(
            limits,
            (float(event.xdata), float(event.ydata)),
            scale,
            bounds,
        )
        self._draw_parameter_selector(self.case_runs)
        self.fig.canvas.draw_idle()

    def _zoom_limits(self, limits, focus, scale, bounds):
        x_min, x_max, y_min, y_max = limits
        bound_x_min, bound_x_max, bound_y_min, bound_y_max = bounds
        new_x_span = max(1e-7, min(bound_x_max - bound_x_min, (x_max - x_min) * scale))
        new_y_span = max(1e-7, min(bound_y_max - bound_y_min, (y_max - y_min) * scale))
        new_x_min, new_x_max = self._center_limited_range(
            focus[0], new_x_span, bound_x_min, bound_x_max
        )
        new_y_min, new_y_max = self._center_limited_range(
            focus[1], new_y_span, bound_y_min, bound_y_max
        )
        return (new_x_min, new_x_max, new_y_min, new_y_max)

    def _center_limited_range(self, focus, span, lower, upper):
        center = min(upper - span / 2.0, max(lower + span / 2.0, focus))
        return center - span / 2.0, center + span / 2.0

    def _draw_blank_state(self) -> None:
        self.case_runs = []
        if self.plot_ax is not None:
            self.plot_ax.remove()
        self.plot_ax = self.fig.add_axes([0.32, 0.45, 0.305, 0.50])
        self.plot_ax.axis("off")
        self.plot_ax.text(
            0.5,
            0.54,
            "Sesion vacia",
            ha="center",
            va="center",
            fontsize=14,
            weight="bold",
            color="0.35",
        )
        self.plot_ax.text(
            0.5,
            0.44,
            "Selecciona un punto en el mapa o introduce nuevos parametros.",
            ha="center",
            va="center",
            fontsize=8,
            color="0.45",
        )
        self._draw_parameter_selector([])
        self.summary_ax.clear()
        self.summary_ax.axis("off")
        self.summary_ax.text(
            0.0,
            0.95,
            "Sesion vacia. Los casos, sucesiones, atractores y datos discretos se han eliminado.",
            va="top",
            ha="left",
            fontsize=8,
            color="0.35",
        )
        self.table_rows_cache = []
        self.table_start = 0
        self.table_max_start = 0
        self._render_table_window(0)
        self._draw_case_list()
        self._draw_music([])

    def _draw_music(self, runs: list[CaseRun]) -> None:
        result_pairs = [(run.case.label, run.result) for run in runs]
        self.music_pattern = generate_strudel_pattern(result_pairs, self._music_config())
        if self.strudel_box is not None:
            self.strudel_box.set_val(self.music_pattern.code)

        self.music_info_ax.clear()
        self.music_info_ax.axis("off")
        lines = [self.music_pattern.description]
        for attractor in self.music_pattern.attractors[:2]:
            cells = ",".join(
                "-" if cell is None else str(cell) for cell in attractor.cells
            )
            notes = " ".join(attractor.notes)
            lines.append(
                f"{attractor.case_label} A{attractor.attractor_index} "
                f"p={attractor.period} [{cells}] -> {notes}"
            )
        self.music_info_ax.text(
            0.0,
            0.98,
            "\n".join(lines),
            va="top",
            ha="left",
            fontsize=6.2,
            family="monospace",
            clip_on=True,
        )

    def _music_config(self) -> MusicMappingConfig:
        criterion = "melodia"
        segmentation = "franjas"
        scale = "menor"
        if self.music_criterion_selector is not None:
            criterion = self.music_criterion_selector.value_selected
        if self.music_segmentation_selector is not None:
            segmentation = self.music_segmentation_selector.value_selected
        if self.music_scale_selector is not None:
            scale = self.music_scale_selector.value_selected
        return MusicMappingConfig(
            criterion=MUSIC_CRITERION_TO_CONFIG[criterion],
            segmentation=MUSIC_SEGMENTATION_TO_CONFIG[segmentation],
            scale=MUSIC_SCALE_TO_CONFIG[scale],
            root="C4",
        )

    def _music_view_data(self) -> dict[str, str]:
        return {
            "criterion": self.music_criterion_selector.value_selected
            if self.music_criterion_selector is not None
            else "melodia",
            "segmentation": self.music_segmentation_selector.value_selected
            if self.music_segmentation_selector is not None
            else "franjas",
            "scale": self.music_scale_selector.value_selected
            if self.music_scale_selector is not None
            else "menor",
        }

    def _apply_music_view_data(self, data: dict[str, str]) -> None:
        criterion = data.get("criterion", "melodia")
        segmentation = data.get("segmentation", "franjas")
        scale = data.get("scale", "menor")
        self._set_radio(
            self.music_criterion_selector,
            MUSIC_CRITERIA,
            criterion,
        )
        self._set_radio(
            self.music_segmentation_selector,
            MUSIC_SEGMENTATIONS,
            segmentation,
        )
        self._set_radio(self.music_scale_selector, MUSIC_SCALES, scale)

    def _on_music_option_changed(self, _label: str) -> None:
        if self._is_blank:
            self._draw_music([])
        else:
            self._draw_music(self.case_runs)
        self.fig.canvas.draw_idle()

    def _on_generate_music(self, _event) -> None:
        self._draw_music([] if self._is_blank else self.case_runs)
        self._message = "Patron Strudel generado a partir de los atractores."
        self._draw_summary(self.case_runs)
        self.fig.canvas.draw_idle()

    def _on_copy_strudel(self, _event) -> None:
        code = self.music_pattern.code if self.music_pattern is not None else "silence"
        try:
            import tkinter as tk

            root = tk.Tk()
            root.withdraw()
            root.clipboard_clear()
            root.clipboard_append(code)
            root.update()
            root.destroy()
            self._message = "Comando Strudel copiado al portapapeles."
        except Exception as exc:
            self._message = f"No se pudo copiar Strudel: {exc}"
        self._draw_summary(self.case_runs)
        self.fig.canvas.draw_idle()

    def _draw_summary(self, runs: list[CaseRun]) -> None:
        self.summary_ax.clear()
        self.summary_ax.axis("off")
        if self._message:
            self.summary_ax.text(
                0.0,
                1.0,
                self._message,
                va="top",
                ha="left",
                fontsize=8,
                color="#8a3b00" if not self._message.startswith("Error:") else "#b00020",
            )

        y = 0.75 if self._message else 0.98
        self.summary_ax.text(
            0.0,
            y,
            f"Casos calculados: {len(runs)} | sistema: {self._selected_coordinates()}",
            va="top",
            ha="left",
            fontsize=9,
            weight="bold",
        )
        for run in runs[:5]:
            sequence = run.result.sequence_result
            attractors = ", ".join(
                f"p={item.attractor.period}" for item in run.result.attractors
            ) or "sin atractor detectado"
            self.summary_ax.text(
                0.0,
                y - 0.18 * (runs.index(run) + 1),
                f"{run.case.label}: {sequence.behavior}; {attractors}; n={len(sequence.terms)}",
                va="top",
                ha="left",
                fontsize=8,
                family="monospace",
            )

    def _draw_table(self, runs: list[CaseRun]) -> None:
        self.visible_table_rows = max(1, self._int_value("table_rows"))
        table_mode = self._selected_table_mode()
        rows = []

        for run in runs:
            if table_mode in {"todo", "sucesion"}:
                for discrete_term in run.result.terms:
                    point = discrete_term.point
                    rows.append(
                        [
                            run.case.label,
                            "S",
                            str(discrete_term.term.index),
                            self._format_number(point.value),
                            self._format_mapping(point.coordinates),
                            self._format_mapping(point.indices),
                            self._format_mapping(point.discrete_coordinates),
                            self._format_optional(point.cell_id),
                        ]
                    )
            if table_mode in {"todo", "atractores"}:
                for attractor_index, discrete_attractor in enumerate(
                    run.result.attractors,
                    start=1,
                ):
                    for point_index, point in enumerate(discrete_attractor.points, start=1):
                        rows.append(
                            [
                                run.case.label,
                                f"A{attractor_index}.{point_index}",
                                "-",
                                self._format_number(point.value),
                                self._format_mapping(point.coordinates),
                                self._format_mapping(point.indices),
                                self._format_mapping(point.discrete_coordinates),
                                self._format_optional(point.cell_id),
                            ]
                        )

        self.table_rows_cache = rows
        self.table_max_start = max(0, len(rows) - self.visible_table_rows)
        self.table_start = self.table_max_start
        self._render_table_window(self.table_start)

    def _on_table_wheel(self, event) -> None:
        if not self.table_rows_cache or self.table_max_start <= 0:
            return
        step = getattr(event, "step", 0)
        if step == 0:
            step = 1 if getattr(event, "button", "") == "up" else -1
        rows_per_tick = max(1, self.visible_table_rows // 2)
        new_start = self.table_start - int(step) * rows_per_tick
        new_start = max(0, min(self.table_max_start, new_start))
        if new_start == self.table_start:
            return
        self.table_start = new_start
        self._render_table_window(self.table_start)
        self.fig.canvas.draw_idle()

    def _render_table_window(self, start: int) -> None:
        self.table_ax.clear()
        self.table_ax.axis("off")
        rows = self.table_rows_cache[start : start + self.visible_table_rows]
        self._draw_table_status(start, len(rows))
        if not rows:
            self.table_ax.text(0.0, 0.5, "Tabla discreta sin datos.", va="center", fontsize=9)
            return

        headers = ["caso", "tipo", "n", "valor", "coord", "idx", "centro", "celda"]
        table = self.table_ax.table(
            cellText=rows,
            colLabels=headers,
            loc="center",
            cellLoc="left",
            colLoc="left",
            colWidths=[0.11, 0.065, 0.055, 0.15, 0.20, 0.12, 0.22, 0.075],
            bbox=[0.0, 0.0, 1.0, 1.0],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(6.7)
        for (row, _column), cell in table.get_celld().items():
            cell.set_edgecolor("0.82")
            cell.set_linewidth(0.45)
            if row == 0:
                cell.set_facecolor("0.92")
                cell.set_text_props(weight="bold")

    def _draw_table_status(self, start: int, visible_count: int) -> None:
        self.table_status_ax.clear()
        self.table_status_ax.axis("off")
        total = len(self.table_rows_cache)
        status = "Tabla sin datos" if total == 0 else f"Filas {start + 1}-{start + visible_count} de {total}"
        self.table_status_ax.text(0.0, 0.5, status, va="center", ha="left", fontsize=8, color="0.35")

    def _draw_case_list(self) -> None:
        self.case_list_ax.clear()
        self.case_list_ax.axis("off")
        maximum_start = max(0, len(self.cases) - 5)
        self._case_list_start = min(maximum_start, max(0, self._case_list_start))
        visible_cases = self.cases[self._case_list_start : self._case_list_start + 5]
        self._visible_case_ids = [case.identifier for case in visible_cases]
        if not visible_cases:
            self.case_list_ax.text(0.0, 0.5, "Sin casos", va="center", fontsize=7)
            return

        self.case_list_ax.set_xlim(0.0, 1.0)
        self.case_list_ax.set_ylim(0.0, len(visible_cases))
        colors = plt.get_cmap("tab10")
        for index, case in enumerate(visible_cases):
            y = len(visible_cases) - index - 0.5
            is_active = case.identifier == self.active_case_id
            self.case_list_ax.text(
                0.0,
                y,
                case.label,
                va="center",
                fontsize=7,
                color=colors(index % 10),
                weight="bold" if is_active else "normal",
            )

    def _on_case_list_click(self, event) -> None:
        if event.ydata is None or not self._is_primary_click(event):
            return
        count = len(self._visible_case_ids)
        index = count - 1 - int(event.ydata)
        if not 0 <= index < count:
            return
        identifier = self._visible_case_ids[index]
        case = self._case_by_id(identifier)
        if case is None:
            return
        self._is_blank = False
        self.active_case_id = case.identifier
        self._apply_recurrence(case.recurrence)
        self._message = f"Caso activo: {case.label}"
        self.redraw()

    def _on_case_list_wheel(self, event) -> None:
        maximum_start = max(0, len(self.cases) - 5)
        if maximum_start == 0:
            return
        step = getattr(event, "step", 0)
        if step == 0:
            step = 1 if getattr(event, "button", "") == "up" else -1
        new_start = self._case_list_start - int(step)
        new_start = min(maximum_start, max(0, new_start))
        if new_start == self._case_list_start:
            return
        self._case_list_start = new_start
        self._draw_case_list()
        self.fig.canvas.draw_idle()

    def _on_sequence_selected(self, _label: str) -> None:
        if self._suspend_callbacks:
            return
        if (
            not self._is_blank
            and _label == "z^2+c"
            and self._last_sequence == "logistica"
        ):
            r_value = self._float_value("r")
            x_value = self._float_value("x0")
            converted = logistic_to_quadratic(r_value, x_value)
            with self._paused_callbacks():
                self._set_text("c_real", converted.c.real)
                self._set_text("c_imag", converted.c.imag)
                self._set_text("z0_real", converted.z0.real)
                self._set_text("z0_imag", converted.z0.imag)
            self._store_logistic_pair(
                r_value,
                x_value,
                source="logistic_isomorphism",
                force_new=False,
            )
        self._last_sequence = _label
        self.active_case_id = None
        self._message = ""
        self.redraw()

    def _on_coordinate_selected(self, _label: str) -> None:
        self._request_redraw()

    def _on_recalculate(self, _event) -> None:
        self._is_blank = False
        self._message = ""
        self.redraw()

    def _on_reset(self, _event) -> None:
        self._is_blank = False
        self.active_case_id = None
        self._apply_config(default_concept_config())
        self._message = "Caso base aplicado."
        self.redraw()

    def _on_clear_session(self, _event) -> None:
        self.cases = []
        self.case_runs = []
        self.active_case_id = None
        self._case_counter = 0
        self._case_list_start = 0
        self._visible_case_ids = []
        self._result_cache.clear()
        self._bifurcation_cache.clear()
        self._mandelbrot_cache.clear()
        self._map_limits = dict(self._MAP_BOUNDS)
        self.table_rows_cache = []
        self.table_start = 0
        self.table_max_start = 0
        with self._paused_callbacks():
            for key in ("x0", "r", "c_real", "c_imag", "z0_real", "z0_imag"):
                self.textboxes[key].set_val("")
            self._set_radio(self.sequence_selector, SEQUENCES, "logistica")
            self._set_radio(self.coordinate_selector, COORDINATES, "cartesiano")
            self._set_radio(self.table_mode_selector, TABLE_MODES, "todo")
        self._last_sequence = "logistica"
        self._is_blank = True
        self._message = "Sesion vacia."
        self.redraw()

    def _on_save_case(self, _event) -> None:
        try:
            self._is_blank = False
            recurrence = self._current_config().recurrence
            case = self._store_case(recurrence, source="manual", force_new=False, make_active=True)
            if recurrence.name == "quadratic_complex":
                self._store_real_axis_logistic_pair(recurrence, case.identifier)
            self._message = f"Caso guardado: {case.label}"
            self.redraw()
        except Exception as exc:
            self._message = f"Error al guardar: {exc}"
            self.redraw()

    def _on_remove_case(self, _event) -> None:
        if self.active_case_id is None:
            self._message = "No hay un caso activo para eliminar."
            self.redraw()
            return
        case = self._case_by_id(self.active_case_id)
        self.cases = [item for item in self.cases if item.identifier != self.active_case_id]
        self.active_case_id = None
        self._message = f"Caso eliminado: {case.label}" if case is not None else "Caso eliminado."
        self.redraw()

    def _on_import(self, _event) -> None:
        path = self._choose_path(open_file=True, extension="json")
        if not path:
            return
        try:
            imported = read_session(path)
            self.cases = imported.cases
            self.active_case_id = imported.active_case_id
            self._map_limits = dict(self._MAP_BOUNDS)
            self._map_limits.update(imported.map_limits)
            self._case_counter = len(self.cases)
            self._case_list_start = max(0, len(self.cases) - 5)
            self._result_cache.clear()
            self._apply_config(imported.config)
            with self._paused_callbacks():
                self._set_radio(self.table_mode_selector, TABLE_MODES, imported.table_mode)
                self._apply_music_view_data(imported.music)
            self._is_blank = False
            self._message = f"Sesion importada: {len(self.cases)} casos."
        except Exception as exc:
            self._message = f"Error al importar: {exc}"
        self.redraw()

    def _on_export_json(self, _event) -> None:
        path = self._choose_path(open_file=False, extension="json")
        if not path:
            return
        try:
            self.redraw()
            data = build_session_data(
                self._current_config(),
                self.cases,
                [(run.case, run.result) for run in self.case_runs],
                active_case_id=self.active_case_id,
                table_mode=self._selected_table_mode(),
                map_limits=self._map_limits,
                music=self._music_view_data(),
            )
            write_session(path, data)
            self._message = "Sesion JSON exportada."
        except Exception as exc:
            self._message = f"Error al exportar JSON: {exc}"
        self.redraw()

    def _on_export_csv(self, _event) -> None:
        path = self._choose_path(open_file=False, extension="csv")
        if not path:
            return
        try:
            self.redraw()
            write_results_csv(path, [(run.case, run.result) for run in self.case_runs])
            self._message = "Resultados CSV exportados."
        except Exception as exc:
            self._message = f"Error al exportar CSV: {exc}"
        self.redraw()

    def _choose_path(self, *, open_file: bool, extension: str) -> str | None:
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            filetypes = [(f"Archivo {extension.upper()}", f"*.{extension}")]
            if open_file:
                path = filedialog.askopenfilename(filetypes=filetypes)
            else:
                path = filedialog.asksaveasfilename(
                    defaultextension=f".{extension}",
                    filetypes=filetypes,
                )
            root.destroy()
            return path or None
        except Exception as exc:
            self._message = f"No se pudo abrir el selector de archivos: {exc}"
            self.redraw()
            return None

    def _store_logistic_pair(
        self,
        r_value: float,
        x_value: float,
        *,
        source: str,
        force_new: bool,
    ) -> None:
        logistic_case = self._store_case(
            RecurrenceConfig(name="logistic", parameters={"r": r_value, "x0": x_value}),
            source=source,
            force_new=force_new,
            make_active=True,
        )
        converted = logistic_to_quadratic(r_value, x_value)
        self._store_case(
            RecurrenceConfig(
                name="quadratic_complex",
                parameters={
                    "c_real": converted.c.real,
                    "c_imag": converted.c.imag,
                    "z0_real": converted.z0.real,
                    "z0_imag": converted.z0.imag,
                },
            ),
            source="logistic_isomorphism",
            metadata={"linked_case": logistic_case.identifier},
            force_new=False,
            make_active=False,
        )
        self.active_case_id = logistic_case.identifier

    def _store_real_axis_logistic_pair(
        self,
        recurrence: RecurrenceConfig,
        linked_case_id: str,
    ) -> None:
        converted = quadratic_to_logistic(
            complex(
                recurrence.parameters.get("c_real", 0.0),
                recurrence.parameters.get("c_imag", 0.0),
            ),
            complex(
                recurrence.parameters.get("z0_real", 0.0),
                recurrence.parameters.get("z0_imag", 0.0),
            ),
        )
        if converted is None:
            return
        self._store_case(
            RecurrenceConfig(
                name="logistic",
                parameters={"r": converted.r, "x0": converted.x0},
            ),
            source="mandelbrot_real_axis",
            metadata={"linked_case": linked_case_id},
            force_new=False,
            make_active=False,
        )

    def _store_case(
        self,
        recurrence: RecurrenceConfig,
        *,
        source: str,
        metadata: dict | None = None,
        force_new: bool,
        make_active: bool,
    ) -> ExplorationCase:
        signature = self._recurrence_signature(recurrence)
        if not force_new:
            if self.active_case_id is not None:
                active = self._case_by_id(self.active_case_id)
                if active is not None and self._recurrence_signature(active.recurrence) == signature:
                    updated = ExplorationCase(
                        identifier=active.identifier,
                        label=active.label,
                        recurrence=recurrence,
                        source=source,
                        metadata=metadata or active.metadata,
                    )
                    self.cases = [
                        updated if item.identifier == updated.identifier else item
                        for item in self.cases
                    ]
                    if make_active:
                        self.active_case_id = updated.identifier
                    return updated
            for case in self.cases:
                if self._recurrence_signature(case.recurrence) == signature:
                    if make_active:
                        self.active_case_id = case.identifier
                    return case

        identifier = self._next_case_id()
        case = ExplorationCase(
            identifier=identifier,
            label=self._case_label(recurrence),
            recurrence=recurrence,
            source=source,
            metadata=metadata or {},
        )
        self.cases.append(case)
        self._case_list_start = max(0, len(self.cases) - 5)
        if make_active:
            self.active_case_id = case.identifier
        return case

    def _next_case_id(self) -> str:
        existing = {case.identifier for case in self.cases}
        while True:
            self._case_counter += 1
            identifier = f"case-{self._case_counter:03d}"
            if identifier not in existing:
                return identifier

    def _case_label(self, recurrence: RecurrenceConfig) -> str:
        number = self._case_counter
        params = recurrence.parameters
        if recurrence.name == "cosine":
            return f"C{number}: x0={params.get('x0', 0.0):.4g}"
        if recurrence.name == "logistic":
            return f"L{number}: r={params.get('r', 0.0):.5g}"
        return (
            f"Q{number}: c={params.get('c_real', 0.0):.4g}"
            f"{params.get('c_imag', 0.0):+.4g}j"
        )

    def _case_by_id(self, identifier: str) -> ExplorationCase | None:
        return next((case for case in self.cases if case.identifier == identifier), None)

    def _recurrence_signature(self, recurrence: RecurrenceConfig) -> tuple:
        return (
            recurrence.name,
            tuple(sorted((key, float(value)) for key, value in recurrence.parameters.items())),
        )

    def _apply_config(self, config: ExplorerConfig) -> None:
        sequence_label = CONFIG_TO_SEQUENCE.get(config.recurrence.name, "z^2+c")
        coordinate_label = CONFIG_TO_COORDINATE.get(
            config.discretization.coordinate_system,
            "polar",
        )
        with self._paused_callbacks():
            self._set_radio(self.sequence_selector, SEQUENCES, sequence_label)
            self._set_radio(self.coordinate_selector, COORDINATES, coordinate_label)
            self._apply_recurrence(config.recurrence, preserve_active=True)

            analysis = config.analysis
            self._set_text("max_terms", analysis.max_terms)
            self._set_text("tolerance", analysis.tolerance)
            self._set_text("max_period", analysis.max_period)
            self._set_text("transient_terms", analysis.transient_terms)

            range_keys = {
                "x": "x_range",
                "y": "y_range",
                "r": "r_range",
                "theta": "theta_range",
            }
            for axis, axis_config in config.discretization.ranges.items():
                self._set_text(
                    range_keys[axis],
                    f"{axis_config.minimum},{axis_config.maximum},{axis_config.steps}",
                )
            self._last_sequence = sequence_label

    def _apply_recurrence(self, recurrence: RecurrenceConfig, *, preserve_active: bool = False) -> None:
        sequence_label = CONFIG_TO_SEQUENCE.get(recurrence.name)
        if sequence_label is None:
            raise ValueError(f"Recurrencia no compatible con el visualizador: {recurrence.name}")
        with self._paused_callbacks():
            self._set_radio(self.sequence_selector, SEQUENCES, sequence_label)
            params = recurrence.parameters
            if recurrence.name == "cosine":
                self._set_text("x0", params.get("x0", 0.2))
            elif recurrence.name == "logistic":
                r_value = params.get("r", 3.2)
                x_value = params.get("x0", 0.2)
                self._set_text("r", r_value)
                self._set_text("x0", x_value)
                converted = logistic_to_quadratic(r_value, x_value)
                self._set_text("c_real", converted.c.real)
                self._set_text("c_imag", converted.c.imag)
                self._set_text("z0_real", converted.z0.real)
                self._set_text("z0_imag", converted.z0.imag)
            else:
                self._set_text("c_real", params.get("c_real", -0.123))
                self._set_text("c_imag", params.get("c_imag", 0.745))
                self._set_text("z0_real", params.get("z0_real", 0.0))
                self._set_text("z0_imag", params.get("z0_imag", 0.0))
            self._last_sequence = sequence_label

    def _set_radio(self, radio, labels, label) -> None:
        if radio is None or label not in labels or radio.value_selected == label:
            return
        radio.set_active(labels.index(label))

    def _set_text(self, key: str, value) -> None:
        self.textboxes[key].set_val(f"{value:.10g}" if isinstance(value, float) else str(value))

    def _is_primary_click(self, event) -> bool:
        return getattr(event, "button", 1) in {1, None}

    def _toolbar_is_active(self) -> bool:
        toolbar = getattr(self.fig.canvas, "toolbar", None)
        return bool(getattr(toolbar, "mode", ""))

    def _draw_error(self, message: str) -> None:
        if self.plot_ax is not None:
            self.plot_ax.remove()
        self.plot_ax = self.fig.add_axes([0.32, 0.45, 0.305, 0.50])
        self.plot_ax.axis("off")
        self.plot_ax.text(0.5, 0.5, message, ha="center", va="center", color="#b00020")
        self.parameter_ax.clear()
        self.parameter_ax.axis("off")
        self.summary_ax.clear()
        self.summary_ax.axis("off")
        self.summary_ax.text(0.0, 0.95, f"Error: {message}", va="top", fontsize=9, color="#b00020")
        self.table_ax.clear()
        self.table_ax.axis("off")
        self.table_status_ax.clear()
        self.table_status_ax.axis("off")
        self._draw_case_list()

    def _format_mapping(self, mapping) -> str:
        if mapping is None:
            return "-"
        return ", ".join(f"{key}={self._format_optional(value)}" for key, value in mapping.items())

    def _format_number(self, value) -> str:
        if isinstance(value, complex):
            return f"{value.real:.5g}{value.imag:+.5g}j"
        return f"{value:.5g}"

    def _format_optional(self, value) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.5g}"
        return str(value)

    def _selected_sequence(self) -> str:
        return SEQUENCES[0] if self.sequence_selector is None else self.sequence_selector.value_selected

    def _selected_coordinates(self) -> str:
        return COORDINATES[0] if self.coordinate_selector is None else self.coordinate_selector.value_selected

    def _selected_table_mode(self) -> str:
        return TABLE_MODES[0] if self.table_mode_selector is None else self.table_mode_selector.value_selected

    def _float_value(self, key: str) -> float:
        return float(self.textboxes[key].text)

    def _optional_float_value(self, key: str) -> float | None:
        text = self.textboxes[key].text.strip()
        return None if not text else float(text)

    def _int_value(self, key: str) -> int:
        return int(float(self.textboxes[key].text))

    def _axis_config(self, key: str) -> AxisConfig:
        parts = [part.strip() for part in self.textboxes[key].text.split(",")]
        if len(parts) != 3:
            raise ValueError(f"{key} debe usar el formato min,max,subdivisiones")
        minimum = float(parts[0])
        maximum = float(parts[1])
        steps = int(float(parts[2]))
        return AxisConfig(minimum=minimum, maximum=maximum, steps=steps)

    def _axis_range_config(self, key: str) -> AxisRangeConfig:
        axis_config = self._axis_config(key)
        return AxisRangeConfig(
            minimum=axis_config.minimum,
            maximum=axis_config.maximum,
            steps=axis_config.steps,
        )


def main() -> None:
    VisualExplorer()
    plt.show()


if __name__ == "__main__":
    main()
