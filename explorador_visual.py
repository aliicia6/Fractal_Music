"""Interactive visual explorer for recurrence sequences.

Run with:
    python explorador_visual.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, RadioButtons, TextBox

from fractal_sequences import (
    AnalysisConfig,
    AxisRangeConfig,
    DiscretizationConfig,
    ExplorerConfig,
    RecurrenceConfig,
    default_concept_config,
    run_exploration,
)


@dataclass(frozen=True)
class AxisConfig:
    minimum: float
    maximum: float
    steps: int


SEQUENCES = ("cos(x)", "logistica", "z^2+c")
COORDINATES = ("x", "cartesiano", "polar", "y", "radio", "angulo")
TABLE_MODES = ("todo", "sucesion", "atractores")
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


class VisualExplorer:
    """Small matplotlib application wired to the sequence framework."""

    def __init__(self) -> None:
        self.fig = plt.figure(figsize=(15, 9))
        self.fig.canvas.manager.set_window_title("Explorador de sucesiones")

        self.plot_ax = None
        self.parameter_ax = self.fig.add_axes([0.74, 0.38, 0.23, 0.57])
        self.parameter_ax.axis("off")
        self.summary_ax = self.fig.add_axes([0.34, 0.20, 0.63, 0.10])
        self.summary_ax.axis("off")
        self.table_ax = self.fig.add_axes([0.34, 0.045, 0.63, 0.125])
        self.table_ax.axis("off")
        self.table_status_ax = self.fig.add_axes([0.34, 0.015, 0.63, 0.025])
        self.table_status_ax.axis("off")
        self.table_rows_cache: list[list[str]] = []
        self.visible_table_rows = 8
        self.table_start = 0
        self.table_max_start = 0
        self._bifurcation_cache: tuple[np.ndarray, np.ndarray] | None = None
        self._mandelbrot_cache: tuple[np.ndarray, tuple[float, float, float, float]] | None = None

        self.widgets: list[object] = []
        self.textboxes: dict[str, TextBox] = {}
        self.sequence_selector: RadioButtons | None = None
        self.coordinate_selector: RadioButtons | None = None
        self.table_mode_selector: RadioButtons | None = None

        self._build_controls()
        self.fig.canvas.mpl_connect("scroll_event", self._on_table_wheel)
        self.fig.canvas.mpl_connect("button_press_event", self._on_parameter_click)
        self.redraw()

    def _build_controls(self) -> None:
        self.fig.text(0.025, 0.965, "Exploracion", fontsize=12, weight="bold")

        self.fig.text(0.03, 0.925, "Sucesion", fontsize=9, weight="bold")
        sequence_ax = self.fig.add_axes([0.03, 0.815, 0.12, 0.10])
        self.sequence_selector = RadioButtons(sequence_ax, SEQUENCES, active=1)
        self.sequence_selector.on_clicked(lambda _label: self.redraw())
        self.widgets.append(self.sequence_selector)

        self.fig.text(0.17, 0.925, "Coordenadas", fontsize=9, weight="bold")
        coordinate_ax = self.fig.add_axes([0.17, 0.785, 0.12, 0.13])
        self.coordinate_selector = RadioButtons(coordinate_ax, COORDINATES, active=1)
        self.coordinate_selector.on_clicked(lambda _label: self.redraw())
        self.widgets.append(self.coordinate_selector)

        self.fig.text(0.03, 0.755, "Parametros", fontsize=9, weight="bold")
        self._add_textbox("x0", "x0", "0.2", 0.71, x=0.03)
        self._add_textbox("r", "r", "3.2", 0.71, x=0.17)
        self._add_textbox("c_real", "c real", "-0.123", 0.655, x=0.03)
        self._add_textbox("c_imag", "c imag", "0.745", 0.655, x=0.17)
        self._add_textbox("z0_real", "z0 real", "0.0", 0.60, x=0.03)
        self._add_textbox("z0_imag", "z0 imag", "0.0", 0.60, x=0.17)

        self.fig.text(0.03, 0.555, "Analisis", fontsize=9, weight="bold")
        self._add_textbox("max_terms", "terminos", "1000", 0.51, x=0.03)
        self._add_textbox("tolerance", "tol", "1e-7", 0.51, x=0.17)
        self._add_textbox("max_period", "periodo", "16", 0.455, x=0.03)
        self._add_textbox("transient_terms", "transitorio", "100", 0.455, x=0.17)
        self._add_textbox("table_rows", "filas tabla", "8", 0.40, x=0.03)

        self.fig.text(0.03, 0.355, "Dominios: min,max,n", fontsize=9, weight="bold")
        self._add_textbox("x_range", "x", "-2,2,16", 0.31, x=0.03, box_width=0.105)
        self._add_textbox("y_range", "y", "-2,2,16", 0.31, x=0.17, box_width=0.105)
        self._add_textbox("r_range", "r", "0,2,12", 0.255, x=0.03, box_width=0.105)
        self._add_textbox(
            "theta_range",
            "theta",
            f"0,{2 * math.pi:.8f},24",
            0.255,
            x=0.17,
            box_width=0.105,
        )

        button_ax = self.fig.add_axes([0.03, 0.195, 0.12, 0.04])
        update_button = Button(button_ax, "Recalcular")
        update_button.on_clicked(lambda _event: self.redraw())
        self.widgets.append(update_button)

        reset_ax = self.fig.add_axes([0.17, 0.195, 0.12, 0.04])
        reset_button = Button(reset_ax, "Caso base")
        reset_button.on_clicked(lambda _event: self._reset_to_concept_case())
        self.widgets.append(reset_button)

        self.fig.text(0.03, 0.17, "Tabla", fontsize=9, weight="bold")
        table_mode_ax = self.fig.add_axes([0.03, 0.07, 0.26, 0.085])
        self.table_mode_selector = RadioButtons(table_mode_ax, TABLE_MODES, active=0)
        self.table_mode_selector.on_clicked(lambda _label: self.redraw())
        self.widgets.append(self.table_mode_selector)

    def _add_textbox(
        self,
        key: str,
        label: str,
        initial: str,
        y: float,
        x: float = 0.03,
        box_width: float = 0.105,
    ) -> None:
        self.fig.text(x, y + 0.038, label, fontsize=8)
        box_ax = self.fig.add_axes([x, y, box_width, 0.032])
        textbox = TextBox(box_ax, "", initial=initial)
        textbox.on_submit(lambda _text: self.redraw())
        self.textboxes[key] = textbox
        self.widgets.append(textbox)

    def redraw(self) -> None:
        try:
            discrete_result = run_exploration(self._current_config())
            result = discrete_result.sequence_result
            coordinate_system = discrete_result.transformer.coordinate_system
            grid = discrete_result.transformer.grid
            discrete_terms = discrete_result.terms
            discrete_attractors = discrete_result.attractors

            self._draw_plot(
                coordinate_system.axes,
                grid,
                discrete_terms,
                discrete_attractors,
            )
            self._draw_parameter_selector()
            self._draw_summary(result, discrete_terms, discrete_attractors)
            self._draw_table(discrete_terms, discrete_attractors)
        except Exception as exc:
            self._draw_error(str(exc))

        self.fig.canvas.draw_idle()

    def _current_config(self) -> ExplorerConfig:
        sequence = self._selected_sequence()
        coordinate_system = self._selected_coordinates()

        recurrence_params = {
            "x0": self._float_value("x0"),
            "r": self._float_value("r"),
            "c_real": self._float_value("c_real"),
            "c_imag": self._float_value("c_imag"),
            "z0_real": self._float_value("z0_real"),
            "z0_imag": self._float_value("z0_imag"),
        }
        ranges = {
            "x": self._axis_range_config("x_range"),
            "y": self._axis_range_config("y_range"),
            "r": self._axis_range_config("r_range"),
            "theta": self._axis_range_config("theta_range"),
        }

        return ExplorerConfig(
            recurrence=RecurrenceConfig(
                name=SEQUENCE_TO_CONFIG[sequence],
                parameters=recurrence_params,
            ),
            analysis=AnalysisConfig(
                max_terms=self._int_value("max_terms"),
                tolerance=self._float_value("tolerance"),
                max_period=self._int_value("max_period"),
                transient_terms=self._int_value("transient_terms"),
                stability_ratio=0.98,
            ),
            discretization=DiscretizationConfig(
                coordinate_system=COORDINATE_TO_CONFIG[coordinate_system],
                ranges=ranges,
            ),
        )

    def _reset_to_concept_case(self) -> None:
        self._apply_config(default_concept_config())
        self.redraw()

    def _apply_config(self, config: ExplorerConfig) -> None:
        sequence_label = CONFIG_TO_SEQUENCE.get(config.recurrence.name, "z^2+c")
        coordinate_label = CONFIG_TO_COORDINATE.get(
            config.discretization.coordinate_system,
            "polar",
        )
        self._set_radio(self.sequence_selector, SEQUENCES, sequence_label)
        self._set_radio(self.coordinate_selector, COORDINATES, coordinate_label)

        params = config.recurrence.parameters
        self.textboxes["x0"].set_val(str(params.get("x0", 0.2)))
        self.textboxes["r"].set_val(str(params.get("r", 3.2)))
        self.textboxes["c_real"].set_val(str(params.get("c_real", -0.123)))
        self.textboxes["c_imag"].set_val(str(params.get("c_imag", 0.745)))
        self.textboxes["z0_real"].set_val(str(params.get("z0_real", 0.0)))
        self.textboxes["z0_imag"].set_val(str(params.get("z0_imag", 0.0)))

        analysis = config.analysis
        self.textboxes["max_terms"].set_val(str(analysis.max_terms))
        self.textboxes["tolerance"].set_val(str(analysis.tolerance))
        self.textboxes["max_period"].set_val(str(analysis.max_period))
        self.textboxes["transient_terms"].set_val(str(analysis.transient_terms))

        for axis, key in [
            ("x", "x_range"),
            ("y", "y_range"),
            ("r", "r_range"),
            ("theta", "theta_range"),
        ]:
            if axis in config.discretization.ranges:
                axis_config = config.discretization.ranges[axis]
                self.textboxes[key].set_val(
                    f"{axis_config.minimum},{axis_config.maximum},{axis_config.steps}"
                )

    def _set_radio(self, radio, labels, label) -> None:
        if radio is None:
            return
        if radio.value_selected == label:
            return
        radio.set_active(labels.index(label))

    def _draw_plot(self, axes, grid, discrete_terms, discrete_attractors) -> None:
        projection = "polar" if axes == ("r", "theta") else None
        if self.plot_ax is not None:
            self.plot_ax.remove()
        self.plot_ax = self.fig.add_axes([0.34, 0.38, 0.37, 0.57], projection=projection)

        if len(axes) == 2 and axes == ("r", "theta"):
            self._draw_polar(grid, discrete_terms, discrete_attractors)
        elif len(axes) == 2:
            self._draw_cartesian(axes, grid, discrete_terms, discrete_attractors)
        else:
            self._draw_one_dimensional(axes[0], grid, discrete_terms, discrete_attractors)

    def _draw_cartesian(self, axes, grid, discrete_terms, discrete_attractors) -> None:
        x_axis, y_axis = axes
        x_subdivision = grid.subdivisions[x_axis]
        y_subdivision = grid.subdivisions[y_axis]

        for edge in x_subdivision.edges():
            self.plot_ax.axvline(edge, color="0.86", linewidth=0.7, zorder=0)
        for edge in y_subdivision.edges():
            self.plot_ax.axhline(edge, color="0.86", linewidth=0.7, zorder=0)

        inside = [item for item in discrete_terms if item.point.inside_domain]
        outside = [item for item in discrete_terms if not item.point.inside_domain]

        if inside:
            xs = [item.point.coordinates[x_axis] for item in inside]
            ys = [item.point.coordinates[y_axis] for item in inside]
            colors = [item.point.cell_id for item in inside]
            self.plot_ax.plot(xs, ys, color="0.55", linewidth=0.7, alpha=0.35, zorder=1)
            self.plot_ax.scatter(xs, ys, c=colors, cmap="viridis", s=18, alpha=0.85, zorder=2)

        if outside:
            xs = [item.point.coordinates[x_axis] for item in outside]
            ys = [item.point.coordinates[y_axis] for item in outside]
            self.plot_ax.scatter(xs, ys, color="0.75", s=10, alpha=0.4, zorder=1)

        self._draw_attractor_markers(discrete_attractors, x_axis, y_axis)
        self.plot_ax.set_xlim(x_subdivision.range.minimum, x_subdivision.range.maximum)
        self.plot_ax.set_ylim(y_subdivision.range.minimum, y_subdivision.range.maximum)
        self.plot_ax.set_xlabel(x_axis)
        self.plot_ax.set_ylabel(y_axis)
        self.plot_ax.set_title("Trayectoria discretizada")
        self.plot_ax.set_aspect("equal", adjustable="box")

    def _draw_polar(self, grid, discrete_terms, discrete_attractors) -> None:
        r_subdivision = grid.subdivisions["r"]
        theta_subdivision = grid.subdivisions["theta"]

        self.plot_ax.set_ylim(r_subdivision.range.minimum, r_subdivision.range.maximum)
        self.plot_ax.set_theta_zero_location("E")
        self.plot_ax.set_theta_direction(1)
        self.plot_ax.grid(True, alpha=0.35)
        self.plot_ax.set_rticks(r_subdivision.edges()[1:])
        self.plot_ax.set_xticks(theta_subdivision.edges()[:-1])

        inside = [item for item in discrete_terms if item.point.inside_domain]
        outside = [item for item in discrete_terms if not item.point.inside_domain]

        if inside:
            theta = [item.point.coordinates["theta"] for item in inside]
            radius = [item.point.coordinates["r"] for item in inside]
            colors = [item.point.cell_id for item in inside]
            self.plot_ax.plot(theta, radius, color="0.55", linewidth=0.7, alpha=0.35, zorder=1)
            self.plot_ax.scatter(theta, radius, c=colors, cmap="viridis", s=18, alpha=0.85, zorder=2)

        if outside:
            theta = [item.point.coordinates["theta"] for item in outside]
            radius = [item.point.coordinates["r"] for item in outside]
            self.plot_ax.scatter(theta, radius, color="0.75", s=10, alpha=0.4, zorder=1)

        self._draw_attractor_markers(discrete_attractors, "theta", "r")
        self.plot_ax.set_title("Trayectoria polar discretizada")

    def _draw_one_dimensional(self, axis, grid, discrete_terms, discrete_attractors) -> None:
        subdivision = grid.subdivisions[axis]
        for edge in subdivision.edges():
            self.plot_ax.axhline(edge, color="0.86", linewidth=0.7, zorder=0)

        inside = [item for item in discrete_terms if item.point.inside_domain]
        outside = [item for item in discrete_terms if not item.point.inside_domain]

        if inside:
            ns = [item.term.index for item in inside]
            values = [item.point.coordinates[axis] for item in inside]
            colors = [item.point.cell_id for item in inside]
            self.plot_ax.plot(ns, values, color="0.55", linewidth=0.7, alpha=0.35, zorder=1)
            self.plot_ax.scatter(ns, values, c=colors, cmap="viridis", s=16, alpha=0.85, zorder=2)

        if outside:
            ns = [item.term.index for item in outside]
            values = [item.point.coordinates[axis] for item in outside]
            self.plot_ax.scatter(ns, values, color="0.75", s=10, alpha=0.4, zorder=1)

        attractor_values = []
        for discrete_attractor in discrete_attractors:
            for point in discrete_attractor.points:
                if point.inside_domain:
                    attractor_values.append(point.coordinates[axis])
        if attractor_values:
            start_x = max(0, len(discrete_terms) - max(20, len(attractor_values) * 4))
            xs = [start_x + index for index in range(len(attractor_values))]
            self.plot_ax.scatter(
                xs,
                attractor_values,
                marker="D",
                s=80,
                color="#d62728",
                edgecolor="black",
                linewidth=0.7,
                zorder=4,
            )

        self.plot_ax.set_ylim(subdivision.range.minimum, subdivision.range.maximum)
        self.plot_ax.set_xlabel("n")
        self.plot_ax.set_ylabel(axis)
        self.plot_ax.set_title(f"Proyeccion discreta sobre {axis}")

    def _draw_attractor_markers(self, discrete_attractors, x_axis, y_axis) -> None:
        xs = []
        ys = []
        for discrete_attractor in discrete_attractors:
            for point in discrete_attractor.points:
                if point.inside_domain:
                    xs.append(point.coordinates[x_axis])
                    ys.append(point.coordinates[y_axis])
        if xs:
            self.plot_ax.scatter(
                xs,
                ys,
                marker="D",
                s=90,
                color="#d62728",
                edgecolor="black",
                linewidth=0.7,
                zorder=4,
                label="atractor",
            )
            self.plot_ax.legend(loc="best")

    def _draw_parameter_selector(self) -> None:
        self.parameter_ax.clear()
        selected = self._selected_sequence()

        if selected == "logistica":
            self._draw_bifurcation_selector()
        elif selected == "z^2+c":
            self._draw_mandelbrot_selector()
        else:
            self.parameter_ax.axis("off")
            self.parameter_ax.text(
                0.5,
                0.5,
                "Sin mapa de parametros",
                ha="center",
                va="center",
                fontsize=10,
                color="0.35",
            )

    def _draw_bifurcation_selector(self) -> None:
        r_values, x_values = self._bifurcation_data()
        selected_r = self._float_value("r")
        selected_x = self._float_value("x0")

        self.parameter_ax.scatter(
            r_values,
            x_values,
            s=0.08,
            color="black",
            alpha=0.35,
            linewidths=0,
        )
        self.parameter_ax.axvline(selected_r, color="#d62728", linewidth=1.0)
        self.parameter_ax.axhline(selected_x, color="#d62728", linewidth=0.8, alpha=0.7)
        self.parameter_ax.scatter(
            [selected_r],
            [selected_x],
            marker="o",
            s=36,
            color="#d62728",
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
        )
        self.parameter_ax.set_xlim(0.0, 4.0)
        self.parameter_ax.set_ylim(0.0, 1.0)
        self.parameter_ax.set_title("Bifurcacion logistica", fontsize=10)
        self.parameter_ax.set_xlabel("r")
        self.parameter_ax.set_ylabel("x")

    def _draw_mandelbrot_selector(self) -> None:
        image, extent = self._mandelbrot_image()
        c_real = self._float_value("c_real")
        c_imag = self._float_value("c_imag")

        self.parameter_ax.imshow(
            image,
            extent=extent,
            origin="lower",
            cmap="magma",
            aspect="auto",
        )
        self.parameter_ax.scatter(
            [c_real],
            [c_imag],
            marker="o",
            s=42,
            color="#2ca02c",
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )
        self.parameter_ax.set_xlim(extent[0], extent[1])
        self.parameter_ax.set_ylim(extent[2], extent[3])
        self.parameter_ax.set_title("Mandelbrot", fontsize=10)
        self.parameter_ax.set_xlabel("Re(c)")
        self.parameter_ax.set_ylabel("Im(c)")

    def _bifurcation_data(self) -> tuple[np.ndarray, np.ndarray]:
        if self._bifurcation_cache is not None:
            return self._bifurcation_cache

        r_values = np.linspace(0.0, 4.0, 520)
        x_values = np.full_like(r_values, 0.5)
        plotted_r = []
        plotted_x = []

        for iteration in range(360):
            x_values = r_values * x_values * (1.0 - x_values)
            if iteration >= 240:
                plotted_r.append(r_values.copy())
                plotted_x.append(x_values.copy())

        self._bifurcation_cache = (
            np.concatenate(plotted_r),
            np.concatenate(plotted_x),
        )
        return self._bifurcation_cache

    def _mandelbrot_image(self) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        if self._mandelbrot_cache is not None:
            return self._mandelbrot_cache

        x_min, x_max = -2.0, 1.0
        y_min, y_max = -1.35, 1.35
        width, height = 360, 300
        iterations = 90

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

        extent = (x_min, x_max, y_min, y_max)
        self._mandelbrot_cache = (escape_counts, extent)
        return self._mandelbrot_cache

    def _on_parameter_click(self, event) -> None:
        if event.inaxes is not self.parameter_ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        selected = self._selected_sequence()
        if selected == "logistica":
            r_value = min(4.0, max(0.0, float(event.xdata)))
            x_value = min(1.0, max(0.0, float(event.ydata)))
            self.textboxes["r"].set_val(f"{r_value:.8g}")
            self.textboxes["x0"].set_val(f"{x_value:.8g}")
            self.redraw()
        elif selected == "z^2+c":
            real_value = min(1.0, max(-2.0, float(event.xdata)))
            imag_value = min(1.35, max(-1.35, float(event.ydata)))
            self.textboxes["c_real"].set_val(f"{real_value:.8g}")
            self.textboxes["c_imag"].set_val(f"{imag_value:.8g}")
            self.textboxes["z0_real"].set_val("0.0")
            self.textboxes["z0_imag"].set_val("0.0")
            self.redraw()

    def _draw_summary(self, result, discrete_terms, discrete_attractors) -> None:
        self.summary_ax.clear()
        self.summary_ax.axis("off")

        occupied = {
            item.point.cell_id
            for item in discrete_terms
            if item.point.inside_domain and item.point.cell_id is not None
        }
        outside_count = sum(1 for item in discrete_terms if not item.point.inside_domain)

        attractor_lines = []
        for index, discrete_attractor in enumerate(discrete_attractors, start=1):
            attractor = discrete_attractor.attractor
            cells = [
                point.cell_id
                for point in discrete_attractor.points
                if point.inside_domain and point.cell_id is not None
            ]
            attractor_lines.append(
                f"A{index}: {attractor.kind}, periodo={attractor.period}, celdas={cells}"
            )
        if not attractor_lines:
            attractor_lines.append("Atractores: no detectados")

        text = (
            f"{result.behavior} | {result.reason} | terminos={len(result.terms)} | "
            f"celdas visitadas={len(occupied)} | fuera={outside_count}\n"
            + "\n".join(attractor_lines[:3])
        )
        self.summary_ax.text(
            0.0,
            0.95,
            text,
            va="top",
            ha="left",
            fontsize=10,
            family="monospace",
        )

    def _draw_table(self, discrete_terms, discrete_attractors) -> None:
        self.visible_table_rows = max(1, self._int_value("table_rows"))
        table_mode = self._selected_table_mode()
        rows = []

        if table_mode in {"todo", "sucesion"}:
            for discrete_term in discrete_terms:
                point = discrete_term.point
                rows.append(
                    [
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
            for attractor_index, discrete_attractor in enumerate(discrete_attractors, start=1):
                for point_index, point in enumerate(discrete_attractor.points, start=1):
                    rows.append(
                        [
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
        if event.inaxes not in {self.table_ax, self.table_status_ax}:
            return
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
            self.table_ax.text(
                0.0,
                0.5,
                "Tabla discreta sin datos.",
                va="center",
                ha="left",
                fontsize=9,
            )
            return

        headers = ["tipo", "n", "valor", "coord", "idx", "centro", "celda"]
        table = self.table_ax.table(
            cellText=rows,
            colLabels=headers,
            loc="center",
            cellLoc="left",
            colLoc="left",
            colWidths=[0.07, 0.06, 0.18, 0.23, 0.13, 0.22, 0.08],
            bbox=[0.0, 0.0, 1.0, 1.0],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.0)

        for (row, _column), cell in table.get_celld().items():
            cell.set_edgecolor("0.82")
            cell.set_linewidth(0.5)
            if row == 0:
                cell.set_facecolor("0.92")
                cell.set_text_props(weight="bold")

    def _draw_table_status(self, start: int, visible_count: int) -> None:
        self.table_status_ax.clear()
        self.table_status_ax.axis("off")
        total = len(self.table_rows_cache)
        if total == 0:
            status = "Tabla sin datos"
        else:
            first = start + 1
            last = start + visible_count
            status = f"Filas {first}-{last} de {total}"
        self.table_status_ax.text(
            0.0,
            0.5,
            status,
            va="center",
            ha="left",
            fontsize=8,
            color="0.35",
        )

    def _draw_error(self, message: str) -> None:
        if self.plot_ax is not None:
            self.plot_ax.remove()
        self.plot_ax = self.fig.add_axes([0.34, 0.38, 0.37, 0.57])
        self.plot_ax.axis("off")
        self.plot_ax.text(0.5, 0.5, message, ha="center", va="center", color="#b00020")
        self.parameter_ax.clear()
        self.parameter_ax.axis("off")

        self.summary_ax.clear()
        self.summary_ax.axis("off")
        self.summary_ax.text(
            0.0,
            0.95,
            f"Error: {message}",
            va="top",
            ha="left",
            fontsize=10,
            color="#b00020",
        )
        self.table_ax.clear()
        self.table_ax.axis("off")
        self.table_status_ax.clear()
        self.table_status_ax.axis("off")

    def _format_mapping(self, mapping) -> str:
        if mapping is None:
            return "-"
        return ", ".join(
            f"{key}={self._format_optional(value)}"
            for key, value in mapping.items()
        )

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
        if self.sequence_selector is None:
            return SEQUENCES[0]
        return self.sequence_selector.value_selected

    def _selected_coordinates(self) -> str:
        if self.coordinate_selector is None:
            return COORDINATES[0]
        return self.coordinate_selector.value_selected

    def _selected_table_mode(self) -> str:
        if self.table_mode_selector is None:
            return TABLE_MODES[0]
        return self.table_mode_selector.value_selected

    def _float_value(self, key: str) -> float:
        return float(self.textboxes[key].text)

    def _int_value(self, key: str) -> int:
        return int(float(self.textboxes[key].text))

    def _axis_config(self, key: str) -> AxisConfig:
        parts = [part.strip() for part in self.textboxes[key].text.split(",")]
        if len(parts) != 3:
            raise ValueError(f"{key} must use the format min,max,steps")
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
