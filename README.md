# Fractal Music

Aplicación de investigación matemática para explorar sucesiones reales y complejas:
la **logística**, el **mapa cuadrático complejo** `z²+c` y el **punto fijo de cos(x)**.
Incluye análisis de atractores, discretización en coordenadas, visualización de
trayectorias, diagrama de bifurcación, conjunto de Mandelbrot y generación de
patrones musicales (Strudel).

El proyecto se desarrolla en **dos implementaciones paralelas** con la misma
funcionalidad:

| Implementación | Lenguaje | UI |
|---|---|---|
| `fractal_sequences/` + `explorador_visual.py` | Python | Matplotlib |
| `src/` (crate `fractal_music`) | Rust | eframe / egui |

Ambas comparten el mismo formato de sesión JSON/CSV y la misma lógica matemática.

---

## Concepto

La configuración reproducible del caso principal:

```python
fractal_sequences.default_concept_config()
```

El pipeline completo se ejecuta con:

```python
fractal_sequences.run_exploration(config)
```

El caso principal define una recurrencia compleja `z[n+1] = z[n]² + c`, calcula
un número configurable de términos, detecta punto fijo / ciclo / divergencia,
proyecta los valores a coordenadas polares, discretiza el dominio `(r, θ)`,
visualiza la trayectoria y atractores, y permite guardar y comparar casos.

---

## Versión Python (Matplotlib)

### Requisitos

- Python 3.10+
- Dependencias: `numpy`, `matplotlib`

### Instalación

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install numpy matplotlib
```

### Ejecución

```powershell
python .\explorador_visual.py
```

### Uso

- En el diagrama de bifurcación y el conjunto de Mandelbrot se selecciona con
  clic izquierdo.
- La rueda del ratón sobre el mapa amplía/reduce la vista.
- Un punto logístico `(r, x0)` se vincula automáticamente con la recurrencia
  compleja equivalente `c = r(2-r)/4`, `z0 = r(1/2-x0)`.
- `Guardar caso` conserva el ejemplo actual; los casos se dibujan a la vez.
- `Exportar JSON`/`CSV` e `Importar JSON` gestionan sesiones completas.
- El menú `Strudel` transforma los atractores discretizados en comandos musicales.

---

## Versión Rust (eframe/egui)

### Requisitos

- **Rust** (edition 2024). Se recomienda instalar con [rustup](https://rustup.rs).
- **Windows**: requiere el **MSVC build tools** (o el toolchain GNU) y el
  **Rust MSVC toolchain** instalado por rustup. Para visualización también se
  necesita el kit de desarrollo de Windows (Winget/SDK) que instala VS Build Tools.
- **macOS**: requiere Xcode Command Line Tools (`xcode-select --install`).
- **Linux**: requiere `libxcb`, `libxkbcommon`, `libgtk-3` (proxy de `rfd`).

### Compilación

```bash
cargo build
```

### Ejecución

```bash
cargo run
```

La ventana se abre con el título **"Fractal Music"** (~1600×980).

### Tests

```bash
cargo test
```

### Distribución portable Windows x64 (offline)

El archivo `.cargo/config.toml` activa el runtime CRT estático para
`x86_64-pc-windows-msvc`. Para crear la distribución sin instalador Tauri:

1. Instala el toolchain MSVC de 64 bits y coloca el runtime fijo completo de
   WebView2 en una carpeta local. Esa carpeta debe contener
   `msedgewebview2.exe`.
2. Ejecuta PowerShell desde el proyecto:

   ```powershell
   .\scripts\package-windows-portable.ps1 -WebView2Runtime C:\ruta\webview2
   ```

El script ejecuta `cargo build --release` y genera
`dist\FractalMusic-Windows-x64.zip`, con la carpeta `Fractal Music` completa.
El runtime WebView2 no se versiona en Git; adjúntalo al Release de GitHub o
proporciónalo al ejecutar el script. Strudel se incluye en el ejecutable
principal mediante `include_bytes!`, se extrae automáticamente a `%TEMP%` y se
abre usando el runtime WebView2 situado junto a `fractal_music.exe`.

> Nota: el crate usa `rfd` (diálogos de archivo) y `arboard` (portapapeles),
> ambos multiplataforma. En Windows los diálogos nativos y el portapapeles
> funcionan sin configuración adicional.

---

## Portabilidad macOS → Windows

El código Rust está diseñado para ser multiplataforma. Puntos clave:

- **UI**: `eframe`/`egui` (OpenGL/Vulkan/WebGPU) funcionan en Windows, macOS y Linux.
- **Diálogos de archivo**: `rfd` usa los diálogos nativos de cada sistema.
- **Portapapeles**: `arboard` abstrae el portapapeles de cada plataforma.
- **Rutas**: el proyecto no codifica rutas absolutas; usa las proporcionadas por
  los diálogos de archivo.
- **Números**: `num-complex` proporciona aritmética compleja portable.

No se requiere ninguna biblioteca específica de macOS en el código Rust.

---

## Estructura del proyecto

```
FractalMusic/
├── Cargo.toml            # Dependencias del crate Rust
├── src/                  # Implementación Rust
│   ├── main.rs           # Entrada de la app eframe
│   ├── lib.rs            # Reexportación del framework
│   ├── ui.rs             # App visual (eframe/egui)
│   ├── analyzer.rs       # Análisis de atractores
│   ├── coordinates.rs    # Sistemas de coordenadas
│   ├── discretization.rs # Discretización uniforme
│   ├── transformations.rs# Proyección discreta
│   ├── music.rs          # Generación de patrones Strudel
│   ├── explorer_session.rs# Sesiones JSON/CSV
│   ├── config.rs         # Configuración reproducible
│   └── ...
├── fractal_sequences/    # Framework Python (espejo)
├── explorador_visual.py  # App visual Python (Matplotlib)
├── tests/                # Tests de integración Rust
└── CONCEPTO.md           # Documento de concepto
```

---

## Licencia

Ver el archivo `LICENSE`.
