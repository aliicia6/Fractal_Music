use eframe::egui;
use fractal_music::ui::VisualExplorerApp;

fn main() -> anyhow::Result<()> {
    let icon = mandelbrot_icon();
    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        viewport: egui::ViewportBuilder::default()
            .with_decorations(false)
            .with_fullscreen(true)
            .with_icon(icon),
        ..Default::default()
    };
    eframe::run_native(
        "Fractal Music",
        options,
        Box::new(|cc| {
            let format = cc
                .wgpu_render_state
                .as_ref()
                .map(|state| state.target_format)
                .unwrap_or(wgpu::TextureFormat::Rgba8Unorm);
            Ok(Box::new(VisualExplorerApp::new(format)))
        }),
    )
    .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    Ok(())
}

fn mandelbrot_icon() -> egui::IconData {
    let size = 64u32;
    let mut rgba = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let c_re = -2.2 + x as f64 * 3.0 / (size - 1) as f64;
            let c_im = 1.35 - y as f64 * 2.7 / (size - 1) as f64;
            let mut z_re = 0.0;
            let mut z_im = 0.0;
            let mut escaped_at = 48;
            for iteration in 0..48 {
                let next_re = z_re * z_re - z_im * z_im + c_re;
                let next_im = 2.0 * z_re * z_im + c_im;
                z_re = next_re;
                z_im = next_im;
                if z_re * z_re + z_im * z_im > 4.0 {
                    escaped_at = iteration;
                    break;
                }
            }
            let color = if escaped_at == 48 {
                [235, 249, 255, 255]
            } else if escaped_at < 7 {
                [12, 30, 58, 255]
            } else {
                [65, 165, 230, 255]
            };
            rgba.extend_from_slice(&color);
        }
    }
    egui::IconData { rgba, width: size, height: size }
}
