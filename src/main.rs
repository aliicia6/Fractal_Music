use visualizador_mandelbrot::ui::VisualExplorerApp;

fn main() -> anyhow::Result<()> {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default().with_inner_size([1600.0, 980.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Explorador de sucesiones",
        options,
        Box::new(|_cc| Ok(Box::new(VisualExplorerApp::default()))),
    )
    .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    Ok(())
}
