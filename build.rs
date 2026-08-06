use std::{env, fs, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=assets/mandelbrot_logo.png");

    if !cfg!(target_os = "windows") {
        return;
    }

    let png_path = PathBuf::from("assets/mandelbrot_logo.png");
    let png = fs::read(&png_path).expect("no se pudo leer el logo de Mandelbrot");
    let ico_path = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR no definido"))
        .join("mandelbrot_logo.ico");

    // Un ICO puede contener directamente una imagen PNG. El logo ya es 64x64,
    // por lo que basta con envolver sus bytes con la cabecera ICO.
    let image_offset = 22u32;
    let image_length = u32::try_from(png.len()).expect("el logo es demasiado grande");
    let mut ico = Vec::with_capacity(image_offset as usize + png.len());
    ico.extend_from_slice(&0u16.to_le_bytes()); // reservado
    ico.extend_from_slice(&1u16.to_le_bytes()); // tipo icono
    ico.extend_from_slice(&1u16.to_le_bytes()); // número de imágenes
    ico.extend_from_slice(&[64, 64, 0, 0]); // anchura, altura, colores, reservado
    ico.extend_from_slice(&1u16.to_le_bytes()); // planos
    ico.extend_from_slice(&32u16.to_le_bytes()); // bits por píxel
    ico.extend_from_slice(&image_length.to_le_bytes());
    ico.extend_from_slice(&image_offset.to_le_bytes());
    ico.extend_from_slice(&png);
    fs::write(&ico_path, ico).expect("no se pudo generar el icono ICO");

    winresource::WindowsResource::new()
        .set_icon(ico_path.to_str().expect("ruta ICO no válida"))
        .compile()
        .expect("no se pudo incrustar el icono en el ejecutable");
}
