// Vertex shader: fullscreen triangle.
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(positions[vertex_index], 0.0, 1.0);
}

// Uniform buffer with the complex plane bounds, viewport size, and overlay points.
const MAX_OVERLAYS: u32 = 64u;
struct MandelbrotUniforms {
    c_min: vec2<f32>,
    c_max: vec2<f32>,
    // Frame origin (top-left) in pixels and frame size in pixels. Both vec2.
    frame_origin: vec2<f32>,
    frame_size: vec2<f32>,
    julia_c: vec2<f32>,
    julia_mode: u32,
    _julia_padding: u32,
    max_iterations: u32,
    overlay_count: u32,
    _padding: u32,
    // Use vec4 so the array stride is 16 bytes (alignment requirement for uniforms).
    overlays: array<vec4<f32>, 64>,
};

@group(0) @binding(0)
var<uniform> uniforms: MandelbrotUniforms;

// Convert a screen-space NDC position to complex-plane coordinates.
fn ndc_to_complex(ndc: vec2<f32>) -> vec2<f32> {
    let t = (ndc + vec2<f32>(1.0, 1.0)) * 0.5; // [0,1]
    let c = mix(uniforms.c_min, uniforms.c_max, t);
    return c;
}

// Fragment shader: compute the Mandelbrot set using the fragment's pixel position.
@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    // Convert fragment pixel position into normalized [0,1] coordinates within the frame
    // using the provided frame_origin (top-left) and frame_size. Then map to complex plane.
    let frag_px = vec2<f32>(frag_coord.x, frag_coord.y);
    // Offset by 0.5 to sample pixel centers and avoid half-pixel alignment issues.
    let rel = (frag_px - uniforms.frame_origin + vec2<f32>(0.5, 0.5)) / uniforms.frame_size; // [0,1]
    // Flip Y because frame origin is top-left while complex coords typically have +Y up
    let t = vec2<f32>(rel.x, 1.0 - rel.y);
    var c = mix(uniforms.c_min, uniforms.c_max, t);

    var z = vec2<f32>(0.0, 0.0);
    if uniforms.julia_mode == 1u {
        z = c;
        c = uniforms.julia_c;
    }
    var iterations: u32 = 0u;
    var escaped = false;

    loop {
        if iterations >= uniforms.max_iterations {
            break;
        }
        // z = z^2 + c
        let zr2 = z.x * z.x - z.y * z.y;
        let zc2 = 2.0 * z.x * z.y;
        z = vec2<f32>(zr2, zc2) + c;

        if dot(z, z) > 4.0 {
            escaped = true;
            break;
        }
        iterations = iterations + 1u;
    }

    // Compute the base color (black if inside, otherwise palette)
    var base_color = vec3<f32>(0.0, 0.0, 0.0);
    if escaped {
        let t = f32(iterations) / f32(uniforms.max_iterations);
        base_color = 0.5 + 0.5 * cos(3.0 + t * 15.0 + vec3<f32>(0.0, 0.6, 1.0));
    }

    // Overlay drawing: check each overlay point and, if close enough in pixel space,
    // blend a highlight color on top.
    var out_color = base_color;
    let overlay_color = vec3<f32>(1.0, 1.0, 1.0);
    let overlay_radius_px = 6.0;

    // Map overlay complex coords to pixel coords and check distance.
    for (var i: u32 = 0u; i < uniforms.overlay_count; i = i + 1u) {
        let ov4 = uniforms.overlays[i];
        let ov = ov4.xy;
        // compute normalized t for overlay within the complex bounds
        let tt = (ov - uniforms.c_min) / (uniforms.c_max - uniforms.c_min);
        // Map to pixel coords inside the frame using frame_origin and frame_size.
        let ov_px = uniforms.frame_origin + vec2<f32>(tt.x * uniforms.frame_size.x, (1.0 - tt.y) * uniforms.frame_size.y);
        let d = distance(frag_px, ov_px);
        if d < overlay_radius_px {
            // simple blend: overlay_color over base
            let alpha = 1.0 - (d / overlay_radius_px);
            out_color = mix(out_color, overlay_color, alpha);
        }
    }

    return vec4<f32>(out_color, 1.0);
}

