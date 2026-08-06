//! GPU renderer for the Mandelbrot set using egui-wgpu's `CallbackTrait`.
//!
//! This draws the Mandelbrot set as a fullscreen triangle with a fragment shader
//! that iterates `z^2 + c` per pixel. The complex-plane bounds (`c_min`, `c_max`)
//! are passed as uniforms so the set stays perfectly aligned with the plot's
//! visible region, and the whole thing is redrawn whenever the user zooms/pans.

use eframe::egui;
use egui_wgpu::{CallbackResources, CallbackTrait, ScreenDescriptor};
use wgpu::util::DeviceExt;

/// The WGSL shader source, embedded at compile time.
const SHADER_SRC: &str = include_str!("mandelbrot.wgsl");

/// Uniform data passed to the shader.
const MAX_OVERLAYS: usize = 64;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MandelbrotUniforms {
    c_min: [f32; 2],
    c_max: [f32; 2],
    frame_origin: [f32; 2],
    frame_size: [f32; 2],
    julia_c: [f32; 2],
    julia_mode: u32,
    _julia_padding: u32,
    max_iterations: u32,
    overlay_count: u32,
    _padding: u32,
    // Use 4-element arrays to ensure 16-byte stride/alignment for uniform arrays.
    overlays: [[f32; 4]; MAX_OVERLAYS],
    // Tail padding to match WGSL alignment/padding (ensures Rust struct size matches shader expectation).
    _tail_padding: [u32; 3],
}

impl MandelbrotUniforms {
    fn new(c_min: [f32; 2], c_max: [f32; 2], frame_origin: [f32; 2], frame_size: [f32; 2]) -> Self {
        Self {
            c_min,
            c_max,
            frame_origin,
            frame_size,
            julia_c: [0.0; 2],
            julia_mode: 0,
            _julia_padding: 0,
            max_iterations: 256,
            overlay_count: 0,
            _padding: 0,
            overlays: [[0.0; 4]; MAX_OVERLAYS],
            _tail_padding: [0; 3],
        }
    }
}

/// A single Mandelbrot callback instance. It is recreated (or updated) each frame
/// with the current plot bounds so the GPU texture matches the visible region.
pub struct MandelbrotCallback {
    c_min: [f32; 2],
    c_max: [f32; 2],
    overlays: Vec<[f32; 2]>,
    // Store the paint callback frame rect (in points) so prepare() can compute
    // the frame pixel origin/size using the ScreenDescriptor.
    frame_rect: egui::Rect,
    target_format: wgpu::TextureFormat,
    julia_c: Option<[f32; 2]>,
}

impl MandelbrotCallback {
    /// Create a new callback for the given complex-plane bounds, overlays, frame rect and target format.
    pub fn new(c_min: [f32; 2], c_max: [f32; 2], overlays: Vec<[f32; 2]>, frame_rect: egui::Rect, target_format: wgpu::TextureFormat) -> Self {
        Self {
            c_min,
            c_max,
            overlays,
            frame_rect,
            target_format,
            julia_c: None,
        }
    }

    /// Creates a callback that renders the Julia set for a fixed complex c.
    pub fn new_julia(
        z_min: [f32; 2],
        z_max: [f32; 2],
        julia_c: [f32; 2],
        frame_rect: egui::Rect,
        target_format: wgpu::TextureFormat,
    ) -> Self {
        Self {
            c_min: z_min,
            c_max: z_max,
            overlays: Vec::new(),
            frame_rect,
            target_format,
            julia_c: Some(julia_c),
        }
    }
}

impl CallbackTrait for MandelbrotCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen_descriptor: &ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let resources = callback_resources
            .entry::<MandelbrotResources>()
            .or_insert_with(|| {
                MandelbrotResources::new(device, self.target_format, screen_descriptor)
            });

        // Compute frame origin and size in pixels from stored frame_rect (in points)
        let pixels_per_point = screen_descriptor.pixels_per_point;
        let frame_min = self.frame_rect.min;
        let frame_size_points = self.frame_rect.size();
        let mut frame_origin_px = [frame_min.x * pixels_per_point, frame_min.y * pixels_per_point];
        let mut frame_size_px = [frame_size_points.x * pixels_per_point, frame_size_points.y * pixels_per_point];

        // Round to integer pixels to avoid subtle subpixel offsets and clamp to screen
        let screen_w = screen_descriptor.size_in_pixels[0] as f32;
        let screen_h = screen_descriptor.size_in_pixels[1] as f32;
        frame_origin_px = [frame_origin_px[0].round(), frame_origin_px[1].round()];
        frame_size_px = [frame_size_px[0].round(), frame_size_px[1].round()];
        frame_origin_px[0] = frame_origin_px[0].clamp(0.0, screen_w);
        frame_origin_px[1] = frame_origin_px[1].clamp(0.0, screen_h);
        frame_size_px[0] = frame_size_px[0].max(1.0).min(screen_w - frame_origin_px[0]);
        frame_size_px[1] = frame_size_px[1].max(1.0).min(screen_h - frame_origin_px[1]);

        // Diagnostic prints removed (alignment confirmed). Keep this comment if future debug is needed.

        let mut uniforms = MandelbrotUniforms::new(self.c_min, self.c_max, frame_origin_px, frame_size_px);
        if let Some(julia_c) = self.julia_c {
            uniforms.julia_c = julia_c;
            uniforms.julia_mode = 1;
        }
        // Fill overlays (limit to MAX_OVERLAYS)
        let count = self.overlays.len().min(MAX_OVERLAYS);
        uniforms.overlay_count = count as u32;
        for i in 0..count {
            let ov = self.overlays[i];
            uniforms.overlays[i] = [ov[0], ov[1], 0.0_f32, 0.0_f32];
        }
        queue.write_buffer(
            &resources.uniform_buffer,
            0,
            bytemuck::bytes_of(&uniforms),
        );
        Vec::new()
    }

fn paint(
        &self,
        _info: epaint::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &CallbackResources,
    ) {
        let Some(resources) = callback_resources.get::<MandelbrotResources>() else {
            return;
        };

        render_pass.set_pipeline(&resources.pipeline);
        render_pass.set_bind_group(0, &resources.bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }
}

/// GPU resources shared across frames for the Mandelbrot renderer.
struct MandelbrotResources {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    _bind_group_layout: wgpu::BindGroupLayout,
    _pipeline_layout: wgpu::PipelineLayout,
}

impl MandelbrotResources {
    fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        screen_descriptor: &ScreenDescriptor,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mandelbrot_shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(SHADER_SRC)),
        });

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mandelbrot_uniforms"),
            contents: bytemuck::bytes_of(&MandelbrotUniforms::new(
                [-2.0, -1.35],
                [1.0, 1.35],
                [0.0_f32, 0.0_f32],
                [
                    screen_descriptor.size_in_pixels[0] as f32,
                    screen_descriptor.size_in_pixels[1] as f32,
                ],
            )),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("mandelbrot_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        has_dynamic_offset: false,
                        min_binding_size: None,
                        ty: wgpu::BufferBindingType::Uniform,
                    },
                    count: None,
                }],
            });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mandelbrot_bind_group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniform_buffer,
                    offset: 0,
                    size: None,
                }),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mandelbrot_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mandelbrot_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            uniform_buffer,
            bind_group,
            _bind_group_layout: bind_group_layout,
            _pipeline_layout: pipeline_layout,
        }
    }
}
