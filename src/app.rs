//! The interactive program: window, camera, egui panel and the render pipelines.
//!
//! Everything physical lives in [`crate::simulation`]. This module draws what
//! that simulation produced and lets a user retune it; it never computes a force.

use wgpu_bootstrap::{
    cgmath::{self, InnerSpace},
    egui,
    util::{
        geometry::icosphere,
        orbit_camera::{CameraUniform, OrbitCamera},
    },
    wgpu::{self, util::DeviceExt},
    App, Context,
};

use crate::timestep::FixedTimestep;

use crate::simulation::{ClothConfig, ClothSimulation, Instance, FIXED_TIME_STEP_SECONDS};

/// One mesh vertex.
///
/// `#[repr(C)]` gives a stable, C-compatible layout so the CPU and GPU agree on
/// field placement; bytemuck allows reinterpreting the struct as raw bytes.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    /// Position (x, y, z) in model space.
    position: [f32; 3],
    /// Normal, used for lighting the obstacle sphere.
    normal: [f32; 3],
    /// RGB colour, each component in 0.0..=1.0.
    color: [f32; 3],
}

impl Vertex {
    /// How the GPU reads a `Vertex` out of the vertex buffer.
    ///
    /// Offsets come from `offset_of!` rather than being written by hand, so the
    /// declaration cannot drift away from the struct it describes.
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: std::mem::offset_of!(Vertex, position) as wgpu::BufferAddress,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::offset_of!(Vertex, normal) as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::offset_of!(Vertex, color) as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

/// How the GPU reads a simulated [`Instance`] as per-instance vertex data.
///
/// Unlike [`Vertex::desc`] the step mode is `Instance`: this data advances once
/// per drawn instance rather than once per vertex. Offsets come from
/// `offset_of!` on the simulation's own struct, so the renderer reads the
/// fields the physics actually wrote.
fn instance_vertex_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Instance>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &[
            wgpu::VertexAttribute {
                offset: std::mem::offset_of!(Instance, position) as wgpu::BufferAddress,
                shader_location: 3,
                format: wgpu::VertexFormat::Float32x3, // The w padding is ignored.
            },
            wgpu::VertexAttribute {
                offset: std::mem::offset_of!(Instance, speed) as wgpu::BufferAddress,
                shader_location: 4,
                format: wgpu::VertexFormat::Float32x3,
            },
        ],
    }
}

/// User-tunable configuration exposed through the egui panel.
///
/// Grid size, spacing and point size require rebuilding the GPU buffers;
/// colours are uploaded in place.
#[derive(Clone)]
pub struct ClothSettings {
    /// N for the N x N grid (particles per side).
    pub grid_size: u32,
    /// Distance between adjacent particles.
    pub spacing: f32,
    /// Render radius of the small sphere drawn at each particle.
    pub point_size: f32,
    /// Cloth RGB colour.
    pub cloth_color: [f32; 3],
    /// Obstacle sphere RGB colour.
    pub sphere_color: [f32; 3],
}

impl Default for ClothSettings {
    fn default() -> Self {
        Self {
            grid_size: 256,               // 256 x 256 grid = 65,536 particles
            spacing: 0.006,               // 6 mm between particles
            point_size: 0.0033,           // Visualisation sphere radius
            cloth_color: [1.0, 0.0, 0.0], // Red
            sphere_color: [0.5, 0.5, 0.5],
        }
    }
}

impl ClothSettings {
    /// The simulation configuration these settings describe.
    fn to_config(&self) -> ClothConfig {
        ClothConfig {
            grid_size: self.grid_size,
            spacing: self.spacing,
            ..ClothConfig::default()
        }
    }
}

/// The interactive application: a [`ClothSimulation`] plus everything needed to
/// look at it and retune it.
pub struct InstanceApp {
    simulation: ClothSimulation,

    // Rendering
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    render_pipeline: wgpu::RenderPipeline,
    num_indices: u32,

    // Obstacle sphere
    sphere_vertex_buffer: wgpu::Buffer,
    sphere_index_buffer: wgpu::Buffer,
    sphere_render_pipeline: wgpu::RenderPipeline,
    num_sphere_indices: u32,

    camera: OrbitCamera,

    // UI state
    settings: ClothSettings,
    pending_settings: ClothSettings,
    paused: bool,

    /// Carries the fraction of a step left over by each frame into the next.
    clock: FixedTimestep,
    /// Steps run by the last frame, shown in the panel.
    steps_last_frame: u32,
}

/// Builds the small sphere drawn once per cloth particle.
///
/// Subdivision level 2 keeps the per-particle triangle count low, which matters
/// because this mesh is drawn tens of thousands of times per frame.
/// Complexity: O(1) in the particle count.
fn create_particle_mesh(sphere_scale: f32, cloth_color: [f32; 3]) -> (Vec<Vertex>, Vec<u32>) {
    let (positions, indices) = icosphere(2);
    let vertices = positions
        .iter()
        .map(|position| Vertex {
            position: (*position * sphere_scale).into(),
            normal: [0.0, 0.0, 0.0], // The cloth is drawn unlit.
            color: cloth_color,
        })
        .collect();
    (vertices, indices)
}

/// Builds the static obstacle sphere.
///
/// Subdivision level 3: more detailed than the particle mesh, and drawn once.
/// Complexity: O(1).
fn create_sphere_vertices(sphere_radius: f32, sphere_color: [f32; 3]) -> (Vec<Vertex>, Vec<u32>) {
    let (positions, indices) = icosphere(3);
    let vertices = positions
        .iter()
        .map(|position| {
            // On a unit sphere the outward normal is the normalised position.
            let normal = position.normalize();
            Vertex {
                position: (normal * sphere_radius).into(),
                normal: normal.into(),
                color: sphere_color,
            }
        })
        .collect();
    (vertices, indices)
}

/// The render-pipeline settings both pipelines share: filled back-face-culled
/// triangles, depth tested, no multisampling.
fn primitive_state() -> wgpu::PrimitiveState {
    wgpu::PrimitiveState {
        topology: wgpu::PrimitiveTopology::TriangleList,
        strip_index_format: None,
        front_face: wgpu::FrontFace::Ccw,
        cull_mode: Some(wgpu::Face::Back),
        polygon_mode: wgpu::PolygonMode::Fill,
        unclipped_depth: false,
        conservative: false,
    }
}

fn depth_state(context: &Context) -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: context.depth_stencil_format(),
        depth_write_enabled: true,
        depth_compare: wgpu::CompareFunction::Less,
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    }
}

fn no_multisampling() -> wgpu::MultisampleState {
    wgpu::MultisampleState {
        count: 1,
        mask: !0,
        alpha_to_coverage_enabled: false,
    }
}

impl InstanceApp {
    /// Builds the application with the default settings.
    pub fn new(context: &Context) -> Self {
        Self::create_with_settings(context, ClothSettings::default())
    }

    /// Builds every GPU resource: the simulation, both meshes, both pipelines
    /// and the camera.
    fn create_with_settings(context: &Context, settings: ClothSettings) -> Self {
        let device = context.device();
        let simulation = ClothSimulation::new(device, &settings.to_config());

        let (vertices, indices) = create_particle_mesh(settings.point_size, settings.cloth_color);
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(vertices.as_slice()),
            // COPY_DST so a colour change can be uploaded in place.
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(indices.as_slice()),
            usage: wgpu::BufferUsages::INDEX,
        });

        let sphere_radius = ClothConfig::default().sphere_radius;
        let (sphere_vertices, sphere_indices) =
            create_sphere_vertices(sphere_radius, settings.sphere_color);
        let sphere_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sphere Vertex Buffer"),
            contents: bytemuck::cast_slice(sphere_vertices.as_slice()),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let sphere_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sphere Index Buffer"),
            contents: bytemuck::cast_slice(sphere_indices.as_slice()),
            usage: wgpu::BufferUsages::INDEX,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Render Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let camera_bind_group_layout = device.create_bind_group_layout(&CameraUniform::desc());
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&camera_bind_group_layout],
            push_constant_ranges: &[],
        });

        // The cloth: one instanced draw of the particle mesh, positioned from
        // the simulation's own buffer.
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc(), instance_vertex_layout()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: context.format(),
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: primitive_state(),
            depth_stencil: Some(depth_state(context)),
            multisample: no_multisampling(),
            multiview: None,
            cache: None,
        });

        // The obstacle sphere: drawn once, lit, so it needs its own entry points
        // and no instance buffer.
        let sphere_render_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Sphere Render Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "sphere_vs_main",
                    buffers: &[Vertex::desc()],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "sphere_fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: context.format(),
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: primitive_state(),
                depth_stencil: Some(depth_state(context)),
                multisample: no_multisampling(),
                multiview: None,
                cache: None,
            });

        let aspect = context.size().x / context.size().y;
        let mut camera = OrbitCamera::new(context, 45.0, aspect, 0.1, 100.0);
        camera
            .set_polar(cgmath::point3(1.5, 0.0, 0.0))
            .update(context);

        Self {
            simulation,
            vertex_buffer,
            index_buffer,
            render_pipeline,
            num_indices: indices.len() as u32,
            sphere_vertex_buffer,
            sphere_index_buffer,
            sphere_render_pipeline,
            num_sphere_indices: sphere_indices.len() as u32,
            camera,
            settings: settings.clone(),
            pending_settings: settings,
            paused: false,
            clock: FixedTimestep::default(),
            steps_last_frame: 0,
        }
    }

    /// Restarts the simulation with the pending settings.
    ///
    /// Only the resources that depend on those settings are rebuilt: the
    /// pipelines and the shader modules do not, so they are kept.
    fn rebuild(&mut self, context: &Context) {
        let device = context.device();
        self.settings = self.pending_settings.clone();
        self.simulation = ClothSimulation::new(device, &self.settings.to_config());

        let (vertices, indices) =
            create_particle_mesh(self.settings.point_size, self.settings.cloth_color);
        self.vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(vertices.as_slice()),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        self.index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(indices.as_slice()),
            usage: wgpu::BufferUsages::INDEX,
        });
        self.num_indices = indices.len() as u32;
    }

    /// Uploads new colours without rebuilding anything.
    ///
    /// Only the two meshes are regenerated, never the particle grid: the colour
    /// picker fires on every frame of a drag, and rebuilding 65,536 particles at
    /// that rate is work nobody asked for.
    fn update_colors(&mut self, context: &Context) {
        let (vertices, _) =
            create_particle_mesh(self.settings.point_size, self.pending_settings.cloth_color);
        context.queue().write_buffer(
            &self.vertex_buffer,
            0,
            bytemuck::cast_slice(vertices.as_slice()),
        );

        let sphere_radius = ClothConfig::default().sphere_radius;
        let (sphere_vertices, _) =
            create_sphere_vertices(sphere_radius, self.pending_settings.sphere_color);
        context.queue().write_buffer(
            &self.sphere_vertex_buffer,
            0,
            bytemuck::cast_slice(sphere_vertices.as_slice()),
        );

        self.settings.cloth_color = self.pending_settings.cloth_color;
        self.settings.sphere_color = self.pending_settings.sphere_color;
    }
}

impl App for InstanceApp {
    fn input(&mut self, input: egui::InputState, context: &Context) {
        self.camera.input(input, context);
    }

    fn render_gui(&mut self, egui_ctx: &egui::Context, context: &Context) {
        egui::Window::new("Cloth Settings").show(egui_ctx, |ui| {
            if ui
                .button(if self.paused { "Resume" } else { "Pause" })
                .clicked()
            {
                self.paused = !self.paused;
            }
            ui.separator();

            ui.label("Cloth color:");
            let mut cloth_color = self.pending_settings.cloth_color;
            if ui.color_edit_button_rgb(&mut cloth_color).changed() {
                self.pending_settings.cloth_color = cloth_color;
                self.update_colors(context);
            }

            ui.label("Sphere color:");
            let mut sphere_color = self.pending_settings.sphere_color;
            if ui.color_edit_button_rgb(&mut sphere_color).changed() {
                self.pending_settings.sphere_color = sphere_color;
                self.update_colors(context);
            }

            ui.separator();
            ui.label("Settings (restart required):");

            ui.horizontal(|ui| {
                ui.label("Grid size:");
                let mut grid_val = self.pending_settings.grid_size as i32;
                if ui
                    .add(egui::Slider::new(&mut grid_val, 64..=512).step_by(64.0))
                    .changed()
                {
                    self.pending_settings.grid_size = grid_val as u32;
                }
            });
            let side = self.pending_settings.grid_size;
            ui.label(format!("  -> {} particles", side * side));

            ui.horizontal(|ui| {
                ui.label("Spacing:");
                ui.add(
                    egui::Slider::new(&mut self.pending_settings.spacing, 0.002..=0.02)
                        .step_by(0.001),
                );
            });

            ui.horizontal(|ui| {
                ui.label("Point size:");
                ui.add(
                    egui::Slider::new(&mut self.pending_settings.point_size, 0.001..=0.01)
                        .step_by(0.0005),
                );
            });

            ui.separator();

            let settings_changed = self.pending_settings.grid_size != self.settings.grid_size
                || self.pending_settings.spacing != self.settings.spacing
                || self.pending_settings.point_size != self.settings.point_size;

            if settings_changed {
                ui.colored_label(egui::Color32::YELLOW, "Pending changes");
                if ui.button("Apply and Restart").clicked() {
                    self.rebuild(context);
                }
            }

            ui.separator();
            ui.label(format!("Particles: {}", self.simulation.particle_count()));
            ui.label(format!(
                "Steps last frame: {} of the {:.1} real time asks for",
                self.steps_last_frame,
                1.0 / (FIXED_TIME_STEP_SECONDS * 60.0)
            ));

            // Dropping simulated time is the right response to a hitch, but on
            // its own it is invisible: the cloth just runs slower than the world.
            if self.clock.has_fallen_behind() {
                ui.colored_label(
                    egui::Color32::YELLOW,
                    format!(
                        "Behind real time by {:.1} s: this machine cannot keep up \
                         with {} particles",
                        self.clock.dropped_seconds(),
                        self.simulation.particle_count()
                    ),
                );
            }
        });
    }

    fn update(&mut self, delta_time: f32, context: &Context) {
        if self.paused {
            self.steps_last_frame = 0;
            return;
        }

        // Run as many fixed steps as the elapsed frame time paid for, so the
        // simulation advances at the same rate whatever the frame rate. The
        // clock keeps the remainder and caps the catch-up; see `timestep`.
        let steps = self.clock.steps_for(delta_time, FIXED_TIME_STEP_SECONDS);
        self.steps_last_frame = steps;
        for _ in 0..steps {
            self.simulation.step(context.device(), context.queue());
        }
    }

    fn render(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        render_pass.set_bind_group(0, self.camera.bind_group(), &[]);

        // The cloth: one instanced draw straight from the simulation's buffer.
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.simulation.current_buffer().slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..self.num_indices, 0, 0..self.simulation.particle_count());

        // The obstacle sphere.
        render_pass.set_pipeline(&self.sphere_render_pipeline);
        render_pass.set_vertex_buffer(0, self.sphere_vertex_buffer.slice(..));
        render_pass.set_index_buffer(
            self.sphere_index_buffer.slice(..),
            wgpu::IndexFormat::Uint32,
        );
        render_pass.draw_indexed(0..self.num_sphere_indices, 0, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, offset_of, size_of};

    #[test]
    fn vertex_layout_matches_the_shader() {
        // shader.wgsl reads position/normal/colour as three vec3<f32> at
        // offsets 0, 12 and 24.
        assert_eq!(size_of::<Vertex>(), 36, "Vertex is 9 f32 = 36 bytes");
        assert_eq!(align_of::<Vertex>(), 4);
        assert_eq!(offset_of!(Vertex, position), 0);
        assert_eq!(offset_of!(Vertex, normal), 12);
        assert_eq!(offset_of!(Vertex, color), 24);
    }

    #[test]
    fn the_vertex_layout_describes_the_struct_it_reads() {
        let desc = Vertex::desc();
        assert_eq!(desc.array_stride, size_of::<Vertex>() as u64);
        assert_eq!(
            desc.attributes[0].offset,
            offset_of!(Vertex, position) as u64
        );
        assert_eq!(desc.attributes[1].offset, offset_of!(Vertex, normal) as u64);
        assert_eq!(desc.attributes[2].offset, offset_of!(Vertex, color) as u64);
    }

    #[test]
    fn the_instance_layout_describes_the_struct_the_physics_wrote() {
        // This used to declare the velocity attribute at offset 12, where the
        // field starts at 16, so @location(4) would have read the last position
        // float and the first two velocity floats. It was harmless only because
        // the render shader declares no location 4. The offsets now come from
        // offset_of! and cannot drift.
        let desc = instance_vertex_layout();
        assert_eq!(desc.array_stride, size_of::<Instance>() as u64);
        assert_eq!(
            desc.attributes[0].offset,
            offset_of!(Instance, position) as u64
        );
        assert_eq!(
            desc.attributes[1].offset,
            offset_of!(Instance, speed) as u64
        );
        assert_eq!(desc.attributes[1].offset, 16);
    }

    #[test]
    fn the_render_shader_reads_only_locations_the_layout_supplies() {
        let shader = include_str!("shader.wgsl");
        for location in [
            "@location(0)",
            "@location(1)",
            "@location(2)",
            "@location(3)",
        ] {
            assert!(shader.contains(location), "shader.wgsl must use {location}");
        }
    }

    #[test]
    fn the_render_shader_binds_nothing_the_pipeline_layout_does_not_provide() {
        // repaired: shader.wgsl declared `@group(1) @binding(1) instances`, an
        // array of a two-vec3 Instance struct that did not match the vec4 layout
        // the compute shader writes. The render pipeline layout lists one bind
        // group, the camera, so nothing could ever have been bound to group 1.
        // It survived because the binding was unused and got pruned.
        let shader = include_str!("shader.wgsl");
        let groups: Vec<&str> = shader
            .match_indices("@group(")
            .map(|(at, _)| {
                let rest = &shader[at + "@group(".len()..];
                &rest[..rest.find(')').expect("unterminated @group")]
            })
            .collect();
        assert!(!groups.is_empty(), "shader.wgsl must bind the camera");
        assert!(
            groups.iter().all(|group| *group == "0"),
            "the render pipeline layout provides one bind group, but the shader \
             declares groups {groups:?}"
        );
    }

    #[test]
    fn the_default_settings_describe_the_default_simulation() {
        let settings = ClothSettings::default();
        let config = settings.to_config();
        assert_eq!(config.grid_size, settings.grid_size);
        assert_eq!(config.spacing, settings.spacing);
        assert_eq!(config.sphere_radius, ClothConfig::default().sphere_radius);
    }
}
