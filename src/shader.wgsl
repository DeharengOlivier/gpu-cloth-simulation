// Camera matrices (view and projection) supplied by the host each frame.
struct CameraUniform {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
};

// Per-particle state. This mirrors the layout written by the compute shader,
// but the render shader only reads the position.
struct Instance {
    position: vec3<f32>,
    speed: vec3<f32>,
};

// Bind group 0 holds the camera uniform (shared by both render pipelines).
// `instances` is declared for completeness but the simulated positions are
// fed through the instance vertex buffer (see InstanceInput below), so this
// storage binding is currently unused by the vertex stage.
@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(1) var<storage> instances: array<Instance>;





// Geometry of a single mesh vertex (shared by the cloth particles and the sphere).
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
};

// Per-instance data: the simulated position of one cloth particle, used to
// translate the shared mesh geometry to where that particle currently is.
struct InstanceInput {
    @location(3) pos: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.color = model.color;
    // Place the mesh at the particle's simulated position, then project to clip space.
    out.clip_position = camera.proj * camera.view * vec4<f32>(model.position + instance.pos, 1.0);
    return out;
}

// Cloth grid fragment shader: emit the flat per-vertex color unchanged.
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}



// Sphere rendering: unlike the cloth, the sphere carries a normal so the
// fragment stage can apply simple diffuse shading.
struct SphereVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

@vertex
fn sphere_vs_main(model: VertexInput) -> SphereVertexOutput {
    var out: SphereVertexOutput;
    out.color = model.color;
    // Transform the normal into view space for lighting (w = 0 drops translation).
    out.normal = (camera.view * vec4<f32>(model.normal, 0.0)).xyz;
    out.clip_position = camera.proj * camera.view * vec4<f32>(model.position, 1.0);
    return out;
