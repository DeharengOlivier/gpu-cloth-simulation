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

