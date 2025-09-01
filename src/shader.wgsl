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
