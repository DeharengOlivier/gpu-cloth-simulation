// Camera matrices (view and projection) supplied by the host each frame.
struct CameraUniform {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
};

// Per-particle state. This mirrors the layout written by the compute shader,
// but the render shader only reads the position.
