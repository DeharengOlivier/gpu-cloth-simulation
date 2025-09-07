// One cloth particle's state. position.xyz / speed.xyz carry the data;
// the w component is padding so the struct matches the std430 16-byte
// alignment expected by the GPU (the host-side Instance uses [f32; 4] too).
struct Instance {
    position: vec4<f32>,
    speed: vec4<f32>,
};

// Time uniform. Declared to match the bind group layout (binding 2) but not
// read by the current simulation step.
