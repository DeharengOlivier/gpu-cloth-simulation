//CAMERA (VUE ET PROJECTION)
struct CameraUniform {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
};

//INSTANCE (pour position et vitesse des objets)
struct Instance {
    position: vec3<f32>,
    speed: vec3<f32>,
};

//ACCES AUX DONNEES DE CAMERA ET INSTANCE
@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(1) var<storage> instances: array<Instance>;





//STRUCTURE VERTEX TISSU
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
};

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
    out.clip_position = camera.proj * camera.view * vec4<f32>(model.position + instance.pos, 1.0);
    return out;
}

// Grid fragment shader
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}



//STRUCTURE VERTEX SPHERE
struct SphereVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

@vertex
fn sphere_vs_main(model: VertexInput) -> SphereVertexOutput {
    var out: SphereVertexOutput;
    out.color = model.color;
    out.normal = (camera.view * vec4<f32>(model.normal, 0.0)).xyz;
    out.clip_position = camera.proj * camera.view * vec4<f32>(model.position, 1.0);
    return out;
}

@fragment
fn sphere_fs_main(in: SphereVertexOutput) -> @location(0) vec4<f32> {
    let light_dir = normalize(vec3<f32>(1.0, 1.0, 1.0));
    let diffuse = max(dot(normalize(in.normal), light_dir), 0.0);
    let final_color = in.color * (diffuse * 0.7 + 0.3);
    return vec4<f32>(final_color, 1.0);
}