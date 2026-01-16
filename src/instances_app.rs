// ============================================================================
// IMPORTS ET DÉPENDANCES
// ============================================================================

// wgpu_bootstrap : Framework pour simplifier l'utilisation de WebGPU/wgpu
use wgpu_bootstrap::{
    cgmath::{self, InnerSpace}, // Bibliothèque mathématique pour calculs 3D (vecteurs, matrices)
    egui,                       // Bibliothèque d'interface graphique immédiate
    util::{
        geometry::icosphere,    // Génère une sphère subdivisée (icosaèdre)
        orbit_camera::{CameraUniform, OrbitCamera}, // Caméra orbitale pour navigation 3D
    },
    wgpu::{self, util::DeviceExt}, // API WebGPU pour calculs GPU
    App, Context,                  // Traits et structures du framework
};
use std::time::{Duration, Instant}; // Gestion du temps pour la physique

// ============================================================================
// STRUCTURES DE DONNÉES GPU
// ============================================================================

/// Structure Vertex : Représente un sommet de géométrie
/// 
/// #[repr(C)] : Garantit que la disposition en mémoire suit le standard C
///              Essentiel pour que Rust et le GPU s'accordent sur l'organisation des données
/// 
/// bytemuck::Pod : "Plain Old Data" - permet la conversion directe bytes <-> structure
/// bytemuck::Zeroable : permet d'initialiser avec des zéros (sécurité mémoire)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],  // Position (x, y, z) dans l'espace 3D
    normal: [f32; 3],    // Vecteur normal pour l'éclairage (non utilisé ici)
    color: [f32; 3],     // Couleur RGB (valeurs entre 0.0 et 1.0)
}

impl Vertex {
    /// Décrit comment le GPU doit interpréter les données de Vertex
    /// 
    /// Le VertexBufferLayout indique au GPU :
    /// - Comment les données sont espacées en mémoire (stride)
    /// - Quels attributs existent et où ils se trouvent (offset)
    /// - Comment passer au vertex suivant (step_mode)
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            // array_stride : Nombre d'octets entre le début d'un Vertex et le suivant
            // Ici : 9 floats × 4 bytes = 36 bytes
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            
            // step_mode : Comment itérer dans le buffer
            // Vertex = un nouveau vertex pour chaque sommet
            // Instance = même vertex pour toutes les instances (utilisé pour Instance)
            step_mode: wgpu::VertexStepMode::Vertex,
            
            // attributes : Liste des attributs et leur emplacement
            attributes: &[
                // Attribut 0 : Position
                wgpu::VertexAttribute {
                    offset: 0,  // Commence au début de la structure
                    shader_location: 0,  // Correspond à @location(0) dans le shader
                    format: wgpu::VertexFormat::Float32x3,  // 3 floats (x, y, z)
                },
                // Attribut 1 : Normal
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,  // Après position (12 bytes)
                    shader_location: 1,  // @location(1) dans le shader
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Attribut 2 : Couleur
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 6]>() as wgpu::BufferAddress,  // Après position + normal (24 bytes)
                    shader_location: 2,  // @location(2) dans le shader
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

/// Structure Instance : Représente une particule du tissu
/// 
/// Concept d'instancing : Au lieu de dupliquer la géométrie (petite sphère) pour chaque particule,
/// on la dessine une fois et on la "réinstancie" à différentes positions.
/// Chaque Instance contient les données uniques par particule.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Instance {
    position: [f32; 4],  // Position (x, y, z) de la particule + padding pour alignement GPU
    speed: [f32; 4],     // Vitesse (vx, vy, vz) + padding
                         // Note : vec4 utilisé pour alignement mémoire GPU (16 bytes)
}

impl Instance {
    /// Layout pour l'instancing
    /// 
    /// Différence clé avec Vertex : step_mode = Instance
    /// Cela signifie que ces données changent une fois par instance dessinée,
    /// pas une fois par sommet.
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Instance>() as wgpu::BufferAddress,
            
            // Instance : Ces données sont les mêmes pour tous les sommets d'une instance,
            // mais changent pour chaque instance
            step_mode: wgpu::VertexStepMode::Instance,
            
            attributes: &[
                // Position de l'instance
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 3,  // @location(3) dans le shader
                    format: wgpu::VertexFormat::Float32x3,  // On ignore le 4ème élément (padding)
                },
                // Vitesse de l'instance (non utilisée en rendu, mais stockée pour le compute)
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32;3]>() as wgpu::BufferAddress,
                    shader_location: 4,  // @location(4) dans le shader
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

/// Structure TimeUniform : Paramètres temporels
/// (Non utilisée actuellement dans ce projet)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct TimeUniform {
    generation_duration: f32,  // Durée pour générer une frame
}

/// Structure PhysicsParams : Paramètres de simulation physique
/// 
/// Concept de Uniform Buffer : Ces données sont constantes pendant un dispatch compute
/// et accessibles par tous les threads du shader. Permet de configurer la simulation
/// sans recompiler les shaders.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct PhysicsParams {
    structural_k: f32,    // Raideur des ressorts structurels (relient voisins directs)
    shear_k: f32,         // Raideur des ressorts de cisaillement (diagonales)
    bend_k: f32,          // Raideur des ressorts de flexion (distance 2)
    damping: f32,         // Coefficient d'amortissement (évite oscillations infinies)
    mass: f32,            // Masse de chaque particule (pour F=ma)
    rest_length: f32,     // Longueur au repos des ressorts (distance naturelle)
    dt: f32,              // Delta time (pas de temps de la simulation)
    friction: f32,        // Coefficient de frottement avec la sphère
    sphere_radius: f32,   // Rayon de la sphère de collision
}

// ============================================================================
// PARAMÈTRES DE L'APPLICATION
// ============================================================================

/// ClothSettings : Configuration modifiable via l'interface utilisateur
/// 
/// Cette structure regroupe tous les paramètres que l'utilisateur peut ajuster
/// en temps réel via l'interface egui. Certains changements nécessitent de
/// reconstruire les buffers GPU (grid_size, spacing).
#[derive(Clone)]
pub struct ClothSettings {
    pub grid_size: u32,        // Taille de la grille N×N (nombre de particules par côté)
    pub spacing: f32,          // Distance entre particules adjacentes
    pub point_size: f32,       // Taille de rendu de chaque particule (rayon de la mini-sphère)
    pub cloth_color: [f32; 3], // Couleur RGB du tissu
    pub sphere_color: [f32; 3],// Couleur RGB de la sphère centrale
}

impl Default for ClothSettings {
    /// Valeurs par défaut des paramètres
    fn default() -> Self {
        Self {
            grid_size: 256,              // Grille 256×256 = 65 536 particules
            spacing: 0.006,              // 6 millimètres entre particules
            point_size: 0.0033,          // Rayon des sphères de visualisation
            cloth_color: [1.0, 0.0, 0.0],// Rouge
            sphere_color: [0.5, 0.5, 0.5],// Gris
        }
    }
}

// ============================================================================
// STRUCTURE PRINCIPALE DE L'APPLICATION
// ============================================================================

/// InstanceApp : Cœur de l'application de simulation de tissu
/// 
/// Cette structure contient TOUS les états et ressources nécessaires pour
/// rendre et simuler le tissu. Elle implémente le trait App de wgpu_bootstrap.
pub struct InstanceApp {
    // === Buffers GPU pour le tissu ===
    vertex_buffer: wgpu::Buffer,        // Géométrie des mini-sphères (vertices)
    instance_buffer: [wgpu::Buffer; 2], // Position/vitesse des particules (technique ping-pong)
    index_buffer: wgpu::Buffer,         // Indices pour dessiner les triangles
    
    // === Pipelines de rendu et compute ===
    render_pipeline: wgpu::RenderPipeline,   // Pipeline pour dessiner le tissu
    compute_pipeline: wgpu::ComputePipeline, // Pipeline pour calculer la physique
    
    // === Métadonnées ===
    num_indices: u32,      // Nombre d'indices pour le tissu
    num_instances: u32,    // Nombre de particules (instances)
    
    // === Caméra ===
    camera: OrbitCamera,   // Caméra contrôlable à la souris
    last_generation: Instant, // Timestamp de la dernière frame (pour timing)
    
    // === Bind groups (liaisons de données GPU) ===
    // Concept : Les bind groups lient les buffers aux shaders
    // Ici on a 2 bind groups pour la technique ping-pong
    bind_group: [wgpu::BindGroup; 2],
    
    // === Buffers GPU pour la sphère centrale ===
    sphere_index_buffer: wgpu::Buffer,
    sphere_vertex_buffer: wgpu::Buffer,
    num_sphere_indices: u32,
    sphere_render_pipeline: wgpu::RenderPipeline,
    
    // === Paramètres d'interface utilisateur ===
    settings: ClothSettings,         // Paramètres actuellement appliqués
    pending_settings: ClothSettings, // Paramètres en attente d'application
    needs_rebuild: bool,             // Flag indiquant qu'une reconstruction est nécessaire
    paused: bool,                    // État pause/play de la simulation
}

// ============================================================================
// GÉNÉRATION DE LA GRILLE DE TISSU
// ============================================================================

/// Génère la grille de particules représentant le tissu
/// 
/// Concept clé : Separation of concerns
/// - Vertices : Géométrie de base (petite sphère) utilisée pour TOUTES les particules
/// - Instances : Position unique de chaque particule dans la grille
/// 
/// Le GPU dessinera la même géométrie (vertices) plusieurs fois, à différentes
/// positions (instances). C'est l'"instanced rendering".
/// 
/// # Arguments
/// * `rows` - Nombre de rangées de particules
/// * `cols` - Nombre de colonnes de particules
/// * `spacing` - Distance entre particules adjacentes
/// * `displacement` - Hauteur initiale du tissu
/// * `sphere_scale` - Rayon des mini-sphères de visualisation
/// * `cloth_color` - Couleur RGB des particules
/// 
/// # Retour
/// (vertices, index_buffer, instances, instances_copy, indices)
fn generate_grid(
    context: &Context,
    rows: u32,
    cols: u32,
    spacing: f32,
    displacement: f32,
    sphere_scale: f32,
    cloth_color: [f32; 3],
) -> (Vec<Vertex>, wgpu::Buffer, Vec<Instance>, Vec<Instance>, Vec<u32>) {
    // Génère une sphère subdivisée (icosaèdre de subdivision niveau 2)
    // Plus le niveau est élevé, plus la sphère est lisse mais coûteuse
    let (positions, indices) = icosphere(2);

    // Crée les vertices : transforme les positions normalisées de la sphère
    // en vertices avec position échelle, normale (ici nulle) et couleur
    let vertices: Vec<Vertex> = positions
        .iter()
        .map(|position| Vertex {
            position: (*position * sphere_scale).into(), // Échelle la sphère
            normal: [0.0, 0.0, 0.0],                     // Normal non utilisé pour le tissu
            color: cloth_color,                          // Couleur du tissu
        })
        .collect();

    // Crée l'index buffer : liste d'indices définissant les triangles
    // Concept : Au lieu de dupliquer les vertices, on référence par indices
    // Ex : triangle (0,1,2) utilise les vertices 0, 1 et 2
    let index_buffer = context
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            // bytemuck::cast_slice convertit Vec<u32> en &[u8] (bytes bruts)
            contents: bytemuck::cast_slice(indices.as_slice()),
            // INDEX : Ce buffer contient des indices pour l'indexed drawing
            usage: wgpu::BufferUsages::INDEX,
        });

    // Génère la grille de particules
    // Concept : Chaque particule est une Instance avec position et vitesse
    // La grille est centrée sur l'origine (0, 0, 0)
    let instances: Vec<Instance> = (0..rows)
        .flat_map(|row| {
            (0..cols).map(move |col| {
                Instance {
                    position: [
                        // Position X : centrée, espacée de 'spacing'
                        (col as f32 - cols as f32 / 2.0) * spacing,
                        // Position Y : hauteur initiale
                        displacement,
                        // Position Z : centrée, espacée de 'spacing'
                        (row as f32 - rows as f32 / 2.0) * spacing,
                        0.0, // Padding pour alignement GPU (vec4)
                    ],
                    speed: [0.0, 0.0, 0.0, 0.0], // Vitesse initiale nulle
                }
            })
        })
        .collect();

    // Clone pour la technique ping-pong (explication plus bas)
    let instances_copy = instances.clone();

    (vertices, index_buffer, instances, instances_copy, indices)
}

// ============================================================================
// CONSTANTES DE SIMULATION
// ============================================================================

/// Pas de temps fixe de la simulation en secondes
/// 
/// Concept : Fixed timestep
/// Au lieu d'utiliser le delta_time réel (variable selon FPS), on utilise
/// un pas de temps fixe pour garantir une simulation déterministe et stable.
/// Valeur : 0.0016s ≈ 1/625 ≈ 625 itérations par seconde
const TAYME: f32 = 0.0016;

/// Taille d'un workgroup GPU (nombre de threads par groupe)
/// 
/// Concept : GPU Compute Workgroups
/// Le GPU organise les threads en groupes (workgroups). Tous les threads
/// d'un workgroup s'exécutent ensemble et peuvent partager de la mémoire.
/// 256 est une valeur classique, bien supportée par la plupart des GPUs.
/// La grid_size doit être divisible par WORKGROUP_SIZE.
const WORKGROUP_SIZE: u32 = 256;

// ============================================================================
// FONCTIONS UTILITAIRES
// ============================================================================

/// Crée les vertices pour la sphère centrale (obstacle)
/// 
/// Cette sphère est statique et sert d'obstacle pour le tissu.
/// Contrairement au tissu, elle n'utilise PAS l'instancing : elle est
/// dessinée une seule fois.
fn create_sphere_vertices(sphere_radius: f32, sphere_color: [f32; 3]) -> (Vec<Vertex>, Vec<u32>) {
    // Subdivision niveau 3 = sphère plus détaillée que le tissu
    let (positions, indices) = icosphere(3);
    let vertices: Vec<Vertex> = positions
        .iter()
        .map(|position| {
            // Pour une sphère, la normale en chaque point pointe vers l'extérieur
            // Normaliser la position donne directement la normale
            let normal = position.normalize();
            Vertex {
                position: (normal * sphere_radius).into(), // Positionne à la surface
                normal: normal.into(),                     // Normale pour l'éclairage
                color: sphere_color,                       // Couleur de la sphère
            }
        })
        .collect();
    (vertices, indices)
}

// ============================================================================
// IMPLÉMENTATION DE L'APPLICATION
// ============================================================================

impl InstanceApp {
    /// Constructeur principal : initialise avec paramètres par défaut
    pub fn new(context: &Context) -> Self {
        let settings = ClothSettings::default();
        Self::create_with_settings(context, settings)
    }

    /// Constructeur avec paramètres personnalisés
    /// 
    /// Cette méthode fait TOUT le travail d'initialisation :
    /// 1. Création des buffers GPU
    /// 2. Compilation des shaders
    /// 3. Configuration des pipelines
    /// 4. Liaison des ressources (bind groups)
    fn create_with_settings(context: &Context, settings: ClothSettings) -> Self {
        // === ÉTAPE 1 : VALIDATION ET GÉNÉRATION DE LA GRILLE ===
        
        // S'assurer que grid_size est divisible par WORKGROUP_SIZE
        // Sinon, le compute shader ne pourra pas traiter toutes les particules
        let grid_size = (settings.grid_size / WORKGROUP_SIZE) * WORKGROUP_SIZE;
        let grid_size = grid_size.max(WORKGROUP_SIZE); // Minimum 256

        // Génère les vertices (géométrie des mini-sphères) et instances (particules)
        let (vertices, index_buffer, instances, instances_copy, indices) = generate_grid(
            &context,
            grid_size,       // Nombre de rangées
            grid_size,       // Nombre de colonnes
            settings.spacing,// Distance entre particules
            0.5,             // Hauteur initiale
            settings.point_size, // Rayon des sphères de visualisation
            settings.cloth_color, // Couleur
        );

        let num_indices = indices.len() as u32;     // Nombre d'indices pour dessiner
        let num_instances = instances.len() as u32; // Nombre de particules totales

        // === ÉTAPE 2 : CRÉATION DES UNIFORM BUFFERS ===
        
        // TimeUniform : non utilisé actuellement mais présent pour compatibilité
        let time_uniform = TimeUniform {
            generation_duration: Duration::new(0, 1_000_000).as_secs_f32(),
        };
        
        // Uniform Buffer : données constantes pendant un draw/compute call
        // COPY_DST : permet de mettre à jour avec queue.write_buffer()
        let time_buffer = context.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Time Uniform Buffer"),
            contents: bytemuck::cast_slice(&[time_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // === ÉTAPE 3 : CRÉATION DES VERTEX ET INSTANCE BUFFERS ===
        
        // Vertex Buffer : géométrie des mini-sphères (partagée par toutes les particules)
        // VERTEX : utilisation en tant que source de données vertex
        // COPY_DST : permet de modifier la couleur dynamiquement
        let vertex_buffer = context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(vertices.as_slice()),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });

        // === TECHNIQUE PING-PONG ===
        // 
        // Problème : Le compute shader doit lire ET écrire les positions/vitesses
        // Solution : 2 buffers qui alternent rôles lecture/écriture
        // 
        // Frame N :   Buffer[0] (lecture) → Compute → Buffer[1] (écriture)
        // Frame N+1 : Buffer[1] (lecture) → Compute → Buffer[0] (écriture)
        // 
        // Cela évite les conflits lecture/écriture (race conditions)
        let instance_buffer = [
            context
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Instance Buffer Ping"),
                    contents: bytemuck::cast_slice(&instances.as_slice()),
                    // STORAGE : accessible en lecture/écriture dans compute shader
                    // VERTEX : utilisable comme source de données d'instancing
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
                }),
            context
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Instance Buffer Pong"),
                    contents: bytemuck::cast_slice(&instances_copy.as_slice()),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
                }),
        ];

        // === ÉTAPE 4 : PARAMÈTRES PHYSIQUES ===
        
        let (_positions, _indices) = icosphere(3); // Non utilisés ici
        let sphere_radius = 0.4; // Rayon de la sphère obstacle

        // Configuration des forces et comportements physiques
        let physics_params = PhysicsParams {
            structural_k: 4000.0 * 1.5,  // Raideur structurelle (liens directs)
            shear_k: 2000.0 * 1.5,       // Raideur cisaillement (diagonales)
            bend_k: 300.0 * 1.5,         // Raideur flexion (distance 2)
            damping: 0.1,                // Amortissement (dissipe énergie)
            mass: 0.1,                   // Masse par particule
            rest_length: settings.spacing, // CRUCIAL : doit = spacing !
            dt: TAYME,                   // Pas de temps
            friction: 0.8,               // Frottement avec sphère
            sphere_radius: sphere_radius,// Rayon de collision
        };

        // Buffer uniform pour les paramètres physiques
        let physics_buffer = context.device().create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Physics Params Buffer"),
                contents: bytemuck::cast_slice(&[physics_params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }
        );

        // === ÉTAPE 5 : CRÉATION DE LA SPHÈRE OBSTACLE ===
        
        // Créer la sphère avec la couleur des settings
        let (sphere_vertices, sphere_indices) = create_sphere_vertices(sphere_radius, settings.sphere_color);

        // Buffers pour la sphère (statique, pas d'instancing)
        let sphere_vertex_buffer = context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Sphere Vertex Buffer"),
                contents: bytemuck::cast_slice(sphere_vertices.as_slice()),
                // COPY_DST : permet de changer la couleur dynamiquement
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });

        let sphere_index_buffer = context
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Sphere Index Buffer"),
                contents: bytemuck::cast_slice(sphere_indices.as_slice()),
                usage: wgpu::BufferUsages::INDEX,
            });

        // === ÉTAPE 6 : COMPILATION DES SHADERS ===
        
        // Shader de rendu (vertex + fragment)
        let shader = context
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
            });

        // Compute shader : calcule la physique sur GPU
        // Replace "WORKGROUP_SIZE" dans le code WGSL par la valeur réelle
        let compute_shader = context
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Compute Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("compute.wgsl")
                        .replace("WORKGROUP_SIZE", &format!("{}", WORKGROUP_SIZE))
                        .into()
                ),
            });

        // === ÉTAPE 7 : BIND GROUP LAYOUTS ===
        // 
        // Concept : Bind Group = ensemble de ressources liées ensemble
        // Un layout décrit QUELLES ressources sont attendues et COMMENT y accéder
        
        // Layout pour la caméra (matrices view + projection)
        let camera_bind_group_layout = context
            .device()
            .create_bind_group_layout(&CameraUniform::desc());

        // Layout pour le compute shader
        // Ce layout définit 4 bindings (ressources) :
        let instance_bind_group_layout = context.device().create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute Bind Group Layout"),
            entries: &[
                // Binding 0 : Buffer de lecture des instances (ping)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE, // Visible seulement dans compute
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false }, // read_write en réalité
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 1 : Buffer d'écriture des instances (pong)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 2 : Time uniform (non utilisé actuellement)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform, // Uniform = lecture seule, constant
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 3 : Paramètres physiques
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // === ÉTAPE 8 : LAYOUTS DE PIPELINE ===
        //
        // Concept : Pipeline Layout = organisation des bind groups pour un pipeline
        // Il définit QUELS bind groups seront utilisés et dans quel ordre
        
        // Layout pour le pipeline de rendu (dessin du tissu)
        // Utilise seulement la caméra (bind group 0)
        let pipeline_layout = context
            .device()
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout], // Bind group 0 = caméra
                push_constant_ranges: &[], // Pas de push constants (données immédiates)
            });

        // Layout pour le pipeline de compute (calcul physique)
        // Utilise les buffers d'instances et les paramètres physiques (bind group 0)
        let compute_pipeline_layout = context.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&instance_bind_group_layout], // Bind group 0 = instances + params
            push_constant_ranges: &[], // Pas de push constants
        });

        // === ÉTAPE 9 : CRÉATION DU PIPELINE DE RENDU ===
        //
        // Concept : Render Pipeline = configuration complète du processus de dessin
        // Définit comment transformer les vertices en pixels à l'écran
        let render_pipeline = context
            .device()
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Render Pipeline"),
                layout: Some(&pipeline_layout), // Utilise le layout défini plus haut
                
                // Stage Vertex : Transforme les positions 3D en coordonnées écran
                vertex: wgpu::VertexState {
                    module: &shader,              // Shader WGSL compilé
                    entry_point: "vs_main",       // Fonction d'entrée dans le shader
                    buffers: &[Vertex::desc(), Instance::desc()], // 2 buffers : géométrie + instances
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                
                // Stage Fragment : Calcule la couleur de chaque pixel
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_main",       // Fonction fragment dans le shader
                    targets: &[Some(wgpu::ColorTargetState {
                        format: context.format(),          // Format de la surface (ex: BGRA8)
                        blend: Some(wgpu::BlendState::REPLACE), // Pas de blending, remplace direct
                        write_mask: wgpu::ColorWrites::ALL,     // Écrit RGBA complet
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                
                // Configuration des primitives (triangles)
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList, // Liste de triangles
                    strip_index_format: None,                        // Pas de triangle strip
                    front_face: wgpu::FrontFace::Ccw,                // Sens antihoraire = face avant
                    cull_mode: Some(wgpu::Face::Back),               // Cull les faces arrière (optimisation)
                    polygon_mode: wgpu::PolygonMode::Fill,           // Remplir les triangles (pas wireframe)
                    unclipped_depth: false,
                    conservative: false,
                },
                
                // Test de profondeur : élimine les pixels cachés derrière d'autres
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: context.depth_stencil_format(),
                    depth_write_enabled: true,                      // Écrit dans le buffer de profondeur
                    depth_compare: wgpu::CompareFunction::Less,     // Garde le pixel le plus proche
                    stencil: wgpu::StencilState::default(),         // Pas de stencil utilisé
                    bias: wgpu::DepthBiasState::default(),
                }),
                
                // Multisampling : antialiasing (désactivé ici pour performance)
                multisample: wgpu::MultisampleState {
                    count: 1,                       // Pas de MSAA
                    mask: !0,                       // Tous les samples actifs
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,  // Pas de rendu stéréoscopique (VR)
                cache: None,      // Pas de cache de pipeline
            });

        // === ÉTAPE 10 : CONFIGURATION DE LA CAMÉRA ===
        //
        // Concept : Orbit Camera = caméra qui tourne autour d'un point central
        // Permet à l'utilisateur de visualiser la scène sous tous les angles
        let aspect = context.size().x / context.size().y; // Ratio largeur/hauteur de la fenêtre
        let mut camera = OrbitCamera::new(
            context,
            45.0,    // FOV (Field of View) en degrés - angle de vision
            aspect,  // Aspect ratio pour éviter la déformation
            0.1,     // Near plane - distance minimale de rendu
            100.0    // Far plane - distance maximale de rendu
        );
        // Positionne la caméra à 1.5 unités du centre, avec coordonnées polaires
        camera
            .set_polar(cgmath::point3(1.5, 0.0, 0.0))
            .update(context); // Calcule les matrices view/projection

        // === ÉTAPE 11 : CRÉATION DU PIPELINE DE COMPUTE ===
        //
        // Concept : Compute Pipeline = configuration pour exécuter des calculs parallèles sur GPU
        // Plus simple qu'un render pipeline car il n'y a pas de vertex/fragment, juste du calcul
        let compute_pipeline = context
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Compute Pipeline"),
                layout: Some(&compute_pipeline_layout), // Layout avec instances + params
                module: &compute_shader,                // Shader WGSL de physique
                entry_point: "computeMain",             // Fonction @compute dans le shader
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        // === ÉTAPE 12 : CRÉATION DES BIND GROUPS (TECHNIQUE PING-PONG) ===
        //
        // Concept crucial : PING-PONG BUFFERS
        // On crée 2 bind groups qui référencent les mêmes buffers mais INVERSÉS
        //
        // Bind Group Ping : lit buffer[0], écrit buffer[1]
        // Bind Group Pong : lit buffer[1], écrit buffer[0]
        //
        // À chaque frame, on alterne entre les deux pour éviter lecture/écriture simultanée
        let bind_group = [
            // === BIND GROUP PING (Frame paire) ===
            context
                .device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Bind Group Ping"),
                    layout: &instance_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,  // Binding 0 = buffer de LECTURE
                            resource: instance_buffer[0].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,  // Binding 1 = buffer d'ÉCRITURE
                            resource: instance_buffer[1].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,  // Time uniform (non utilisé)
                            resource: time_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,  // Paramètres physiques
                            resource: physics_buffer.as_entire_binding(),
                        }
                    ],
                }),
            // === BIND GROUP PONG (Frame impaire) ===
            // INVERSION : ce qui était lecture devient écriture et vice-versa
            context
                .device()
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Bind Group Pong"),
                    layout: &instance_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,  // Binding 0 = buffer de LECTURE (maintenant buffer[1])
                            resource: instance_buffer[1].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,  // Binding 1 = buffer d'ÉCRITURE (maintenant buffer[0])
                            resource: instance_buffer[0].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,  // Time uniform (identique)
                            resource: time_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,  // Paramètres physiques (identique)
                            resource: physics_buffer.as_entire_binding(),
                        }
                    ],
                }),
        ];

        // === ÉTAPE 13 : CRÉATION DU PIPELINE POUR LA SPHÈRE ===
        //
        // La sphère centrale utilise un pipeline séparé car :
        // - Elle n'utilise PAS l'instancing (dessinée une seule fois)
        // - Elle a des entry points différents dans le shader (sphere_vs_main, sphere_fs_main)
        let sphere_shader = context
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Sphere Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
            });

        // Layout identique au tissu : seulement la caméra
        let sphere_pipeline_layout = context
            .device()
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Sphere Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout],
                push_constant_ranges: &[],
            });

        // Pipeline de rendu pour la sphère
        // Différence clé : buffers: &[Vertex::desc()] - PAS d'Instance::desc() !
        // La sphère est statique, pas d'instancing nécessaire
        let sphere_render_pipeline = context
            .device()
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Sphere Render Pipeline"),
                layout: Some(&sphere_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &sphere_shader,
                    entry_point: "sphere_vs_main",       // Entry point spécifique sphère
                    buffers: &[Vertex::desc()],          // SEULEMENT vertices, pas d'instances
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &sphere_shader,
                    entry_point: "sphere_fs_main",       // Entry point spécifique sphère
                    targets: &[Some(wgpu::ColorTargetState {
                        format: context.format(),
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: context.depth_stencil_format(),
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
                cache: None,
            });

        // === ÉTAPE 14 : RETOUR DE LA STRUCTURE COMPLÈTE ===
        //
        // Tout est maintenant initialisé et prêt à l'emploi !
        // La structure contient tous les buffers, pipelines et états nécessaires
        Self {
            // Buffers du tissu
            vertex_buffer,         // Géométrie des mini-sphères
            instance_buffer,       // [2] buffers pour ping-pong
            index_buffer,          // Indices de triangles
            
            // Pipelines GPU
            render_pipeline,       // Dessin du tissu
            compute_pipeline,      // Calcul de physique
            
            // Métadonnées
            num_indices,           // Nombre d'indices à dessiner
            num_instances,         // Nombre de particules
            
            // Caméra et timing
            camera,                // Caméra orbitale contrôlable
            last_generation: Instant::now(), // Timestamp pour timing
            
            // Bind groups ping-pong
            bind_group,            // [2] bind groups alternants
            
            // Sphère centrale
            sphere_index_buffer,
            sphere_vertex_buffer,
            num_sphere_indices: sphere_indices.len() as u32,
            sphere_render_pipeline,
            
            // Paramètres utilisateur
            settings: settings.clone(),        // Paramètres actuels appliqués
            pending_settings: settings,        // Paramètres modifiés dans l'UI
            needs_rebuild: false,              // Flag reconstruction nécessaire
            paused: false,                     // État pause/play
        }
    }

    fn rebuild(&mut self, context: &Context) {
        // Cette méthode permet de reconstruire toute la simulation avec de nouveaux paramètres (taille, espacement, etc.)
        // Elle est appelée quand l'utilisateur clique sur "Appliquer et Redémarrer" dans l'UI.
        // On crée une nouvelle instance de l'application avec les nouveaux paramètres,
        // puis on remplace tous les buffers et pipelines par les nouveaux.
        let new_app = Self::create_with_settings(context, self.pending_settings.clone());
        self.vertex_buffer = new_app.vertex_buffer;
        self.instance_buffer = new_app.instance_buffer;
        self.index_buffer = new_app.index_buffer;
        self.num_indices = new_app.num_indices;
        self.num_instances = new_app.num_instances;
        self.bind_group = new_app.bind_group;
        self.sphere_vertex_buffer = new_app.sphere_vertex_buffer;
        self.sphere_index_buffer = new_app.sphere_index_buffer;
        self.num_sphere_indices = new_app.num_sphere_indices;
        self.settings = self.pending_settings.clone();
        self.needs_rebuild = false;
    }

    /// Met à jour dynamiquement les couleurs sans reconstruire les buffers
    /// 
    /// Cette méthode est appelée quand l'utilisateur change les couleurs via l'UI.
    /// Elle régénère les vertices avec les nouvelles couleurs et les envoie au GPU.
    fn update_colors(&mut self, context: &Context) {
        // Cette méthode permet de changer dynamiquement la couleur du tissu et de la sphère
        // sans avoir à tout reconstruire (plus rapide et fluide pour l'utilisateur)
        // On régénère les vertices avec la nouvelle couleur puis on les copie dans le buffer GPU
        let grid_size = (self.settings.grid_size / WORKGROUP_SIZE) * WORKGROUP_SIZE;
        let grid_size = grid_size.max(WORKGROUP_SIZE);
        let (new_vertices, _, _, _, _) = generate_grid(
            context,
            grid_size,
            grid_size,
            self.settings.spacing,
            0.5,
            self.settings.point_size,
            self.pending_settings.cloth_color, // Nouvelle couleur choisie dans l'UI
        );
        // Copie les nouveaux vertices dans le buffer GPU (write_buffer nécessite COPY_DST)
        context.queue().write_buffer(
            &self.vertex_buffer,
            0,  // Début du buffer
            bytemuck::cast_slice(&new_vertices),
        );

        // Idem pour la sphère centrale (obstacle)
        let (sphere_vertices, _) = create_sphere_vertices(0.4, self.pending_settings.sphere_color);
        context.queue().write_buffer(
            &self.sphere_vertex_buffer,
            0,
            bytemuck::cast_slice(&sphere_vertices),
        );

        // On met à jour les couleurs dans la structure settings
        self.settings.cloth_color = self.pending_settings.cloth_color;
        self.settings.sphere_color = self.pending_settings.sphere_color;
    }
}

impl App for InstanceApp {
    fn input(&mut self, input: egui::InputState, context: &Context) {
        // Gestion des entrées utilisateur (souris, clavier) pour la caméra orbitale
        self.camera.input(input, context);
    }

    fn render_gui(&mut self, egui_ctx: &egui::Context, context: &Context) {
        // Affiche la fenêtre de contrôle de l'interface graphique (egui)
        // Permet à l'utilisateur de modifier les couleurs, la taille de la grille, l'espacement, etc.
        egui::Window::new("Paramètres du Tissu").show(egui_ctx, |ui| {
            // Bouton pause/play pour arrêter ou reprendre la simulation
            if ui.button(if self.paused { "▶ Reprendre" } else { "⏸ Pause" }).clicked() {
                self.paused = !self.paused;
            }
            ui.separator();

            // Sélecteur de couleur pour le tissu
            ui.label("Couleur du tissu:");
            let mut cloth_color = self.pending_settings.cloth_color;
            if ui.color_edit_button_rgb(&mut cloth_color).changed() {
                self.pending_settings.cloth_color = cloth_color;
                self.update_colors(context); // Applique immédiatement la nouvelle couleur
            }

            // Sélecteur de couleur pour la sphère centrale
            ui.label("Couleur de la sphère:");
            let mut sphere_color = self.pending_settings.sphere_color;
            if ui.color_edit_button_rgb(&mut sphere_color).changed() {
                self.pending_settings.sphere_color = sphere_color;
                self.update_colors(context); // Applique immédiatement la nouvelle couleur
            }

            ui.separator();
            ui.label("Paramètres (redémarrage requis):");

            // Slider pour le nombre de points (taille de la grille)
            ui.horizontal(|ui| {
                ui.label("Taille grille:");
                let mut grid_val = self.pending_settings.grid_size as i32;
                if ui.add(egui::Slider::new(&mut grid_val, 64..=512).step_by(64.0)).changed() {
                    self.pending_settings.grid_size = grid_val as u32;
                }
            });
            ui.label(format!("  → {} particules", self.pending_settings.grid_size * self.pending_settings.grid_size));

            // Slider pour l'espacement entre les points
            ui.horizontal(|ui| {
                ui.label("Espacement:");
                ui.add(egui::Slider::new(&mut self.pending_settings.spacing, 0.002..=0.02).step_by(0.001));
            });

            // Slider pour la taille visuelle des points
            ui.horizontal(|ui| {
                ui.label("Taille points:");
                ui.add(egui::Slider::new(&mut self.pending_settings.point_size, 0.001..=0.01).step_by(0.0005));
            });

            ui.separator();

            // Affiche un avertissement si des paramètres nécessitent une reconstruction
            let settings_changed = self.pending_settings.grid_size != self.settings.grid_size
                || self.pending_settings.spacing != self.settings.spacing
                || self.pending_settings.point_size != self.settings.point_size;

            if settings_changed {
                ui.colored_label(egui::Color32::YELLOW, "⚠️ Changements en attente");
                if ui.button("🔄 Appliquer et Redémarrer").clicked() {
                    self.rebuild(context); // Reconstruit toute la simulation
                }
            }

            ui.separator();
            // Affiche le nombre total de particules simulées
            ui.label(format!("Particules: {}", self.num_instances));
        });
    }

    fn update(&mut self, delta_time: f32, context: &Context) {
        // Cette méthode est appelée à chaque frame pour faire avancer la simulation physique
        // Elle gère le "fixed timestep" pour la stabilité numérique
        if self.paused {
            // Si la simulation est en pause, on ne fait rien
            return;
        }

        let fixed_timestep = TAYME; // Pas de temps fixe (ex: 0.0016s)
        let mut accumulated_time = delta_time;

        // On peut accumuler du temps si le rendu est plus lent que la simulation
        // On exécute autant de steps de simulation que nécessaire pour rattraper le temps écoulé
        while accumulated_time >= fixed_timestep {
            // On crée un encodeur de commandes GPU pour le compute shader
            let mut encoder = context.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Compute Encoder"),
            });

            {
                // On démarre un "compute pass" pour exécuter le shader de physique
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Compute Pass"),
                    timestamp_writes: None,
                });

                // On sélectionne le pipeline compute (physique)
                compute_pass.set_pipeline(&self.compute_pipeline);
                // On lie le bind group courant (ping ou pong)
                compute_pass.set_bind_group(0, &self.bind_group[0], &[]);
                // On lance le shader sur tous les points (num_instances / WORKGROUP_SIZE workgroups)
                compute_pass.dispatch_workgroups(self.num_instances / WORKGROUP_SIZE, 1, 1);
            }

            // On soumet les commandes au GPU
            context.queue().submit(std::iter::once(encoder.finish()));

            // === TECHNIQUE PING-PONG ===
            // On échange les buffers de lecture/écriture pour la prochaine frame
            // Cela permet d'éviter les conflits d'accès mémoire sur le GPU
            self.instance_buffer.swap(0, 1);
            self.bind_group.swap(0, 1);

            // On décrémente le temps accumulé
            accumulated_time -= fixed_timestep;

            // On met à jour le timestamp de la dernière génération
            self.last_generation = Instant::now();
        }
    }
    fn render(&self, render_pass: &mut wgpu::RenderPass<'_>) {
        // Cette méthode dessine la scène à chaque frame
        // 1. On lie la caméra (matrices de vue/projection)
        render_pass.set_bind_group(0, self.camera.bind_group(), &[]);

        // 2. On dessine le tissu (toutes les particules via instancing)
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..)); // Géométrie de la mini-sphère
        render_pass.set_vertex_buffer(1, self.instance_buffer[0].slice(..)); // Positions des particules
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..self.num_indices, 0, 0..self.num_instances);

        // 3. On dessine la sphère centrale (obstacle)
        render_pass.set_pipeline(&self.sphere_render_pipeline);
        render_pass.set_vertex_buffer(0, self.sphere_vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.sphere_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..self.num_sphere_indices, 0, 0..1);
    }
}