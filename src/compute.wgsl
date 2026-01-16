//INSTANCES POSITION +VITESSE
struct Instance {
    position: vec4<f32>,
    speed: vec4<f32>,
};
//jsp
struct TimeUniform {
    generation_duration: f32,
};

//PARAMETRES POUR CALCUL FORCES
struct PhysicsParams {
    structural_k: f32,
    shear_k: f32,
    bend_k: f32,
    damping: f32,
    mass: f32,
    rest_length: f32,
    dt: f32,
    friciton: f32,
    sphere_radius: f32,
};

//ACCES AUX DONNEES POSITION+VITESSE, jsp, PARAMETRES
@group(0) @binding(0) var<storage, read_write> instances_ping: array<Instance>;
@group(0) @binding(1) var<storage, read_write> instances_pong: array<Instance>;
@group(0) @binding(2) var<uniform> time: TimeUniform;
@group(0) @binding(3) var<uniform> physics: PhysicsParams;





//CST FORCE GRAVITE ET POSITION SOL
const GRAVITY: f32 = -0.3;      //-0.5
const GROUND: f32 = -1.0;
const sqrt_of_two: f32 = 1.41421356237309504880168872420969807856967187537694807317667973799073247846210703885038753432764157273501384623;

//LOI HOOK (=CALCUL RESSORT): F=−k⋅(l−l0)
fn calculate_spring_force(pos1: vec3<f32>, pos2: vec3<f32>, vel1: vec3<f32>, vel2: vec3<f32>, rest_length: f32, k: f32,  damping: f32) -> vec3<f32> {
    let delta = pos2 - pos1;
    let velocity_delta = vel2 - vel1;
    let current_length = length(delta);

    if (current_length < 0.0001) {
        return vec3<f32>(0.0);
    }

    let direction = delta / current_length;

    //Force Ressort
    let spring_force = k * (current_length - rest_length) * direction;
    //Coefficient d'amortissement (Fd= -cd.v ) pour pas osciller
    let damping_force = damping * dot(velocity_delta, direction) * direction;

    return spring_force + damping_force;
}


//LIAISON PROCHE PARTICULES (COMME MAILLAGE)
fn enforce_distance_constraint(pos1: ptr<function, vec3<f32>>, pos2: ptr<function, vec3<f32>>, rest_length: f32, max_stretch: f32) {
    let delta = *pos2 - *pos1;
    let current_length = length(delta);
    
    if current_length > rest_length * max_stretch {
        let correction = delta * (1.0 - (rest_length * max_stretch) / current_length);
        *pos1 += correction * 0.5;
        *pos2 -= correction * 0.5;
    }
}






//SHADER CALCUL PRINCIPAL
@compute @workgroup_size(WORKGROUP_SIZE)
fn computeMain(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    var instance = instances_ping[index];

    //TAILLE MAX ETIREMENT DU TISSU
    let max_stretch = 100.0; // Allow 10% stretch

    //LOCALISATION PARTICULE DS GRILLE
    let grid_size = u32(sqrt(f32(arrayLength(&instances_ping))));
    let row = index / grid_size;
    let col = index % grid_size;

    //LOCALISATION PARTICULE DS ESPACE
    let pos = instance.position.xyz;
    let speed = instance.speed.xyz;
    var total_force = vec3<f32>(0.0, 0.0, 0.0);


    //RESSORT VIA LOI HOOK
    //RESSORT STRUCTUREL POUR RELIER PARTICULE AVEC VOISINS (GAUCHE, DROITE, BAS, HAUT)
    // Voisin gauche
    if (col > 0) {
        let left_index = index - 1;
        let left_pos = instances_ping[left_index].position.xyz;
        let left_speed = instances_ping[left_index].speed.xyz;
        total_force += calculate_spring_force(pos, left_pos, speed, left_speed, physics.rest_length, physics.structural_k, physics.damping);  
    }

    //Voisin droit
    if (col < grid_size - 1) {
        let right_index = index + 1;
        let right_pos = instances_ping[right_index].position.xyz;
        let right_speed = instances_ping[right_index].speed.xyz;
        total_force += calculate_spring_force(pos, right_pos, speed, right_speed, physics.rest_length, physics.structural_k, physics.damping);
    }

    //Voisin haut
    if (row > 0) {
        let up_index = index - grid_size;
        let up_pos = instances_ping[up_index].position.xyz;
        let up_speed = instances_ping[up_index].speed.xyz;
        total_force += calculate_spring_force(pos, up_pos, speed, up_speed, physics.rest_length, physics.structural_k, physics.damping);
    }

    //Voisin bas
    if (row < grid_size - 1) {
        let down_index = index + grid_size;
        let down_pos = instances_ping[down_index].position.xyz;
        let down_speed = instances_ping[down_index].speed.xyz;
        total_force += calculate_spring_force(pos, down_pos, speed, down_speed, physics.rest_length, physics.structural_k, physics.damping);
    }


    //RESSORTS DE CISAILLEMENT POUR RELIER PARTICULE AVEC VOISINS(DIAGONALES)
    //Diagonale haut-gauche
    if (row > 0 && col > 0) {
        let diag_index = index - grid_size - 1;
        let diag_pos = instances_ping[diag_index].position.xyz;
        let diag_speed = instances_ping[diag_index].speed.xyz;
        total_force += calculate_spring_force(pos, diag_pos, speed, diag_speed, physics.rest_length * sqrt_of_two, physics.shear_k, physics.damping);
    }

    //Diagonale haut-droite
    if (row > 0 && col < grid_size - 1) {
        let diag_index = index - grid_size + 1;
        let diag_pos = instances_ping[diag_index].position.xyz;
        let diag_speed = instances_ping[diag_index].speed.xyz;
        total_force += calculate_spring_force(pos, diag_pos, speed, diag_speed, physics.rest_length * sqrt_of_two, physics.shear_k, physics.damping);
    }

    //Diagonale bas-gauche
    if (row < grid_size - 1 && col > 0) {
        let diag_index = index + grid_size - 1;
        let diag_pos = instances_ping[diag_index].position.xyz;
        let diag_speed = instances_ping[diag_index].speed.xyz;
        total_force += calculate_spring_force(pos, diag_pos, speed, diag_speed, physics.rest_length * sqrt_of_two, physics.shear_k, physics.damping);
    }

    //Diagonale bas-droite
    if (row < grid_size - 1 && col < grid_size - 1) {
        let diag_index = index + grid_size + 1;
        let diag_pos = instances_ping[diag_index].position.xyz;
        let diag_speed = instances_ping[diag_index].speed.xyz;
        total_force += calculate_spring_force(pos, diag_pos, speed, diag_speed, physics.rest_length * sqrt_of_two, physics.shear_k, physics.damping);
    }

    //RESSORTS FLEXION POUR STABILITE ET RIGIDITE DU MAILLAGE
    //Horizontale gauche
    if (col > 1) {
        let bend_index = index - 2;
        let bend_pos = instances_ping[bend_index].position.xyz;
        let bend_speed = instances_ping[bend_index].speed.xyz;
        total_force += calculate_spring_force(pos, bend_pos, speed, bend_speed, physics.rest_length * 2.0, physics.bend_k, physics.damping);
    }

    //Horizontale droite
    if (col < grid_size - 2) {
        let bend_index = index + 2;
        let bend_pos = instances_ping[bend_index].position.xyz;
        let bend_speed = instances_ping[bend_index].speed.xyz;
        total_force += calculate_spring_force(pos, bend_pos, speed, bend_speed, physics.rest_length * 2.0, physics.bend_k, physics.damping);
    }

    //Verticale-haut
    if (row > 1) {
        let bend_index = index - (grid_size * 2);
        let bend_pos = instances_ping[bend_index].position.xyz;
        let bend_speed = instances_ping[bend_index].speed.xyz;
        total_force += calculate_spring_force(pos, bend_pos, speed, bend_speed, physics.rest_length * 2.0, physics.bend_k, physics.damping);
    }

    //Verticale-bas
    if (row < grid_size - 2) {
        let bend_index = index + (grid_size * 2);
        let bend_pos = instances_ping[bend_index].position.xyz;
        let bend_speed = instances_ping[bend_index].speed.xyz;
        total_force += calculate_spring_force(pos, bend_pos, speed, bend_speed, physics.rest_length * 2.0, physics.bend_k, physics.damping);
    }




    //FORCE D AMORTISSEMENT
    let damping_force = -physics.damping * instance.speed.xyz;
    total_force += damping_force;

    //GRAVITE VERS BAS
    total_force += vec3<f32>(0.0, GRAVITY * physics.mass, 0.0);


    //COLLISION TISSU-SPHERE AVEC FORCE DE FROTTEMENT (TG A LA SURFACE DE SPHERE)
    let distance = length(instance.position.xyz);
    let radius = physics.sphere_radius;
    
    if (distance < radius) {
        let normal = normalize(instance.position.xyz);
        
        //Repositionnement sur Surface Sphère
        instance.position.x = normal.x * radius;
        instance.position.y = normal.y * radius;
        instance.position.z = normal.z * radius;


        //FORCE FROTTEMENT (=Ff=−min(∣Rot∣,cf∣Ron∣)1t)
        //Resultante +Vecteur normal (centre de la sphère au point)
        let Ro = total_force;
        let In = normal;
        
        //Composante normale (Ro.n)
        let Ro_n_magnitude = dot(Ro, In);
        let Ro_n = In * Ro_n_magnitude;
        
        //Composante tangentielle (Ro.t)
        let Ro_t = Ro - Ro_n;
        let Ro_t_magnitude = length(Ro_t);
        
        //Si Composante tangentielle pas nulle
        if (Ro_t_magnitude > 0.0001) {
            let It = Ro_t / Ro_t_magnitude;
            
            //Coefficient de frottement
            let cf = 0.9; // Ajustez cette valeur selon vos besoins
            
            //Force forttement
            let friction_magnitude = min(Ro_t_magnitude, cf * abs(Ro_n_magnitude));
            let friction_force = -friction_magnitude * It;
        
            total_force += friction_force;
        }

        //VITESSE AVEC AMORTISSEMENT
        let damping = 0.5;
        let dot_product = dot(instance.speed.xyz, normal);
        instance.speed.x = (instance.speed.x - 2.0 * dot_product * normal.x) * damping;
        instance.speed.y = (instance.speed.y - 2.0 * dot_product * normal.y) * damping;
        instance.speed.z = (instance.speed.z - 2.0 * dot_product * normal.z) * damping;
    }




    //COLLISION SOL
    //Position Sol
    if (instance.position.y < GROUND) {
        instance.position.y = GROUND;
        let ground_damping = 0.2;
        instance.speed.y = -instance.speed.y * ground_damping;
    }

    //Update Vitesse
    let acceleration = total_force / physics.mass;
    instance.speed.x += acceleration.x * physics.dt;
    instance.speed.y += acceleration.y * physics.dt;
    instance.speed.z += acceleration.z * physics.dt;

    //Update Position
    instance.position.x += instance.speed.x * physics.dt;
    instance.position.y += instance.speed.y * physics.dt;
    instance.position.z += instance.speed.z * physics.dt;

    //Contraintes de distance avec Voisins
    //Voisin gauche
    if (col > 0) {
        var pos1 = instance.position.xyz;
        var pos2 = instances_ping[index - 1].position.xyz;
        enforce_distance_constraint(&pos1, &pos2, physics.rest_length, max_stretch);
        instance.position.x = pos1.x;
        instance.position.y = pos1.y;
        instance.position.z = pos1.z;

    }
    //Voisin droit
    if (col < grid_size - 1) {
        var pos1 = instance.position.xyz;
        var pos2 = instances_ping[index + 1].position.xyz;
        enforce_distance_constraint(&pos1, &pos2, physics.rest_length, max_stretch);
        instance.position.x = pos1.x;
        instance.position.y = pos1.y;
        instance.position.z = pos1.z;
    }
    //Voisin haut
    if (row > 0) {
        var pos1 = instance.position.xyz;
        var pos2 = instances_ping[index - grid_size].position.xyz;
        enforce_distance_constraint(&pos1, &pos2, physics.rest_length, max_stretch);
        instance.position.x = pos1.x;
        instance.position.y = pos1.y;
        instance.position.z = pos1.z;
    }
    // Voisin bas
    if (row < grid_size - 1) {
        var pos1 = instance.position.xyz;
        var pos2 = instances_ping[index + grid_size].position.xyz;
        enforce_distance_constraint(&pos1, &pos2, physics.rest_length, max_stretch);
        instance.position.x = pos1.x;
        instance.position.y = pos1.y;
        instance.position.z = pos1.z;
    }
    //Voisin Diagonale haut-gauche
    if (row > 0 && col > 0) {
        var pos1 = instance.position.xyz;
        var pos2 = instances_ping[index - grid_size - 1].position.xyz;
        enforce_distance_constraint(&pos1, &pos2, physics.rest_length * sqrt_of_two, max_stretch);
        instance.position.x = pos1.x;
        instance.position.y = pos1.y;
        instance.position.z = pos1.z;
    }
    //Voisin Diagonale haut-doite
    if (row > 0 && col < grid_size - 1) {
        var pos1 = instance.position.xyz;
        var pos2 = instances_ping[index - grid_size + 1].position.xyz;
        enforce_distance_constraint(&pos1, &pos2, physics.rest_length * sqrt_of_two, max_stretch);
        instance.position.x = pos1.x;
        instance.position.y = pos1.y;
        instance.position.z = pos1.z;
    }
     //Voisin Diagonale bas-gauche
    if (row < grid_size - 1 && col > 0) {
        var pos1 = instance.position.xyz;
        var pos2 = instances_ping[index + grid_size - 1].position.xyz;
        enforce_distance_constraint(&pos1, &pos2, physics.rest_length * sqrt_of_two, max_stretch);
        instance.position.x = pos1.x;
        instance.position.y = pos1.y;
        instance.position.z = pos1.z;
    }
     //Voisin Diagonale bas-droite
    if (row < grid_size - 1 && col < grid_size - 1) {
        var pos1 = instance.position.xyz;
        var pos2 = instances_ping[index + grid_size + 1].position.xyz;
        enforce_distance_constraint(&pos1, &pos2, physics.rest_length * sqrt_of_two, max_stretch);
        instance.position.x = pos1.x;
        instance.position.y = pos1.y;
        instance.position.z = pos1.z;
    }


    instances_pong[index] = instance;
}