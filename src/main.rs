// Importe le module contenant notre application de simulation
mod instances_app;

use std::sync::Arc;

use crate::instances_app::InstanceApp;
use wgpu_bootstrap::{egui, Runner};

fn main() {
    // === QU'EST-CE QUE LE RUNNER ? ===
    //
    // Le Runner est un composant du framework wgpu_bootstrap qui gère TOUTE
    // l'infrastructure nécessaire pour faire tourner une application GPU :
    //
    // 1. Création de la fenêtre (avec winit)
    // 2. Initialisation de WebGPU/wgpu (device, queue, surface)
    // 3. Boucle de rendu principale (game loop)
    // 4. Gestion des événements (souris, clavier, redimensionnement)
    // 5. Intégration d'egui (interface graphique)
    // 6. Gestion du temps (delta_time, FPS)
    //
    // Sans le Runner, il faudrait écrire ~200 lignes de code boilerplate
    // pour gérer tout ça manuellement.
    
    let mut runner = Runner::new(
        "Simulation de Tissu GPU",
        
        // Largeur initiale de la fenêtre en pixels
        900,
        
        // Hauteur initiale de la fenêtre en pixels
        700,
        
        // Couleur de fond de l'interface egui (gris clair)
        // Ceci n'affecte PAS le rendu 3D, seulement l'UI
        egui::Color32::from_rgb(245, 245, 245),
        
        // Nombre de samples MSAA (antialiasing) - 32 = haute qualité
        // Plus élevé = meilleure qualité mais plus lent
        32,

        // Mode de présentation (0 = VSync activé, 1 = immédiat)
        // 0 = synchronisé avec l'écran (60 FPS typique)
        0,
        
        // === FONCTION DE CRÉATION DE L'APP ===
        //
        // Box::new(|context| ...) est une closure (fonction anonyme) qui :
        // 1. Reçoit le Context GPU (device, queue, format de surface)
        // 2. Crée notre InstanceApp avec ce contexte
        // 3. La met dans un Arc (pointeur partagé thread-safe)
        //
        // Le Runner appelle cette fonction UNE FOIS au démarrage pour
        // initialiser notre application.
        Box::new(|context| Arc::new(InstanceApp::new(context))),
    );
    
    // Lance la boucle de rendu infinie
    // Cette fonction ne retourne JAMAIS (jusqu'à fermeture de la fenêtre)
    // Elle appelle en boucle : update() → render() → render_gui()
    runner.run();
}
