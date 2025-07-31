#!/usr/bin/env python3
"""
Script d'export du modèle MNIST PyTorch vers ONNX
"""

import os
import argparse
import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np

from src.model import MNISTModel, MNISTNet
from src.data_loader import MNISTDataLoader

def export_to_onnx(model_path, onnx_path, batch_size=1):
    """Exporte le modèle PyTorch vers ONNX"""
    
    print("🔄 Début de l'export ONNX...")
    
    # Charger le modèle PyTorch
    print(f"📂 Chargement du modèle depuis {model_path}")
    device = torch.device('cpu')  # ONNX fonctionne mieux avec CPU
    model = MNISTModel(device=device)
    model.load_model(model_path)
    model.model.eval()
    
    # Créer un exemple d'entrée
    dummy_input = torch.randn(batch_size, 1, 28, 28, device=device)
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Export vers ONNX
    print(f"Export vers {onnx_path}...")
    
    torch.onnx.export(
        model.model,                     # Modèle PyTorch
        dummy_input,                     # Exemple d'entrée
        onnx_path,                       # Chemin de sortie
        export_params=True,              # Exporter les paramètres
        opset_version=17,                # Version ONNX (compatible avec onnxruntime-web)
        do_constant_folding=True,        # Optimisations
        input_names=['input'],           # Noms des entrées
        output_names=['output'],         # Noms des sorties
        dynamic_axes={                   # Axes dynamiques
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print("✅ Export ONNX terminé!")
    
    # Vérification du modèle ONNX
    print("🔍 Vérification du modèle ONNX...")
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✅ Modèle ONNX valide!")
        
        # Afficher les informations du modèle
        print(f"\n📊 Informations du modèle ONNX:")
        print(f"   - Version ONNX: {onnx_model.opset_import[0].version}")
        print(f"   - Inputs: {[input.name for input in onnx_model.graph.input]}")
        print(f"   - Outputs: {[output.name for output in onnx_model.graph.output]}")
        
        # Taille du fichier
        file_size = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"   - Taille du fichier: {file_size:.2f} MB")
        
    except Exception as e:
        print(f"❌ Erreur lors de la vérification: {e}")
        return False
    
    return True

def test_onnx_model(onnx_path, pytorch_model_path, num_tests=5):
    """Teste le modèle ONNX et compare avec PyTorch"""
    
    print(f"\n🧪 Test du modèle ONNX avec {num_tests} échantillons...")
    
    # Charger le modèle PyTorch original
    device = torch.device('cpu')
    pytorch_model = MNISTModel(device=device)
    pytorch_model.load_model(pytorch_model_path)
    pytorch_model.model.eval()
    
    # Charger le modèle ONNX
    ort_session = ort.InferenceSession(onnx_path)
    
    # Charger quelques données de test
    data_loader = MNISTDataLoader(batch_size=1)
    _, _, test_loader = data_loader.load_data()
    
    print("\n📊 Comparaison PyTorch vs ONNX:")
    print("-" * 60)
    
    correct_matches = 0
    
    for i, (data, target) in enumerate(test_loader):
        if i >= num_tests:
            break
        
        # Prédiction PyTorch
        with torch.no_grad():
            pytorch_output = pytorch_model.model(data)
            pytorch_pred = torch.argmax(pytorch_output, dim=1).item()
            pytorch_conf = torch.softmax(pytorch_output, dim=1).max().item()
        
        # Prédiction ONNX
        ort_inputs = {ort_session.get_inputs()[0].name: data.numpy()}
        ort_output = ort_session.run(None, ort_inputs)[0]
        
        # Convertir log_softmax en probabilities
        onnx_probs = np.exp(ort_output)
        onnx_pred = np.argmax(onnx_probs, axis=1)[0]
        onnx_conf = np.max(onnx_probs)
        
        # Vérifier la correspondance
        match = "✅" if pytorch_pred == onnx_pred else "❌"
        if pytorch_pred == onnx_pred:
            correct_matches += 1
        
        true_label = target.item()
        
        print(f"Échantillon {i+1}: Vrai={true_label} | "
              f"PyTorch={pytorch_pred}({pytorch_conf:.3f}) | "
              f"ONNX={onnx_pred}({onnx_conf:.3f}) {match}")
    
    accuracy = correct_matches / num_tests * 100
    print("-" * 60)
    print(f"Correspondance PyTorch-ONNX: {correct_matches}/{num_tests} ({accuracy:.1f}%)")
    
    if accuracy == 100:
        print("Parfait! Le modèle ONNX est identique au modèle PyTorch!")
    elif accuracy >= 95:
        print("👍 Très bon! Différences mineures acceptables.")
    else:
        print("Attention: Différences significatives détectées.")
    
    return accuracy

def optimize_onnx_model(onnx_path, optimized_path):
    """Optimise le modèle ONNX pour le web"""
    
    print(f"\nOptimisation du modèle ONNX...")
    
    try:
        import onnxoptimizer
        
        # Charger le modèle
        model = onnx.load(onnx_path)
        
        # Appliquer les optimisations
        optimized_model = onnxoptimizer.optimize(model)
        
        # Sauvegarder le modèle optimisé
        onnx.save(optimized_model, optimized_path)
        
        # Comparer les tailles
        original_size = os.path.getsize(onnx_path) / (1024 * 1024)
        optimized_size = os.path.getsize(optimized_path) / (1024 * 1024)
        reduction = (1 - optimized_size/original_size) * 100
        
        print(f"📊 Résultats de l'optimisation:")
        print(f"   - Taille originale: {original_size:.2f} MB")
        print(f"   - Taille optimisée: {optimized_size:.2f} MB")
        print(f"   - Réduction: {reduction:.1f}%")
        
        return True
        
    except ImportError:
        print("onnxoptimizer non installé. Installation...")
        os.system("pip install onnxoptimizer")
        return False
    except Exception as e:
        print(f"❌ Erreur lors de l'optimisation: {e}")
        return False

def create_web_assets(onnx_path):
    """Crée les assets pour l'utilisation web"""
    
    print(f"\n🌐 Création des assets web...")
    
    # Créer le dossier static si il n'existe pas
    os.makedirs('static/models', exist_ok=True)
    
    # Copier le modèle ONNX dans static
    import shutil
    web_model_path = 'static/models/mnist_model.onnx'
    shutil.copy2(onnx_path, web_model_path)
    
    # Créer un fichier de métadonnées JSON
    import json
    
    metadata = {
        "model_name": "MNIST CNN",
        "framework": "PyTorch -> ONNX",
        "input_shape": [1, 1, 28, 28],
        "output_shape": [1, 10],
        "input_name": "input",
        "output_name": "output",
        "classes": list(range(10)),
        "preprocessing": {
            "normalize": True,
            "mean": 0.1307,
            "std": 0.3081
        }
    }
    
    with open('static/models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Assets web créés:")
    print(f"   - Modèle: {web_model_path}")
    print(f"   - Métadonnées: static/models/model_metadata.json")

def main():
    parser = argparse.ArgumentParser(description='Exporter le modèle MNIST vers ONNX')
    parser.add_argument('--model_path', type=str, default='models/mnist_model.pth',
                       help='Chemin du modèle PyTorch (défaut: models/mnist_model.pth)')
    parser.add_argument('--onnx_path', type=str, default='models/mnist_model.onnx',
                       help='Chemin de sortie ONNX (défaut: models/mnist_model.onnx)')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Taille du batch pour l\'export (défaut: 1)')
    parser.add_argument('--test', action='store_true',
                       help='Tester le modèle ONNX après export')
    parser.add_argument('--optimize', action='store_true',
                       help='Optimiser le modèle ONNX')
    parser.add_argument('--web', action='store_true',
                       help='Créer les assets pour le web')
    
    args = parser.parse_args()
    
    print("=== 🔄 Export ONNX du Modèle MNIST ===\n")
    
    # Vérifier que le modèle PyTorch existe
    if not os.path.exists(args.model_path):
        print(f"❌ Erreur: Le modèle {args.model_path} n'existe pas.")
        print("Veuillez d'abord entraîner le modèle avec: python train.py")
        return
    
    # Créer le dossier models si nécessaire
    os.makedirs('models', exist_ok=True)
    
    # Export vers ONNX
    success = export_to_onnx(args.model_path, args.onnx_path, args.batch_size)
    
    if not success:
        print("❌ L'export ONNX a échoué.")
        return
    
    # Test du modèle ONNX
    if args.test:
        test_onnx_model(args.onnx_path, args.model_path)
    
    # Optimisation
    if args.optimize:
        optimized_path = args.onnx_path.replace('.onnx', '_optimized.onnx')
        optimize_onnx_model(args.onnx_path, optimized_path)
    
    # Création des assets web
    if args.web:
        create_web_assets(args.onnx_path)
    
    print(f"\nExport ONNX terminé avec succès!")
    print(f"📁 Modèle ONNX: {args.onnx_path}")
    
    # Instructions pour l'utilisation
    print(f"\nUtilisation:")
    print(f"   - Web: python export_onnx.py --web")
    print(f"   - Test: python export_onnx.py --test")
    print(f"   - Optimisation: python export_onnx.py --optimize")

if __name__ == "__main__":
    main() 