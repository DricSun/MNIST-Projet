#!/usr/bin/env python3
"""
Script d'entraînement pour le modèle de classification MNIST avec PyTorch
"""

import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import torch

from src.model import MNISTModel
from src.data_loader import MNISTDataLoader

def plot_training_history(history, save_path=None):
    """Visualise l'historique d'entraînement"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Graphique de la précision
    epochs = range(1, len(history['train_accuracies']) + 1)
    ax1.plot(epochs, history['train_accuracies'], label='Précision d\'entraînement', marker='o', color='#2E86C1')
    ax1.plot(epochs, history['val_accuracies'], label='Précision de validation', marker='s', color='#E74C3C')
    ax1.set_title('Précision du modèle', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Époque')
    ax1.set_ylabel('Précision')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Graphique de la perte
    ax2.plot(epochs, history['train_losses'], label='Perte d\'entraînement', marker='o', color='#2E86C1')
    ax2.plot(epochs, history['val_losses'], label='Perte de validation', marker='s', color='#E74C3C')
    ax2.set_title('Perte du modèle', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Époque')
    ax2.set_ylabel('Perte')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Graphiques sauvegardés dans {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Entraîner le modèle MNIST avec PyTorch')
    parser.add_argument('--epochs', type=int, default=20, help='Nombre d\'époques (défaut: 20)')
    parser.add_argument('--batch_size', type=int, default=64, help='Taille du batch (défaut: 64)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Taux d\'apprentissage (défaut: 0.001)')
    parser.add_argument('--model_path', type=str, default='models/mnist_model.pth', 
                       help='Chemin pour sauvegarder le modèle')
    parser.add_argument('--visualize', action='store_true', 
                       help='Visualiser les échantillons de données')
    parser.add_argument('--use_augmentation', action='store_true',
                       help='Utiliser l\'augmentation de données')
    parser.add_argument('--no_save', action='store_true', 
                       help='Ne pas sauvegarder le modèle')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device à utiliser (cpu, cuda, mps, auto)')
    
    args = parser.parse_args()
    
    print("=== Projet de Classification MNIST avec PyTorch ===\n")
    
    # Configuration du device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"CUDA détecté: {torch.cuda.get_device_name()}")
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            print("🍎 Apple Silicon détecté: utilisation de MPS")
        else:
            device = torch.device('cpu')
            print("Utilisation du CPU")
    else:
        device = torch.device(args.device)
        print(f"Device spécifié: {device}")
    
    # Créer les dossiers nécessaires
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Charger les données
    print("\n1. Chargement des données...")
    data_loader = MNISTDataLoader(batch_size=args.batch_size)
    train_loader, val_loader, test_loader = data_loader.load_data(use_augmentation=args.use_augmentation)
    
    # Afficher les statistiques des données
    data_loader.get_data_stats()
    
    # Analyser la distribution
    data_loader.analyze_data_distribution()
    
    # Visualiser les échantillons si demandé
    if args.visualize:
        print("\n2. Visualisation des échantillons...")
        data_loader.visualize_samples(save_path='results/data_samples.png')
    
    # Créer et configurer le modèle
    print("\n3. Construction du modèle...")
    model = MNISTModel(device=device)
    model.get_model_summary()
    
    # Entraîner le modèle
    print(f"\n4. Entraînement du modèle ({args.epochs} époques)...")
    print(f"   Paramètres:")
    print(f"   - Batch size: {args.batch_size}")
    print(f"   - Learning rate: {args.learning_rate}")
    print(f"   - Augmentation: {'Oui' if args.use_augmentation else 'Non'}")
    print(f"   - Device: {device}")
    
    start_time = datetime.now()
    history = model.train(
        train_loader, 
        val_loader, 
        epochs=args.epochs, 
        learning_rate=args.learning_rate
    )
    end_time = datetime.now()
    
    training_time = end_time - start_time
    print(f"\nTemps d'entraînement: {training_time}")
    
    # Évaluer sur les données de test
    print("\n5. Évaluation sur les données de test...")
    test_loss, test_accuracy, _, _ = model.evaluate(test_loader)
    print(f"📊 Perte de test: {test_loss:.4f}")
    print(f"Précision de test: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Statistiques détaillées
    final_train_acc = history['train_accuracies'][-1]
    final_val_acc = history['val_accuracies'][-1]
    best_val_acc = max(history['val_accuracies'])
    
    print(f"\n📈 Résumé des performances:")
    print(f"   - Précision finale (train): {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
    print(f"   - Précision finale (val): {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
    print(f"   - Meilleure précision (val): {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"   - Précision de test: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Visualiser l'historique d'entraînement
    print("\n6. Visualisation des résultats...")
    plot_training_history(history, save_path='results/training_history.png')
    
    # Sauvegarder le modèle
    if not args.no_save:
        print(f"\n7. Sauvegarde du modèle...")
        model.save_model(args.model_path)
        
        # Export automatique vers ONNX
        print(f"\n8. Export automatique vers ONNX...")
        try:
            onnx_path = args.model_path.replace('.pth', '.onnx')
            
            # Import conditionnel pour éviter les erreurs si ONNX n'est pas installé
            import torch.onnx as torch_onnx
            
            # Créer un exemple d'entrée
            dummy_input = torch.randn(1, 1, 28, 28, device='cpu')
            
            # Copier le modèle sur CPU pour l'export ONNX
            model_cpu = MNISTModel(device=torch.device('cpu'))
            model_cpu.model.load_state_dict(model.model.state_dict())
            model_cpu.model.eval()
            
            # Export ONNX
            torch_onnx.export(
                model_cpu.model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            print(f"✅ Modèle ONNX sauvegardé: {onnx_path}")
            
        except ImportError:
            print("ONNX non installé. Pour l'export ONNX:")
            print("   pip install onnx onnxruntime")
        except Exception as e:
            print(f"Erreur lors de l'export ONNX: {e}")
        
        # Sauvegarder les métriques finales
        metrics_file = 'results/training_metrics.txt'
        with open(metrics_file, 'w', encoding='utf-8') as f:
            f.write(f"=== Métriques d'entraînement MNIST (PyTorch) ===\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Device: {device}\n")
            f.write(f"Époques: {args.epochs}\n")
            f.write(f"Batch size: {args.batch_size}\n")
            f.write(f"Learning rate: {args.learning_rate}\n")
            f.write(f"Augmentation: {'Oui' if args.use_augmentation else 'Non'}\n")
            f.write(f"Temps d'entraînement: {training_time}\n")
            f.write(f"Précision finale (train): {final_train_acc:.4f}\n")
            f.write(f"Précision finale (val): {final_val_acc:.4f}\n")
            f.write(f"Meilleure précision (val): {best_val_acc:.4f}\n")
            f.write(f"Précision de test: {test_accuracy:.4f}\n")
            f.write(f"Perte de test: {test_loss:.4f}\n")
            
            # Ajout de l'historique complet
            f.write(f"\n=== Historique d'entraînement ===\n")
            for epoch in range(len(history['train_accuracies'])):
                f.write(f"Époque {epoch+1}: Train Acc={history['train_accuracies'][epoch]:.4f}, "
                       f"Val Acc={history['val_accuracies'][epoch]:.4f}, "
                       f"Train Loss={history['train_losses'][epoch]:.4f}, "
                       f"Val Loss={history['val_losses'][epoch]:.4f}\n")
        
        print(f"📄 Métriques sauvegardées dans {metrics_file}")
    
    # Messages de fin
    print("\n" + "="*60)
    print("✅ Entraînement terminé avec succès!")
    print(f"Précision finale: {test_accuracy*100:.2f}%")
    
    if test_accuracy > 0.98:
        print("Excellent résultat! Le modèle a une très bonne performance.")
    elif test_accuracy > 0.95:
        print("👍 Bon résultat! Le modèle fonctionne bien.")
    else:
        print("Le modèle pourrait être amélioré. Essayez plus d'époques ou l'augmentation de données.")
    
    print(f"\nPour tester le modèle: python app.py")
    print(f"📊 Pour évaluer: python evaluate.py --model_path {args.model_path}")

if __name__ == "__main__":
    main() 