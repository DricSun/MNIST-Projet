#!/usr/bin/env python3
"""
Script d'évaluation pour le modèle de classification MNIST avec PyTorch
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import torch

from src.model import MNISTModel
from src.data_loader import MNISTDataLoader

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Affiche la matrice de confusion avec un style amélioré"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    
    # Créer la heatmap avec un style moderne
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10),
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    
    plt.title('Matrice de Confusion', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Prédiction', fontsize=14)
    plt.ylabel('Vraie Classe', fontsize=14)
    
    # Ajouter des statistiques
    total = np.sum(cm)
    accuracy = np.trace(cm) / total
    plt.figtext(0.12, 0.02, f'Précision globale: {accuracy:.3f} ({accuracy*100:.1f}%)', 
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Matrice de confusion sauvegardée dans {save_path}")
    
    plt.tight_layout()
    plt.show()

def plot_prediction_examples(model, test_loader, num_examples=20, save_path=None):
    """Affiche des exemples de prédictions avec un style amélioré"""
    
    # Récupérer un batch de données
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    
    # Sélectionner les premiers échantillons
    images = images[:num_examples]
    labels = labels[:num_examples]
    
    fig, axes = plt.subplots(4, 5, figsize=(16, 13))
    axes = axes.ravel()
    
    model.model.eval()
    with torch.no_grad():
        for i in range(num_examples):
            # Dénormaliser l'image pour l'affichage
            img = images[i].squeeze().numpy()
            img = img * 0.3081 + 0.1307  # Inverse de la normalisation
            img = np.clip(img, 0, 1)
            
            true_label = labels[i].item()
            
            # Prédiction
            pred_label, confidence = model.predict(images[i].unsqueeze(0))
            
            # Affichage
            axes[i].imshow(img, cmap='gray')
            
            # Couleur du titre selon la correction de la prédiction
            color = '#2E8B57' if pred_label == true_label else '#DC143C'  # Vert ou rouge
            style = 'normal' if pred_label == true_label else 'italic'
            weight = 'bold' if pred_label == true_label else 'normal'
            
            axes[i].set_title(f'Vrai: {true_label}, Prédit: {pred_label}\nConfiance: {confidence:.3f}', 
                             color=color, fontsize=10, style=style, weight=weight)
            axes[i].axis('off')
            
            # Ajouter une bordure colorée
            for spine in axes[i].spines.values():
                spine.set_visible(True)
                spine.set_color(color)
                spine.set_linewidth(2)
    
    plt.suptitle('Exemples de Prédictions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"🖼️  Exemples sauvegardés dans {save_path}")
    
    plt.show()

def analyze_errors(model, test_loader, save_path=None, top_k=10):
    """Analyse détaillée des erreurs de classification"""
    print("🔍 Analyse des erreurs en cours...")
    
    model.model.eval()
    all_predictions = []
    all_confidences = []
    all_targets = []
    error_images = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(model.device)
            
            for i in range(data.size(0)):
                pred, conf = model.predict(data[i].unsqueeze(0))
                all_predictions.append(pred)
                all_confidences.append(conf)
                all_targets.append(target[i].item())
                
                # Stocker les erreurs avec leurs images
                if pred != target[i].item():
                    error_images.append({
                        'image': data[i].cpu(),
                        'true_label': target[i].item(),
                        'pred_label': pred,
                        'confidence': conf
                    })
            
            if (batch_idx + 1) % 50 == 0:
                print(f"   Traité {(batch_idx + 1) * test_loader.batch_size} échantillons")
    
    all_predictions = np.array(all_predictions)
    all_confidences = np.array(all_confidences)
    all_targets = np.array(all_targets)
    
    # Identifier les erreurs
    errors = all_predictions != all_targets
    error_indices = np.where(errors)[0]
    
    print(f"\n📈 Statistiques d'erreurs:")
    print(f"   - Nombre total d'erreurs: {len(error_indices)} sur {len(all_targets)}")
    print(f"   - Taux d'erreur: {len(error_indices)/len(all_targets)*100:.2f}%")
    print(f"   - Précision: {(1 - len(error_indices)/len(all_targets))*100:.2f}%")
    
    if len(error_indices) > 0:
        # Analyser les erreurs par classe
        print(f"\n📊 Erreurs par classe vraie:")
        error_by_class = {}
        for digit in range(10):
            digit_mask = all_targets == digit
            digit_errors = np.sum(errors & digit_mask)
            total_digit = np.sum(digit_mask)
            error_rate = digit_errors / total_digit * 100 if total_digit > 0 else 0
            error_by_class[digit] = error_rate
            print(f"   Chiffre {digit}: {digit_errors}/{total_digit} erreurs ({error_rate:.1f}%)")
        
        # Classe la plus problématique
        worst_class = max(error_by_class.items(), key=lambda x: x[1])
        best_class = min(error_by_class.items(), key=lambda x: x[1])
        print(f"\n🔴 Classe la plus problématique: {worst_class[0]} ({worst_class[1]:.1f}% d'erreurs)")
        print(f"🟢 Classe la plus fiable: {best_class[0]} ({best_class[1]:.1f}% d'erreurs)")
        
        # Matrice de confusion des erreurs
        print(f"\n🎯 Analyse des confusions:")
        conf_matrix = confusion_matrix(all_targets[errors], all_predictions[errors])
        
        # Top des confusions
        confusion_pairs = []
        for i in range(10):
            for j in range(10):
                if conf_matrix[i][j] > 0:
                    confusion_pairs.append((i, j, conf_matrix[i][j]))
        
        confusion_pairs.sort(key=lambda x: x[2], reverse=True)
        
        print(f"   Top 5 des confusions:")
        for i, (true_class, pred_class, count) in enumerate(confusion_pairs[:5]):
            total_true = np.sum(all_targets == true_class)
            percentage = count / total_true * 100
            print(f"   {i+1}. {true_class} → {pred_class}: {count} fois ({percentage:.1f}%)")
        
        # Afficher les erreurs les plus confiantes
        if len(error_images) >= top_k:
            print(f"\n🎭 Top {top_k} des erreurs les plus confiantes:")
            
            # Trier par confiance décroissante
            error_images.sort(key=lambda x: x['confidence'], reverse=True)
            top_errors = error_images[:top_k]
            
            fig, axes = plt.subplots(2, 5, figsize=(16, 8))
            axes = axes.ravel()
            
            for i, error in enumerate(top_errors):
                # Dénormaliser l'image
                img = error['image'].squeeze().numpy()
                img = img * 0.3081 + 0.1307
                img = np.clip(img, 0, 1)
                
                axes[i].imshow(img, cmap='gray')
                axes[i].set_title(f'Vrai: {error["true_label"]}, Prédit: {error["pred_label"]}\n'
                                 f'Conf: {error["confidence"]:.3f}', 
                                 color='red', fontsize=10, weight='bold')
                axes[i].axis('off')
                
                # Bordure rouge pour les erreurs
                for spine in axes[i].spines.values():
                    spine.set_visible(True)
                    spine.set_color('red')
                    spine.set_linewidth(2)
            
            plt.suptitle(f'Top {top_k} des Erreurs les Plus Confiantes', 
                        fontsize=14, fontweight='bold', color='red')
            plt.tight_layout()
            
            if save_path:
                error_save_path = save_path.replace('.png', '_errors.png')
                plt.savefig(error_save_path, dpi=300, bbox_inches='tight')
                print(f"❌ Analyse des erreurs sauvegardée dans {error_save_path}")
            
            plt.show()
            
            # Afficher quelques statistiques sur ces erreurs
            print(f"\n📊 Statistiques des erreurs les plus confiantes:")
            true_labels = [e['true_label'] for e in top_errors]
            pred_labels = [e['pred_label'] for e in top_errors]
            confidences = [e['confidence'] for e in top_errors]
            
            print(f"   - Confiance moyenne: {np.mean(confidences):.3f}")
            print(f"   - Confiance max: {np.max(confidences):.3f}")
            print(f"   - Confiance min: {np.min(confidences):.3f}")
            
            # Classes les plus représentées dans les erreurs confiantes
            from collections import Counter
            true_counter = Counter(true_labels)
            pred_counter = Counter(pred_labels)
            
            print(f"   - Classes vraies les plus confondues: {true_counter.most_common(3)}")
            print(f"   - Prédictions les plus confiantes (erronées): {pred_counter.most_common(3)}")
    
    return error_indices, all_predictions, all_confidences

def main():
    parser = argparse.ArgumentParser(description='Évaluer le modèle MNIST avec PyTorch')
    parser.add_argument('--model_path', type=str, default='models/mnist_model.pth',
                       help='Chemin du modèle à évaluer')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Taille du batch pour l\'évaluation')
    parser.add_argument('--analyze_errors', action='store_true',
                       help='Analyser les erreurs de classification')
    parser.add_argument('--show_examples', action='store_true',
                       help='Afficher des exemples de prédictions')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device à utiliser (cpu, cuda, mps, auto)')
    
    args = parser.parse_args()
    
    print("=== 🔍 Évaluation du Modèle MNIST (PyTorch) ===\n")
    
    # Configuration du device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"🚀 CUDA détecté: {torch.cuda.get_device_name()}")
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            print("🍎 Apple Silicon détecté: utilisation de MPS")
        else:
            device = torch.device('cpu')
            print("💻 Utilisation du CPU")
    else:
        device = torch.device(args.device)
        print(f"🎯 Device spécifié: {device}")
    
    # Vérifier que le modèle existe
    if not os.path.exists(args.model_path):
        print(f"❌ Erreur: Le modèle {args.model_path} n'existe pas.")
        print("Veuillez d'abord entraîner le modèle avec: python train.py")
        return
    
    # Créer les dossiers nécessaires
    os.makedirs('results', exist_ok=True)
    
    # Charger les données de test
    print("\n1. Chargement des données de test...")
    data_loader = MNISTDataLoader(batch_size=args.batch_size)
    train_loader, val_loader, test_loader = data_loader.load_data()
    
    print(f"   - Dataset de test: {len(test_loader.dataset)} échantillons")
    print(f"   - Batch size: {args.batch_size}")
    
    # Charger le modèle
    print("\n2. Chargement du modèle...")
    model = MNISTModel(device=device)
    model.load_model(args.model_path)
    
    # Afficher l'architecture
    print("\n3. Architecture du modèle:")
    model.get_model_summary()
    
    # Évaluation globale
    print("\n4. Évaluation globale...")
    test_loss, test_accuracy, all_predictions, all_targets = model.evaluate(test_loader)
    
    print(f"📊 Résultats de l'évaluation:")
    print(f"   - Perte de test: {test_loss:.4f}")
    print(f"   - Précision de test: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Rapport de classification détaillé
    print("\n5. Rapport de classification détaillé:")
    class_names = [f'Chiffre {i}' for i in range(10)]
    classification_rep = classification_report(
        all_targets, all_predictions, 
        target_names=class_names,
        digits=4
    )
    print(classification_rep)
    
    # Matrice de confusion
    print("\n6. Génération de la matrice de confusion...")
    plot_confusion_matrix(all_targets, all_predictions, save_path='results/confusion_matrix.png')
    
    # Exemples de prédictions
    if args.show_examples:
        print("\n7. Affichage d'exemples de prédictions...")
        plot_prediction_examples(model, test_loader, save_path='results/prediction_examples.png')
    
    # Analyse des erreurs
    if args.analyze_errors:
        print("\n8. Analyse détaillée des erreurs...")
        error_indices, predictions, confidences = analyze_errors(
            model, test_loader, save_path='results/error_analysis.png')
    
    # Statistiques par classe
    print("\n9. Statistiques par classe:")
    for i in range(10):
        class_mask = np.array(all_targets) == i
        class_preds = np.array(all_predictions)[class_mask]
        class_targets = np.array(all_targets)[class_mask]
        
        if len(class_targets) > 0:
            class_accuracy = np.mean(class_preds == class_targets)
            print(f"   Chiffre {i}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%) "
                  f"- {len(class_targets)} échantillons")
    
    # Sauvegarde du rapport complet
    print("\n10. Sauvegarde du rapport d'évaluation...")
    report_file = 'results/evaluation_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=== Rapport d'Évaluation MNIST (PyTorch) ===\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Modèle: {args.model_path}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Batch size: {args.batch_size}\n\n")
        
        f.write(f"=== Résultats Globaux ===\n")
        f.write(f"Précision: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)\n")
        f.write(f"Perte: {test_loss:.4f}\n")
        f.write(f"Échantillons testés: {len(all_targets)}\n\n")
        
        f.write(f"=== Rapport de Classification ===\n")
        f.write(classification_rep)
        f.write("\n")
        
        f.write(f"=== Précision par Classe ===\n")
        for i in range(10):
            class_mask = np.array(all_targets) == i
            class_preds = np.array(all_predictions)[class_mask]
            class_targets = np.array(all_targets)[class_mask]
            
            if len(class_targets) > 0:
                class_accuracy = np.mean(class_preds == class_targets)
                f.write(f"Chiffre {i}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%) "
                       f"- {len(class_targets)} échantillons\n")
    
    print(f"📄 Rapport complet sauvegardé dans {report_file}")
    
    # Messages de fin
    print("\n" + "="*60)
    print("✅ Évaluation terminée avec succès!")
    print(f"🏆 Précision finale: {test_accuracy*100:.2f}%")
    
    if test_accuracy > 0.99:
        print("🌟 Performance exceptionnelle! Modèle de très haute qualité.")
    elif test_accuracy > 0.98:
        print("⭐ Excellente performance! Modèle de haute qualité.")
    elif test_accuracy > 0.95:
        print("👍 Bonne performance! Modèle fonctionnel.")
    else:
        print("💡 Performance modeste. Le modèle pourrait être amélioré.")
    
    print(f"\n📊 Voir les résultats dans le dossier 'results/'")

if __name__ == "__main__":
    from datetime import datetime
    main() 