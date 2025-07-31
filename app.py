#!/usr/bin/env python3
"""
Interface web Flask pour la classification de chiffres MNIST avec PyTorch
"""

import os
import io
import base64
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from flask_cors import CORS
import cv2
import torch

from src.model import MNISTModel
from src.data_loader import MNISTDataLoader

app = Flask(__name__, template_folder='docs')
app.secret_key = 'mnist_classification_secret_key'
CORS(app)

# Configuration
MODEL_PATH = 'models/mnist_model.pth'

# Créer les dossiers nécessaires
os.makedirs('docs', exist_ok=True)

# Charger le modèle globalement
model = None
device = None

def get_device():
    """Détermine le meilleur device disponible"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def load_model():
    """Charge le modèle MNIST PyTorch"""
    global model, device
    if model is None:
        if os.path.exists(MODEL_PATH):
            device = get_device()
            model = MNISTModel(device=device)
            model.load_model(MODEL_PATH)
            print(f"✅ Modèle PyTorch chargé depuis {MODEL_PATH}")
            print(f"🎯 Device utilisé: {device}")
        else:
            print(f"❌ Modèle non trouvé dans {MODEL_PATH}")
            print("Veuillez d'abord entraîner le modèle avec: python train.py")
    return model



def process_canvas_image(image_data):
    """Traite une image dessinée sur le canvas avec PyTorch"""
    try:
        print(f"🔍 DEBUG Canvas - Début du traitement")
        print(f"🔍 Données reçues: {len(image_data)} caractères, début: {image_data[:50]}...")
        
        # Décoder l'image base64
        header, encoded = image_data.split(',', 1)
        image_bytes = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_bytes))
        
        print(f"🔍 Image décodée: taille={image.size}, mode={image.mode}")
        
        # Convertir en niveaux de gris avec gestion de la transparence
        if image.mode == 'RGBA':
            # Créer un fond blanc et composer l'image dessus
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])  # Utiliser le canal alpha comme masque
            image = background.convert('L')
            print(f"🔍 Conversion RGBA→L avec fond blanc")
        elif image.mode != 'L':
            image = image.convert('L')
        
        # Convertir en array numpy
        img_array = np.array(image)
        print(f"🔍 Array original: shape={img_array.shape}, min/max={img_array.min()}/{img_array.max()}")
        
        # Redimensionner à 28x28
        img_resized = cv2.resize(img_array, (28, 28))
        print(f"🔍 Après resize: shape={img_resized.shape}, min/max={img_resized.min()}/{img_resized.max()}")
        
        # Inverser les couleurs (canvas noir sur blanc -> blanc sur noir)
        img_resized = 255 - img_resized
        print(f"🔍 Après inversion: min/max={img_resized.min()}/{img_resized.max()}")
        
        # Normaliser (0-255 -> 0-1)
        img_normalized = img_resized.astype('float32') / 255.0
        print(f"🔍 Après norm 0-1: min/max={img_normalized.min():.3f}/{img_normalized.max():.3f}")
        
        # Appliquer la même normalisation que pour l'entraînement
        img_normalized = (img_normalized - 0.1307) / 0.3081
        print(f"🔍 Après norm MNIST: min/max={img_normalized.min():.3f}/{img_normalized.max():.3f}")
        
        # Vérifier les valeurs uniques
        unique_vals = np.unique(img_normalized)
        print(f"🔍 Valeurs uniques: {len(unique_vals)} (échantillon: {unique_vals[:5]})")
        
        # Convertir en tensor PyTorch
        img_tensor = torch.FloatTensor(img_normalized).unsqueeze(0).unsqueeze(0)
        print(f"🔍 Tensor final: shape={img_tensor.shape}")
        
        return img_tensor, img_resized
    
    except Exception as e:
        print(f"❌ Erreur lors du traitement canvas: {e}")
        import traceback
        traceback.print_exc()
        return None, None

@app.route('/')
def index():
    """Page d'accueil"""
    return render_template('index.html')



@app.route('/predict_canvas', methods=['POST'])
def predict_canvas():
    """Prédiction à partir du dessin sur canvas"""
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'Données d\'image manquantes'}), 400
        
        # Charger le modèle
        model = load_model()
        if model is None:
            return jsonify({'error': 'Modèle non disponible. Veuillez d\'abord entraîner le modèle avec: python train.py'}), 500
        
        # Traiter l'image du canvas
        processed_img, display_img = process_canvas_image(data['image'])
        if processed_img is None:
            return jsonify({'error': 'Erreur lors du traitement de l\'image du canvas'}), 400
        
        # Faire la prédiction
        print(f"🔍 DEBUG - Avant prédiction, input shape: {processed_img.shape}")
        prediction, confidence = model.predict(processed_img)
        print(f"🔍 DEBUG - Après prédiction: {prediction} (confiance: {confidence:.3f})")
        
        response_data = {
            'prediction': int(prediction),
            'confidence': float(confidence),
            'message': f'Le chiffre prédit est {prediction} avec une confiance de {confidence*100:.1f}%'
        }
        
        print(f"🎨 Canvas - Prédiction: {prediction}, Confiance: {confidence:.3f}")
        return jsonify(response_data)
    
    except Exception as e:
        print(f"Erreur dans predict_canvas: {str(e)}")
        return jsonify({'error': f'Erreur interne du serveur: {str(e)}'}), 500

@app.route('/model_info')
def model_info():
    """Informations sur le modèle PyTorch"""
    try:
        model = load_model()
        if model is None:
            return jsonify({'error': 'Modèle non disponible'}), 500
        
        # Compter les paramètres
        total_params = sum(p.numel() for p in model.model.parameters())
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        
        info = {
            'model_loaded': True,
            'model_path': MODEL_PATH,
            'framework': 'PyTorch',
            'device': str(device),
            'input_shape': [1, 28, 28],
            'output_classes': 10,
            'class_names': [str(i) for i in range(10)],
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': round(total_params * 4 / (1024 * 1024), 2),  # Estimation en float32
            'architecture': {
                'type': 'Convolutional Neural Network (CNN)',
                'layers': [
                    'Conv2d(1, 32, kernel_size=3, padding=1)',
                    'MaxPool2d(2, 2)',
                    'Conv2d(32, 64, kernel_size=3, padding=1)', 
                    'MaxPool2d(2, 2)',
                    'Conv2d(64, 64, kernel_size=3, padding=1)',
                    'Linear(3136, 64)',
                    'Dropout(0.5)',
                    'Linear(64, 10)'
                ]
            }
        }
        
        return jsonify(info)
    
    except Exception as e:
        return jsonify({'error': f'Erreur: {str(e)}'}), 500

@app.route('/health')
def health():
    """Endpoint de santé"""
    global model, device
    
    health_info = {
        'status': 'healthy',
        'model_available': model is not None,
        'framework': 'PyTorch',
        'pytorch_version': torch.__version__
    }
    
    if device:
        health_info['device'] = str(device)
        
    if torch.cuda.is_available():
        health_info['cuda_available'] = True
        health_info['cuda_device_count'] = torch.cuda.device_count()
        if torch.cuda.current_device() >= 0:
            health_info['cuda_current_device'] = torch.cuda.get_device_name()
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        health_info['mps_available'] = True
    
    return jsonify(health_info)

@app.route('/test_prediction')
def test_prediction():
    """Endpoint de test pour vérifier que le modèle fonctionne"""
    try:
        model = load_model()
        if model is None:
            return jsonify({'error': 'Modèle non disponible'}), 500
        
        # Créer une image de test simple (chiffre 0)
        test_image = np.zeros((28, 28), dtype=np.float32)
        
        # Dessiner un cercle simple pour ressembler à un 0
        center = (14, 14)
        radius = 8
        y, x = np.ogrid[:28, :28]
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        outer_mask = (x - center[0])**2 + (y - center[1])**2 <= (radius-2)**2
        test_image[mask & ~outer_mask] = 1.0
        
        # Normaliser comme dans l'entraînement
        test_image = (test_image - 0.1307) / 0.3081
        
        # Convertir en tensor PyTorch
        test_tensor = torch.FloatTensor(test_image).unsqueeze(0).unsqueeze(0)
        
        # Prédiction
        prediction, confidence = model.predict(test_tensor)
        
        return jsonify({
            'test_successful': True,
            'prediction': int(prediction),
            'confidence': float(confidence),
            'message': f'Test réussi! Prédiction: {prediction}, Confiance: {confidence:.3f}'
        })
    
    except Exception as e:
        return jsonify({
            'test_successful': False,
            'error': str(e)
        }), 500

@app.errorhandler(413)
def too_large(e):
    """Gestionnaire d'erreur pour les fichiers trop volumineux"""
    return jsonify({'error': 'Fichier trop volumineux. Taille maximale: 16MB'}), 413

@app.errorhandler(500)
def internal_error(e):
    """Gestionnaire d'erreur interne"""
    return jsonify({'error': 'Erreur interne du serveur'}), 500

if __name__ == '__main__':
    print("=== 🔥 Interface Web MNIST avec PyTorch ===")
    print("Initialisation...")
    
    # Détection du device
    device = get_device()
    print(f"🎯 Device détecté: {device}")
    
    # Information sur PyTorch
    print(f"🔥 Version PyTorch: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"⚡ CUDA disponible: {torch.cuda.device_count()} device(s)")
        print(f"   GPU actuel: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("🍎 Apple Silicon MPS disponible")
    
    print("\nChargement du modèle...")
    load_model()
    
    if model is not None:
        print("✅ Modèle chargé avec succès!")
        print("🌐 Démarrage du serveur web...")
        print("📍 Accédez à l'application: http://localhost:5001")
        print("\n🎯 Endpoints disponibles:")
        print("   - GET  /              : Interface principale")
        print("   - POST /predict_canvas: Prédiction par dessin")
        print("   - GET  /model_info    : Informations du modèle")
        print("   - GET  /health        : État du système")
        print("   - GET  /test_prediction: Test du modèle")
    else:
        print("⚠️  Modèle non trouvé, mais le serveur démarre quand même.")
        print("   Entraînez d'abord le modèle avec: python train.py")
        print("🌐 Serveur disponible sur: http://localhost:5001")
        print("   (La fonctionnalité de prédiction sera désactivée)")
    
    print("\n" + "="*50)
    
    app.run(debug=True, host='0.0.0.0', port=5001) 