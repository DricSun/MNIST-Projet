# Projet de Classification MNIST avec PyTorch

Un projet complet de reconnaissance de chiffres manuscrits utilisant un réseau de neurones convolutionnel (CNN) avec **PyTorch** et une interface web interactive avec Flask.

## Lien du site

https://dricsun.github.io/MNIST-Projet/

## 📋 Aperçu du Projet

Ce projet implémente un système de classification automatique de chiffres manuscrits (0-9) basé sur le célèbre dataset MNIST. Il comprend :

- **Modèle CNN optimisé** avec PyTorch pour la reconnaissance de chiffres
- **Interface web interactive** pour tester le modèle en temps réel
- **Scripts d'entraînement et d'évaluation** complets avec visualisations avancées
- **Support multi-plateforme** : CPU, CUDA (NVIDIA), MPS (Apple Silicon)

## Fonctionnalités

- **Classification en temps réel** de chiffres manuscrits
- **Canvas interactif** pour dessiner des chiffres
- 📁 **Upload d'images** avec préprocessing automatique
- 📊 **Visualisations détaillées** des performances et métriques
- 🔍 **Analyse avancée des erreurs** de classification
- **Interface responsive** compatible mobile et desktop
- **Accélération GPU** automatique (CUDA/MPS)
- **Framework PyTorch** moderne et flexible

## Installation

### Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)

### Installation des dépendances

```bash
# Naviguer vers le dossier du projet
cd mnist-classification-project

# Installer les dépendances
pip install -r requirements.txt
```

**Note :** PyTorch s'installera automatiquement avec le support approprié pour votre système (CPU, CUDA, ou MPS).

## 📚 Structure du Projet

```
mnist-classification-project/
├── 📁 src/
│   ├── __init__.py
│   ├── model.py              # Modèle CNN PyTorch + architecture
│   └── data_loader.py        # DataLoaders PyTorch + préprocessing
├── 📁 models/
│   ├── __init__.py
│   └── mnist_model.pth       # Modèle entraîné (généré après entraînement)
├── 📁 templates/
│   └── index.html            # Interface web moderne
├── 📁 static/
│   └── uploads/              # Images uploadées (généré automatiquement)
├── 📁 results/               # Visualisations et métriques (généré automatiquement)
├── 📁 data/                  # Dataset MNIST (téléchargé automatiquement)
├── 🐍 train.py               # Script d'entraînement PyTorch
├── 🔍 evaluate.py            # Script d'évaluation et analyse
├── 🌐 app.py                 # Application web Flask
├── 📋 requirements.txt       # Dépendances PyTorch
└── 📖 README.md              # Cette documentation
```

## Utilisation

### 1. Entraînement du Modèle

```bash
# Entraînement basique (détection automatique GPU/CPU)
python train.py

# Entraînement avec paramètres personnalisés
python train.py --epochs 30 --batch_size 128 --learning_rate 0.001

# Entraînement avec augmentation de données et visualisations
python train.py --epochs 25 --use_augmentation --visualize

# Forcer l'utilisation d'un device spécifique
python train.py --device cuda    # Force CUDA
python train.py --device mps     # Force Apple Silicon MPS
python train.py --device cpu     # Force CPU

# Voir toutes les options
python train.py --help
```

**Options disponibles :**
- `--epochs` : Nombre d'époques d'entraînement (défaut: 20)
- `--batch_size` : Taille des batches (défaut: 64)
- `--learning_rate` : Taux d'apprentissage (défaut: 0.001)
- `--model_path` : Chemin de sauvegarde du modèle (défaut: models/mnist_model.pth)
- `--visualize` : Afficher des échantillons de données
- `--use_augmentation` : Utiliser l'augmentation de données
- `--device` : Device à utiliser (auto, cpu, cuda, mps)
- `--no_save` : Ne pas sauvegarder le modèle

### 2. Évaluation du Modèle

```bash
# Évaluation basique
python evaluate.py

# Évaluation complète avec analyse d'erreurs
python evaluate.py --analyze_errors --show_examples

# Utiliser un modèle spécifique avec un batch size plus grand
python evaluate.py --model_path models/mon_modele.pth --batch_size 128
```

**Options disponibles :**
- `--model_path` : Chemin du modèle à évaluer
- `--batch_size` : Taille du batch pour l'évaluation
- `--analyze_errors` : Analyser les erreurs de classification en détail
- `--show_examples` : Afficher des exemples de prédictions
- `--device` : Device à utiliser

### 3. Interface Web

```bash
# Lancer l'application web
python app.py
```

Puis ouvrez votre navigateur à l'adresse : `http://localhost:5000`

**Endpoints disponibles :**
- `GET /` : Interface principale
- `POST /predict_upload` : Prédiction via upload d'image
- `POST /predict_canvas` : Prédiction via dessin canvas
- `GET /model_info` : Informations détaillées sur le modèle
- `GET /health` : État de santé du système
- `GET /test_prediction` : Test de fonctionnement du modèle

## 🌐 Interface Web

L'interface web offre deux modes d'interaction :

### 📁 Upload d'Image
- Glissez-déposez ou sélectionnez une image de chiffre
- Formats supportés : PNG, JPG, JPEG, GIF, BMP
- Préprocessing automatique vers 28x28 pixels
- Affichage de l'image traitée

### Dessin Interactif
- Canvas HTML5 responsive pour dessiner des chiffres
- Support tactile pour mobiles et tablettes
- Prédiction instantanée avec score de confiance
- Interface intuitive et moderne

## Architecture du Modèle PyTorch

Le modèle CNN utilise l'architecture suivante :

```python
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.5)
```

**Architecture détaillée :**
```
Entrée (1, 28, 28)
    ↓
Conv2d(1→32, 3x3) + ReLU + MaxPool2d(2x2)
    ↓
Conv2d(32→64, 3x3) + ReLU + MaxPool2d(2x2)
    ↓
Conv2d(64→64, 3x3) + ReLU
    ↓
Flatten → Linear(576→64) + ReLU + Dropout(0.5)
    ↓
Linear(64→10) + LogSoftmax
```

**Optimisations PyTorch incluses :**
- **Adam Optimizer** avec learning rate adaptatif
- **ReduceLROnPlateau** pour ajuster automatiquement le taux d'apprentissage
- **Early Stopping** pour éviter le surapprentissage
- **Dropout** pour la régularisation
- **Normalisation des données** avec mean/std du dataset MNIST
- **Support GPU automatique** (CUDA/MPS)

## 📈 Performances Attendues

- **Précision sur le test** : ~99.2%
- **Temps d'entraînement** : 2-5 minutes (GPU) / 5-15 minutes (CPU)
- **Taille du modèle** : ~1.5MB
- **Temps de prédiction** : <50ms par image
- **Support des devices** : CPU, CUDA, Apple Silicon MPS

## Support GPU

Le projet détecte et utilise automatiquement l'accélération matérielle disponible :

### NVIDIA CUDA
```bash
# Vérification automatique
python train.py  # Utilisera CUDA si disponible
```

### Apple Silicon (M1/M2/M3)
```bash
# Support MPS automatique sur macOS
python train.py  # Utilisera MPS si disponible
```

### CPU Fallback
Le projet fonctionne parfaitement sur CPU si aucun GPU n'est disponible.

## Personnalisation

### Modifier l'Architecture du Modèle

Éditez le fichier `src/model.py` dans la classe `MNISTNet` :

```python
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        # Ajoutez/modifiez les couches ici
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)  # Plus de filtres
        # ...
```

### Ajuster les Hyperparamètres

```bash
# Modifier les paramètres d'entraînement
python train.py --learning_rate 0.0001 --batch_size 128 --epochs 50

# Utiliser l'augmentation de données
python train.py --use_augmentation
```

### Personnaliser les DataLoaders

Modifiez `src/data_loader.py` pour ajuster :
- Transformations des données
- Taille des batches
- Augmentation de données
- Normalisation personnalisée

## 🐛 Dépannage

### Erreur "Modèle non trouvé"
```bash
# Entraînez d'abord le modèle
python train.py
```

### Problèmes de dépendances PyTorch
```bash
# Réinstallez PyTorch pour votre système
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # Pour CUDA 11.8
# ou
pip install torch torchvision  # Pour CPU/MPS
```

### CUDA Out of Memory
```bash
# Réduisez la taille du batch
python train.py --batch_size 32
# ou forcez l'utilisation du CPU
python train.py --device cpu
```

### Problèmes avec l'interface web
```bash
# Vérifiez que le modèle existe
ls models/mnist_model.pth

# Testez le modèle directement
python -c "from src.model import MNISTModel; m = MNISTModel(); print('OK')"

# Redémarrez l'application
python app.py
```

## 📊 Métriques et Visualisations

Le projet génère automatiquement :

- **Graphiques d'entraînement** : précision et perte par époque
- **Matrice de confusion** interactive avec statistiques
- **Analyse des erreurs** : top des erreurs les plus confiantes
- **Distribution des classes** par ensemble de données
- **Exemples de prédictions** avec scores de confiance
- **Rapport d'évaluation complet** au format texte

Tous les résultats sont sauvegardés dans le dossier `results/`.

## 🤝 Contribution

1. Forkez le projet
2. Créez une branche pour votre fonctionnalité (`git checkout -b feature/AmazingFeature`)
3. Committez vos changements (`git commit -m 'Add some AmazingFeature'`)
4. Pushez vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une Pull Request

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.

## 🙏 Remerciements

- **Dataset MNIST** fourni par Yann LeCun et al.
- **PyTorch** pour le framework de deep learning moderne
- **Flask** pour l'interface web légère et flexible
- **torchvision** pour les transformations et datasets
- La **communauté PyTorch** pour les outils et ressources
- **Apple** et **NVIDIA** pour l'accélération matérielle

## 📞 Support

Pour toute question ou problème :

1. 📖 Consultez d'abord cette documentation
2. 🐛 Vérifiez la section Dépannage
3. 🔍 Recherchez dans les issues existantes
4. ❓ Créez une nouvelle issue avec :
   - Description détaillée du problème
   - Version de PyTorch (`python -c "import torch; print(torch.__version__)"`)
   - Système d'exploitation
   - Logs d'erreur complets

---

**Amusez-vous bien avec PyTorch et la reconnaissance de chiffres !**

> Ce projet démontre les capacités modernes de PyTorch pour le deep learning avec une interface utilisateur professionnelle. Parfait pour l'apprentissage, la recherche, ou comme base pour des projets plus avancés. 
