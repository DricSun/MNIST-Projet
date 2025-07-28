import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image
import os

class MNISTDataLoader:
    def __init__(self, batch_size=32, validation_split=0.1):
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Transformations pour normaliser les données
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # Mean et std du dataset MNIST
        ])
        
        # Transformations pour l'augmentation des données (optionnel)
        self.transform_augmented = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    def load_data(self, use_augmentation=False):
        """Charge les données MNIST depuis torchvision"""
        print("Chargement des données MNIST...")
        
        # Choisir les transformations
        transform = self.transform_augmented if use_augmentation else self.transform
        
        # Charger les datasets
        train_dataset_full = torchvision.datasets.MNIST(
            root='./data', 
            train=True, 
            download=True, 
            transform=transform
        )
        
        self.test_dataset = torchvision.datasets.MNIST(
            root='./data', 
            train=False, 
            download=True, 
            transform=self.transform  # Pas d'augmentation pour le test
        )
        
        # Division train/validation
        train_size = int((1 - self.validation_split) * len(train_dataset_full))
        val_size = len(train_dataset_full) - train_size
        
        self.train_dataset, self.val_dataset = random_split(
            train_dataset_full, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # Pour la reproductibilité
        )
        
        # Créer les DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        print(f"Données chargées:")
        print(f"  - Entraînement: {len(self.train_dataset)} échantillons")
        print(f"  - Validation: {len(self.val_dataset)} échantillons")
        print(f"  - Test: {len(self.test_dataset)} échantillons")
        print(f"  - Batch size: {self.batch_size}")
        
        return self.train_loader, self.val_loader, self.test_loader
    
    def get_data_stats(self):
        """Affiche les statistiques des données"""
        if self.train_loader is None:
            print("Veuillez d'abord charger les données avec load_data()")
            return
        
        print("\n=== Statistiques des données ===")
        print(f"Shape des images: {(1, 28, 28)}")
        print(f"Type de données: torch.FloatTensor")
        
        # Calculer les statistiques sur un batch
        data_iter = iter(self.train_loader)
        images, labels = next(data_iter)
        
        print(f"Valeurs min/max: {images.min():.3f} / {images.max():.3f}")
        print(f"Moyenne: {images.mean():.3f}")
        print(f"Écart-type: {images.std():.3f}")
        
        # Distribution des classes
        all_labels = []
        for _, labels in self.train_loader:
            all_labels.extend(labels.numpy())
        
        unique, counts = np.unique(all_labels, return_counts=True)
        print(f"\nDistribution des classes (entraînement):")
        for digit, count in zip(unique, counts):
            print(f"  Chiffre {digit}: {count} échantillons")
    
    def visualize_samples(self, num_samples=10, save_path=None):
        """Visualise quelques échantillons des données"""
        if self.train_loader is None:
            print("Veuillez d'abord charger les données avec load_data()")
            return
        
        # Récupérer un batch de données
        data_iter = iter(self.train_loader)
        images, labels = next(data_iter)
        
        # Sélectionner les premiers échantillons
        images = images[:num_samples]
        labels = labels[:num_samples]
        
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        axes = axes.ravel()
        
        for i in range(num_samples):
            # Dénormaliser l'image pour l'affichage
            img = images[i].squeeze().numpy()
            img = img * 0.3081 + 0.1307  # Inverse de la normalisation
            img = np.clip(img, 0, 1)
            
            label = labels[i].item()
            
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'Label: {label}')
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Échantillons sauvegardés dans {save_path}")
        
        plt.show()
    
    def preprocess_image(self, image_path):
        """Préprocesse une image externe pour la prédiction"""
        
        # Charger l'image
        if isinstance(image_path, str):
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            # Si c'est déjà un array numpy
            img = image_path
        
        # Redimensionner à 28x28
        img = cv2.resize(img, (28, 28))
        
        # Inverser les couleurs si nécessaire (fond blanc -> fond noir)
        if np.mean(img) > 127:
            img = 255 - img
        
        # Normaliser (0-255 -> 0-1)
        img = img.astype('float32') / 255.0
        
        # Appliquer la même normalisation que pour l'entraînement
        img = (img - 0.1307) / 0.3081
        
        # Convertir en tensor PyTorch avec la bonne forme (1, 1, 28, 28)
        img_tensor = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0)
        
        return img_tensor
    
    def create_custom_dataset(self, images, labels):
        """Crée un dataset personnalisé à partir d'images et de labels"""
        
        class CustomMNISTDataset(Dataset):
            def __init__(self, images, labels, transform=None):
                self.images = images
                self.labels = labels
                self.transform = transform
            
            def __len__(self):
                return len(self.images)
            
            def __getitem__(self, idx):
                image = self.images[idx]
                label = self.labels[idx]
                
                if self.transform:
                    if isinstance(image, np.ndarray):
                        image = Image.fromarray(image.astype('uint8'))
                    image = self.transform(image)
                
                return image, label
        
        return CustomMNISTDataset(images, labels, self.transform)
    
    def get_sample_batch(self, dataset='train'):
        """Récupère un batch d'échantillons pour les tests"""
        if dataset == 'train' and self.train_loader:
            data_iter = iter(self.train_loader)
        elif dataset == 'val' and self.val_loader:
            data_iter = iter(self.val_loader)
        elif dataset == 'test' and self.test_loader:
            data_iter = iter(self.test_loader)
        else:
            print(f"Dataset '{dataset}' non disponible ou non chargé")
            return None, None
        
        images, labels = next(data_iter)
        return images, labels
    
    def analyze_data_distribution(self):
        """Analyse la distribution des données dans tous les ensembles"""
        if not all([self.train_loader, self.val_loader, self.test_loader]):
            print("Veuillez d'abord charger toutes les données")
            return
        
        sets = {
            'Train': self.train_loader,
            'Validation': self.val_loader,
            'Test': self.test_loader
        }
        
        print("\n=== Analyse de la distribution des classes ===")
        
        for set_name, loader in sets.items():
            all_labels = []
            for _, labels in loader:
                all_labels.extend(labels.numpy())
            
            unique, counts = np.unique(all_labels, return_counts=True)
            
            print(f"\n{set_name}:")
            for digit, count in zip(unique, counts):
                percentage = count / len(all_labels) * 100
                print(f"  Chiffre {digit}: {count:4d} ({percentage:5.1f}%)")
    
    def save_processed_data(self, filepath):
        """Sauvegarde les données préprocessées"""
        if not all([self.train_dataset, self.val_dataset, self.test_dataset]):
            print("Aucune donnée à sauvegarder")
            return
        
        torch.save({
            'train_dataset': self.train_dataset,
            'val_dataset': self.val_dataset,
            'test_dataset': self.test_dataset,
            'batch_size': self.batch_size,
            'validation_split': self.validation_split
        }, filepath)
        
        print(f"Données sauvegardées dans {filepath}")
    
    def load_processed_data(self, filepath):
        """Charge des données préprocessées"""
        if not os.path.exists(filepath):
            print(f"Le fichier {filepath} n'existe pas")
            return False
        
        data = torch.load(filepath)
        
        self.train_dataset = data['train_dataset']
        self.val_dataset = data['val_dataset'] 
        self.test_dataset = data['test_dataset']
        self.batch_size = data.get('batch_size', 32)
        self.validation_split = data.get('validation_split', 0.1)
        
        # Recréer les DataLoaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        
        print(f"Données chargées depuis {filepath}")
        return True 