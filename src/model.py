import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os

class MNISTNet(nn.Module):
    """Réseau CNN pour la classification MNIST avec PyTorch"""
    
    def __init__(self):
        super(MNISTNet, self).__init__()
        
        # Couches de convolution
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Couches de pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Couches fully connected
        self.fc1 = nn.Linear(64 * 7 * 7, 64)  # 64 canaux * 7 * 7 après pooling
        self.fc2 = nn.Linear(64, 10)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Première couche conv + activation + pooling
        x = self.pool(F.relu(self.conv1(x)))
        
        # Deuxième couche conv + activation + pooling
        x = self.pool(F.relu(self.conv2(x)))
        
        # Troisième couche conv + activation
        x = F.relu(self.conv3(x))
        
        # Aplatir pour les couches FC
        x = x.view(-1, 64 * 7 * 7)
        
        # Première couche FC + dropout
        x = self.dropout(F.relu(self.fc1(x)))
        
        # Couche de sortie
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)

class MNISTModel:
    """Wrapper pour l'entraînement et l'évaluation du modèle MNIST"""
    
    def __init__(self, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MNISTNet().to(self.device)
        self.optimizer = None
        self.scheduler = None
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
        print(f"Utilisation du device: {self.device}")
    
    def get_model_summary(self):
        """Affiche un résumé du modèle"""
        print("\n=== Architecture du Modèle CNN ===")
        print(self.model)
        
        # Calculer le nombre de paramètres
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"\nNombre total de paramètres: {total_params:,}")
        print(f"Paramètres entraînables: {trainable_params:,}")
        
        return self.model
    
    def train_epoch(self, train_loader):
        """Entraîne le modèle pour une époque"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Reset des gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = F.nll_loss(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistiques
            running_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}')
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader):
        """Évalue le modèle sur les données de validation"""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        val_loss /= total
        val_acc = correct / total
        
        return val_loss, val_acc
    
    def train(self, train_loader, val_loader, epochs=20, learning_rate=0.001):
        """Entraîne le modèle complet"""
        
        # Configuration de l'optimiseur et du scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', patience=3, factor=0.5
        )
        
        print(f"\n=== Début de l'entraînement ({epochs} époques) ===")
        
        best_val_acc = 0.0
        patience_counter = 0
        early_stop_patience = 5
        
        for epoch in range(epochs):
            print(f"\nÉpoque {epoch+1}/{epochs}")
            print("-" * 40)
            
            # Entraînement
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Scheduler step
            self.scheduler.step(val_loss)
            
            # Sauvegarder les métriques
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Sauvegarder le meilleur modèle
                self.save_checkpoint('best_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping activé après {epoch+1} époques")
                print(f"Meilleure précision de validation: {best_val_acc:.4f}")
                break
        
        print(f"\n✅ Entraînement terminé!")
        print(f"Meilleure précision de validation: {best_val_acc:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }
    
    def evaluate(self, test_loader):
        """Évalue le modèle sur les données de test"""
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                all_preds.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy())
        
        test_loss /= total
        test_acc = correct / total
        
        return test_loss, test_acc, all_preds, all_targets
    
    def predict(self, image):
        """Prédit la classe d'une image"""
        print(f"🔍 DEBUG Model.predict - Input type: {type(image)}")
        if hasattr(image, 'shape'):
            print(f"🔍 DEBUG Model.predict - Input shape: {image.shape}")
        if hasattr(image, 'min') and hasattr(image, 'max'):
            print(f"🔍 DEBUG Model.predict - Input min/max: {image.min():.3f}/{image.max():.3f}")
        
        self.model.eval()
        
        # Convertir l'image en tensor PyTorch
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:  # Image 2D
                image = image.reshape(1, 1, 28, 28)
            elif len(image.shape) == 3:  # Image 3D
                image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
            elif len(image.shape) == 4:  # Déjà en format batch
                pass
            
            image_tensor = torch.FloatTensor(image).to(self.device)
        else:
            image_tensor = image.to(self.device)
        
        print(f"🔍 DEBUG Model.predict - Tensor shape: {image_tensor.shape}")
        print(f"🔍 DEBUG Model.predict - Tensor min/max: {image_tensor.min():.3f}/{image_tensor.max():.3f}")
        print(f"🔍 DEBUG Model.predict - Device: {self.device}")
        
        with torch.no_grad():
            output = self.model(image_tensor)
            print(f"🔍 DEBUG Model.predict - Raw output: {output[0][:5]}...")  # Premiers 5 logits
            
            probabilities = F.softmax(output, dim=1)
            prediction = output.argmax(dim=1).item()
            confidence = probabilities.max().item()
            
            print(f"🔍 DEBUG Model.predict - Probabilities: {probabilities[0]}")
            print(f"🔍 DEBUG Model.predict - Final prediction: {prediction}, confidence: {confidence:.3f}")
        
        return prediction, confidence
    
    def save_model(self, filepath):
        """Sauvegarde le modèle"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }, filepath)
        print(f"Modèle sauvegardé dans {filepath}")
    
    def load_model(self, filepath):
        """Charge un modèle sauvegardé"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Le fichier {filepath} n'existe pas")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Charger l'historique si disponible
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
            self.train_accuracies = checkpoint['train_accuracies']
            self.val_losses = checkpoint['val_losses']
            self.val_accuracies = checkpoint['val_accuracies']
        
        self.model.eval()
        print(f"Modèle chargé depuis {filepath}")
        return self.model
    
    def save_checkpoint(self, filename):
        """Sauvegarde un checkpoint durant l'entraînement"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }, filename) 