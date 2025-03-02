import os
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms
import pandas as pd
import numpy as np

class MaskDataset(Dataset):
    """
    Dataset personalizado para as imagens de máscaras faciais.
    """
    
    def __init__(self, root_dir, transform=None, split='train'):
        """
        Inicializa o dataset.
        
        Args:
            root_dir (str): Diretório raiz contendo as imagens
            transform (callable, optional): Transformações a serem aplicadas
            split (str): 'train', 'val' ou 'test'
        """
        self.root_dir = os.path.join(root_dir, split.capitalize())
        self.transform = transform
        self.classes = ['WithMask', 'WithoutMask']
        
        # Lista todas as imagens e seus rótulos
        self.images = []
        self.labels = []
        
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.exists(class_dir):
                continue
                
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(class_idx)
        
        # Converte para numpy arrays
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        
        print(f'Carregadas {len(self.images)} imagens para {split}')
        print(f'Distribuição de classes: {np.bincount(self.labels)}')
    
    def __len__(self):
        """Retorna o tamanho total do dataset."""
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Retorna um item do dataset.
        
        Args:
            idx (int): Índice do item
            
        Returns:
            tuple: (imagem, rótulo)
        """
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

    def get_class_weights(self):
        """
        Calcula os pesos das classes para lidar com desbalanceamento.
        
        Returns:
            torch.Tensor: Pesos das classes
        """
        class_counts = np.bincount(self.labels)
        total = len(self.labels)
        weights = torch.FloatTensor(total / (len(class_counts) * class_counts))
        return weights

def create_data_loaders(root_dir, batch_size=32, num_workers=4):
    """
    Cria data loaders para treino, validação e teste.
    
    Args:
        root_dir (str): Diretório raiz contendo as imagens
        batch_size (int): Tamanho do batch
        num_workers (int): Número de workers para carregamento paralelo
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Transformações padrão
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Cria os datasets
    train_dataset = MaskDataset(root_dir, transform=train_transform, split='train')
    val_dataset = MaskDataset(root_dir, transform=val_test_transform, split='validation')
    test_dataset = MaskDataset(root_dir, transform=val_test_transform, split='test')
    
    # Cria os data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader
