import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class MaskDetector(nn.Module):
    """
    Modelo de CNN para detecção de máscaras faciais baseado em transfer learning.
    """
    
    def __init__(self, pretrained=True, freeze_features=True):
        """
        Inicializa o modelo de detecção de máscaras.
        
        Args:
            pretrained (bool): Se True, usa pesos pré-treinados do ImageNet
            freeze_features (bool): Se True, congela as camadas do backbone
        """
        super(MaskDetector, self).__init__()
        
        # Usa ResNet50 como backbone
        self.backbone = models.resnet50(pretrained=pretrained)
        
        if freeze_features:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Substitui a última camada por nossa própria classificação
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)  # 2 classes: com máscara e sem máscara
        )

    def forward(self, x):
        """
        Forward pass do modelo.
        
        Args:
            x (torch.Tensor): Batch de imagens de entrada
            
        Returns:
            torch.Tensor: Logits para cada classe
        """
        return self.backbone(x)

    def predict(self, x):
        """
        Realiza a predição para uma imagem.
        
        Args:
            x (torch.Tensor): Imagem de entrada
            
        Returns:
            tuple: (predição, probabilidades)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x.unsqueeze(0))
            probs = F.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1)
            return pred.item(), probs.squeeze().numpy()

class MaskDetectorTrainer:
    """
    Classe para treinar o modelo de detecção de máscaras.
    """
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Inicializa o treinador.
        
        Args:
            model (MaskDetector): Modelo a ser treinado
            device (str): Dispositivo para treinamento ('cuda' ou 'cpu')
        """
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Definição da função de perda e otimizador
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.001,
            weight_decay=1e-5
        )
        
    def train_epoch(self, train_loader):
        """
        Treina o modelo por uma época.
        
        Args:
            train_loader (DataLoader): Loader dos dados de treinamento
            
        Returns:
            float: Perda média da época
        """
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        """
        Valida o modelo.
        
        Args:
            val_loader (DataLoader): Loader dos dados de validação
            
        Returns:
            tuple: (acurácia média, perda média)
        """
        self.model.eval()
        val_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                val_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                
        val_loss /= len(val_loader)
        accuracy = correct / len(val_loader.dataset)
        
        return accuracy, val_loss
