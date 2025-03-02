import os
import argparse
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models.mask_detector import MaskDetector, MaskDetectorTrainer
from utils.dataset import create_data_loaders
from preprocessing.image_processor import ImageProcessor

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Treinar detector de máscaras faciais')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Diretório contendo os dados')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Tamanho do batch')
    parser.add_argument('--epochs', type=int, default=10,
                      help='Número de épocas')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Taxa de aprendizado')
    parser.add_argument('--output_dir', type=str, default='results',
                      help='Diretório para salvar resultados')
    return parser.parse_args()

def plot_training_history(history, output_dir):
    """Plota e salva o histórico de treinamento."""
    plt.figure(figsize=(12, 4))
    
    # Plot de perda
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Treino')
    plt.plot(history['val_loss'], label='Validação')
    plt.title('Perda durante Treinamento')
    plt.xlabel('Época')
    plt.ylabel('Perda')
    plt.legend()
    
    # Plot de acurácia
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Validação')
    plt.title('Acurácia durante Treinamento')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()

def main():
    """Função principal de treinamento."""
    args = parse_args()
    
    # Cria diretório de saída se não existir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configuração do device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Usando device: {device}')
    
    # Carrega os dados
    train_loader, val_loader, test_loader = create_data_loaders(
        args.data_dir,
        batch_size=args.batch_size
    )
    print('Dados carregados com sucesso')
    
    # Inicializa o modelo e treinador
    model = MaskDetector(pretrained=True)
    trainer = MaskDetectorTrainer(model, device=device)
    
    # Histórico de treinamento
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Loop de treinamento
    print('Iniciando treinamento...')
    for epoch in range(args.epochs):
        # Treina por uma época
        train_loss = trainer.train_epoch(train_loader)
        
        # Valida o modelo
        val_acc, val_loss = trainer.validate(val_loader)
        
        # Atualiza histórico
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Época [{epoch+1}/{args.epochs}]')
        print(f'Perda Treino: {train_loss:.4f}')
        print(f'Perda Val: {val_loss:.4f}')
        print(f'Acurácia Val: {val_acc:.4f}')
        print('-' * 50)
    
    # Salva o modelo
    model_path = os.path.join(args.output_dir, 'mask_detector.pth')
    torch.save(model.state_dict(), model_path)
    print(f'Modelo salvo em {model_path}')
    
    # Plota histórico de treinamento
    plot_training_history(history, args.output_dir)
    print(f'Gráficos salvos em {args.output_dir}')
    
    # Avalia no conjunto de teste
    test_acc, test_loss = trainer.validate(test_loader)
    print(f'Resultados no conjunto de teste:')
    print(f'Acurácia: {test_acc:.4f}')
    print(f'Perda: {test_loss:.4f}')

if __name__ == '__main__':
    main()
