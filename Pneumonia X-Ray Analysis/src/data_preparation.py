"""
Módulo para preparação e pré-processamento dos dados de raio-X.
"""

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def criar_geradores_dados(diretorio_base, tamanho_batch=32, tamanho_imagem=(224, 224)):
    """
    Cria geradores de dados para treinamento, validação e teste.
    """
    # Configuração para aumento de dados (apenas para treino)
    gerador_treino = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    # Configuração para teste (apenas rescale)
    gerador_teste = ImageDataGenerator(rescale=1./255)

    # Criar geradores
    print("\nCarregando dados de treinamento...")
    gerador_treino_flow = gerador_treino.flow_from_directory(
        os.path.join(diretorio_base, 'train'),
        target_size=tamanho_imagem,
        batch_size=tamanho_batch,
        class_mode='binary',
        subset='training'
    )
    
    print("\nCarregando dados de validação...")
    gerador_val_flow = gerador_treino.flow_from_directory(
        os.path.join(diretorio_base, 'train'),
        target_size=tamanho_imagem,
        batch_size=tamanho_batch,
        class_mode='binary',
        subset='validation'
    )
    
    print("\nCarregando dados de teste...")
    gerador_teste_flow = gerador_teste.flow_from_directory(
        os.path.join(diretorio_base, 'test'),
        target_size=tamanho_imagem,
        batch_size=tamanho_batch,
        class_mode='binary',
        shuffle=False
    )
    
    return gerador_treino_flow, gerador_val_flow, gerador_teste_flow

if __name__ == "__main__":
    # Teste dos geradores
    diretorio_base = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    print(f"Diretório base: {diretorio_base}")
    
    try:
        gerador_treino, gerador_val, gerador_teste = criar_geradores_dados(diretorio_base)
        
        print("\nInformações dos geradores:")
        print(f"Amostras de treino: {gerador_treino.samples}")
        print(f"Classes de treino: {gerador_treino.class_indices}")
        print(f"Amostras de validação: {gerador_val.samples}")
        print(f"Amostras de teste: {gerador_teste.samples}")
        
        # Testar carregamento de um batch
        x, y = next(gerador_treino)
        print(f"\nFormato do batch de treino: {x.shape}")
        print(f"Formato das labels: {y.shape}")
        
    except Exception as e:
        print(f"Erro ao criar geradores: {str(e)}")
