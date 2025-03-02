"""
Script para avaliação do modelo treinado.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from data_preparation import criar_geradores_dados

def avaliar_modelo(diretorio_base, caminho_modelo):
    """
    Avalia o modelo no conjunto de teste.
    """
    # Carregar modelo
    modelo = load_model(caminho_modelo)
    
    # Preparar dados de teste
    _, _, gerador_teste = criar_geradores_dados(diretorio_base)
    
    # Avaliar modelo
    print("\nAvaliando modelo no conjunto de teste...")
    metricas = modelo.evaluate(gerador_teste)
    print("\nMétricas no conjunto de teste:")
    for nome, valor in zip(modelo.metrics_names, metricas):
        print(f"{nome}: {valor:.4f}")
    
    # Gerar predições
    print("\nGerando predições...")
    predicoes = modelo.predict(gerador_teste)
    predicoes_classes = (predicoes > 0.5).astype(int)
    
    # Relatório de classificação
    print("\nGerando relatório de classificação...")
    print("\nRelatório detalhado:")
    print(classification_report(gerador_teste.classes, predicoes_classes,
                              target_names=['Normal', 'Pneumonia']))
    
    # Matriz de confusão
    print("\nGerando matriz de confusão...")
    matriz_conf = confusion_matrix(gerador_teste.classes, predicoes_classes)
    
    # Plotar matriz de confusão
    plt.figure(figsize=(8, 6))
    sns.heatmap(matriz_conf, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Pneumonia'],
                yticklabels=['Normal', 'Pneumonia'])
    plt.title('Matriz de Confusão')
    plt.ylabel('Real')
    plt.xlabel('Predito')
    
    # Salvar matriz de confusão
    diretorio_resultados = os.path.join(os.path.dirname(caminho_modelo), 'resultados')
    os.makedirs(diretorio_resultados, exist_ok=True)
    plt.savefig(os.path.join(diretorio_resultados, 'confusion_matrix.png'))
    plt.close()

if __name__ == "__main__":
    # Configurar caminhos
    diretorio_base = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    caminho_modelo = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                 'models', 'best_model.h5')
    
    print(f"Diretório base: {diretorio_base}")
    print(f"Caminho do modelo: {caminho_modelo}")
    
    try:
        avaliar_modelo(diretorio_base, caminho_modelo)
        print("\nAvaliação concluída com sucesso!")
        
    except Exception as e:
        print(f"Erro durante a avaliação: {str(e)}")
