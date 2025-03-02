"""
Script principal para treinamento do modelo.
"""

import os
import json
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from data_preparation import criar_geradores_dados
from model import criar_modelo

def treinar_modelo(diretorio_base, diretorio_modelos, epocas=50):
    """
    Treina o modelo de detecção de pneumonia.
    """
    # Criar diretórios se não existirem
    os.makedirs(diretorio_modelos, exist_ok=True)
    os.makedirs(os.path.join(diretorio_modelos, 'logs'), exist_ok=True)
    
    # Preparar geradores de dados
    gerador_treino, gerador_val, _ = criar_geradores_dados(diretorio_base)
    
    # Criar modelo
    modelo = criar_modelo()
    
    # Configurar callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join(diretorio_modelos, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        CSVLogger(
            os.path.join(diretorio_modelos, 'logs', 'training_history.csv'),
            append=True
        )
    ]
    
    # Treinar modelo
    print("\nIniciando treinamento...")
    historico = modelo.fit(
        gerador_treino,
        epochs=epocas,
        validation_data=gerador_val,
        callbacks=callbacks,
        verbose=1
    )
    
    # Salvar modelo final
    modelo.save(os.path.join(diretorio_modelos, 'modelo_final.h5'))
    
    # Salvar histórico de treinamento
    with open(os.path.join(diretorio_modelos, 'logs', 'historico.json'), 'w') as f:
        json.dump(historico.history, f)
    
    return historico

if __name__ == "__main__":
    # Configurar caminhos
    diretorio_base = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    diretorio_modelos = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    
    print(f"Diretório base: {diretorio_base}")
    print(f"Diretório modelos: {diretorio_modelos}")
    
    try:
        historico = treinar_modelo(diretorio_base, diretorio_modelos)
        print("\nTreinamento concluído com sucesso!")
        
    except Exception as e:
        print(f"Erro durante o treinamento: {str(e)}")
