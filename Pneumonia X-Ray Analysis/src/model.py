"""
Módulo para definição da arquitetura do modelo CNN.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def criar_modelo(tamanho_imagem=(224, 224)):
    """
    Cria um modelo CNN para classificação de pneumonia.
    """
    modelo = Sequential([
        # Primeiro bloco convolucional
        Conv2D(32, (3, 3), activation='relu', input_shape=(*tamanho_imagem, 3)),
        MaxPooling2D(2, 2),
        
        # Segundo bloco convolucional
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        # Terceiro bloco convolucional
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        # Camadas densas
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    # Compilar o modelo
    modelo.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'Precision', 'Recall']
    )
    
    return modelo

if __name__ == "__main__":
    # Teste de criação do modelo
    try:
        modelo = criar_modelo()
        print("\nResumo do modelo:")
        modelo.summary()
        
    except Exception as e:
        print(f"Erro ao criar modelo: {str(e)}")
