import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Otimizações para CPU
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)

# Configurações
IMG_SIZE = 224
BATCH_SIZE = 32  # Aumentado para melhor utilização da CPU
EPOCHS = 100     # Aumentado para permitir mais tempo de treinamento
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "dataset_balanceado")  # Usando dataset balanceado
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "best_model.h5")
os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

class CustomProgressCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\nIniciando época {epoch+1}/{self.params['epochs']}")
    
    def on_epoch_end(self, epoch, logs=None):
        print(f"\nÉpoca {epoch+1}/{self.params['epochs']}")
        print(f"Loss: {logs['loss']:.4f}")
        print(f"Acurácia: {logs['accuracy']:.4f}")
        print(f"Val Loss: {logs['val_loss']:.4f}")
        print(f"Val Acurácia: {logs['val_accuracy']:.4f}")

def create_model(num_classes):
    """Cria o modelo usando MobileNetV2 (melhor para CPU)"""
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Fine-tuning em mais camadas
    for layer in base_model.layers[:-50]:  # Congelando menos camadas
        layer.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def plot_training_history(history):
    """Plota o histórico de treinamento"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Treino')
    plt.plot(history.history['val_accuracy'], label='Validação')
    plt.title('Acurácia do Modelo')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Treino')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title('Loss do Modelo')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'training_history.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plota a matriz de confusão"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Matriz de Confusão')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'confusion_matrix.png'))
    plt.close()

def main():
    # Preparar os dados com augmentation mais agressivo
    print("\nCarregando e preparando os dados...")
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,          # Aumentado
        width_shift_range=0.2,      # Aumentado
        height_shift_range=0.2,     # Aumentado
        zoom_range=0.2,             # Aumentado
        horizontal_flip=True,
        vertical_flip=True,         # Adicionado
        brightness_range=[0.8,1.2], # Adicionado
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    validation_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    # Criar e compilar o modelo
    print("\nCriando o modelo...")
    num_classes = len(train_generator.class_indices)
    model = create_model(num_classes)
    
    # Compilar com learning rate menor
    model.compile(
        optimizer=Adam(learning_rate=0.0005),  # Learning rate reduzido
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Resumo do modelo
    print("\nArquitetura do modelo:")
    model.summary()

    # Callbacks ajustados
    callbacks = [
        ModelCheckpoint(
            CHECKPOINT_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,           # Aumentado
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=8,           # Aumentado
            min_lr=1e-6,
            verbose=1
        ),
        CustomProgressCallback()
    ]

    # Treinar o modelo
    print("\nIniciando treinamento...")
    print(f"Usando imagens de tamanho {IMG_SIZE}x{IMG_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Número máximo de épocas: {EPOCHS}")
    print(f"Classes encontradas: {list(train_generator.class_indices.keys())}")
    print(f"Número de imagens de treino: {train_generator.n}")
    print(f"Número de imagens de validação: {validation_generator.n}")
    
    # Calcular steps por época
    steps_per_epoch = train_generator.n // BATCH_SIZE
    validation_steps = validation_generator.n // BATCH_SIZE
    
    if train_generator.n % BATCH_SIZE > 0:
        steps_per_epoch += 1
    if validation_generator.n % BATCH_SIZE > 0:
        validation_steps += 1
    
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )

    # Avaliar o modelo
    print("\nAvaliando o modelo...")
    model.load_weights(CHECKPOINT_PATH)
    
    print("\nGerando predições no conjunto de validação...")
    validation_generator.reset()
    y_pred = model.predict(
        validation_generator,
        steps=validation_steps,
        verbose=1
    )
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = validation_generator.classes[:len(y_pred_classes)]  # Ajustar para o número correto de predições
    
    # Gerar relatório de classificação
    class_names = list(train_generator.class_indices.keys())
    report = classification_report(y_true, y_pred_classes, target_names=class_names)
    print("\nRelatório de Classificação:")
    print(report)
    
    # Salvar o relatório
    with open(os.path.join(BASE_DIR, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Plotar gráficos
    plot_training_history(history)
    plot_confusion_matrix(y_true, y_pred_classes, class_names)
    
    print("\nTreinamento concluído! Os resultados foram salvos em:")
    print("- Modelo: checkpoints/best_model.h5")
    print("- Histórico de treinamento: training_history.png")
    print("- Matriz de confusão: confusion_matrix.png")
    print("- Relatório de classificação: classification_report.txt")

if __name__ == "__main__":
    main()
