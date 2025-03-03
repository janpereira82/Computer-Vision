from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def create_model(num_classes=4, input_shape=(224, 224, 3)):
    """
    Cria um modelo de CNN baseado no EfficientNetB0 com transfer learning
    """
    # Carregar o modelo base pré-treinado
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Congelar as camadas do modelo base
    base_model.trainable = False
    
    # Adicionar camadas personalizadas
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # Primeiro bloco denso
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Segundo bloco denso
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    # Terceiro bloco denso
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Criar o modelo final
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

def compile_model(model):
    """
    Compila o modelo com otimizador e função de perda apropriados
    """
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
def get_callbacks(checkpoint_path):
    """
    Define callbacks para treinamento
    """
    callbacks = [
        ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=7,
            min_lr=1e-6
        )
    ]
    return callbacks

def fine_tune_model(model):
    """
    Descongelar algumas camadas do modelo base para fine-tuning
    """
    # Descongelar os últimos blocos do EfficientNet
    for layer in model.layers[-30:]:
        layer.trainable = True
    
    # Recompilar com uma taxa de aprendizado menor
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
