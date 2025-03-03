import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataPreprocessor:
    def __init__(self, data_dir, image_size=(224, 224)):
        self.data_dir = data_dir
        self.image_size = image_size
        self.classes = ['aranha', 'cobra', 'escorpiao', 'lagarta']
        self.datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
    def load_and_preprocess_data(self):
        """Carrega e preprocessa as imagens do diretório de dados."""
        images = []
        labels = []
        
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_path):
                continue
                
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                try:
                    image = cv2.imread(image_path)
                    if image is None:
                        continue
                    
                    # Converter BGR para RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Redimensionar
                    image = cv2.resize(image, self.image_size)
                    
                    # Normalizar
                    image = image / 255.0
                    
                    images.append(image)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"Erro ao processar {image_path}: {str(e)}")
        
        return np.array(images), np.array(labels)
    
    def prepare_data(self, test_size=0.2, validation_split=0.2):
        """Prepara os dados para treinamento, validação e teste."""
        X, y = self.load_and_preprocess_data()
        
        # Converter labels para one-hot encoding
        y = to_categorical(y, num_classes=len(self.classes))
        
        # Dividir em conjuntos de treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Dividir o conjunto de treino em treino e validação
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, 
            test_size=validation_split, 
            random_state=42,
            stratify=y_train
        )
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
    
    def get_train_generator(self, X_train, y_train, batch_size=32):
        """Retorna um gerador de dados aumentados para treinamento"""
        return self.datagen.flow(X_train, y_train, batch_size=batch_size)
