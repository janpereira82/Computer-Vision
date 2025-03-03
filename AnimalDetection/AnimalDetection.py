import cv2 as cv
import numpy as np
import tensorflow as tf
import os

# Lista de nomes das classes
name = ['aranha', 'cobra', 'escorpião', 'lagarta']

# Carregar o modelo
model = tf.keras.models.load_model('models/ANIMALS.h5')

def preprocess_image(image_path):
    # Ler a imagem
    img = cv.imread(image_path)
    # Verificar se a imagem foi carregada corretamente
    if img is None:
        raise ValueError(f"Imagem não encontrada ou não pode ser carregada: {image_path}")
    # Preprocessar a imagem (redimensionar e normalizar)
    normalized_image = np.expand_dims(cv.resize(cv.cvtColor(img, cv.COLOR_BGR2RGB), (224, 224)), axis=0)
    return normalized_image

def main():
    # Solicitar o caminho da imagem ao usuário
    image_path = input("Digite o caminho da imagem (png, jpg, jpeg): ")
    
    # Verificar se o caminho da imagem existe
    if not os.path.isfile(image_path):
        print(f"Arquivo não encontrado: {image_path}")
        return
    
    # Preprocessar a imagem
    normalized_image = preprocess_image(image_path)
    
    # Fazer a predição
    predictions = model.predict(normalized_image)
    
    # Exibir os resultados
    predicted_class = name[np.argmax(predictions)]
    print(f"Resultado da predição: {predicted_class}")
    print("Predições para cada classe:")
    for i, cls in enumerate(name):
        print(f"{cls}: {predictions[0][i] * 100:.2f}%")

if __name__ == "__main__":
    main()