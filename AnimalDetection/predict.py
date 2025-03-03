import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import argparse

class AnimalDetector:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.classes = ['aranha', 'cobra', 'escorpiao', 'lagarta']
        self.image_size = (224, 224)
        
    def preprocess_image(self, image):
        """Preprocessa a imagem para predição"""
        # Converter BGR para RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Redimensionar
        image_resized = cv2.resize(image_rgb, self.image_size)
        
        # Normalizar e adicionar dimensão do batch
        image_normalized = image_resized / 255.0
        image_batch = np.expand_dims(image_normalized, axis=0)
        
        return image_batch
    
    def predict(self, image):
        """Realiza a predição em uma imagem"""
        # Preprocessar imagem
        processed_image = self.preprocess_image(image)
        
        # Fazer predição
        predictions = self.model.predict(processed_image)
        
        # Obter classe com maior probabilidade
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        
        return {
            'class': self.classes[class_idx],
            'confidence': confidence,
            'predictions': dict(zip(self.classes, predictions[0]))
        }
    
    def predict_and_draw(self, image):
        """Realiza a predição e desenha o resultado na imagem"""
        # Fazer predição
        result = self.predict(image)
        
        # Criar cópia da imagem
        output_image = image.copy()
        
        # Adicionar texto com a predição
        text = f"{result['class']}: {result['confidence']*100:.1f}%"
        cv2.putText(
            output_image,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        return output_image, result

def main():
    parser = argparse.ArgumentParser(description='Detectar animais em imagens')
    parser.add_argument('--image', required=True, help='Caminho para a imagem')
    parser.add_argument('--model', default='models/animal_detection_model.h5', help='Caminho para o modelo')
    parser.add_argument('--save', action='store_true', help='Salvar imagem com predição')
    args = parser.parse_args()
    
    # Carregar imagem
    image = cv2.imread(args.image)
    if image is None:
        print(f"Erro: Não foi possível carregar a imagem: {args.image}")
        return
    
    # Criar detector
    detector = AnimalDetector(args.model)
    
    # Fazer predição
    output_image, result = detector.predict_and_draw(image)
    
    # Mostrar resultados
    print("\nResultados da detecção:")
    print(f"Animal detectado: {result['class']}")
    print(f"Confiança: {result['confidence']*100:.1f}%")
    print("\nProbabilidades para cada classe:")
    for animal, prob in result['predictions'].items():
        print(f"{animal}: {prob*100:.1f}%")
    
    # Mostrar imagem
    cv2.imshow('Detecção', output_image)
    
    # Salvar imagem se solicitado
    if args.save:
        output_path = f"output_{args.image.split('/')[-1]}"
        cv2.imwrite(output_path, output_image)
        print(f"\nImagem com predição salva em: {output_path}")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
