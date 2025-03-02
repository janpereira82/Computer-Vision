import argparse
import torch
import cv2
import numpy as np
from PIL import Image
from models.mask_detector import MaskDetector
from preprocessing.image_processor import ImageProcessor

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Detectar máscaras em imagens')
    parser.add_argument('--image_path', type=str, required=True,
                      help='Caminho para a imagem')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Caminho para o modelo treinado')
    parser.add_argument('--output_path', type=str, default=None,
                      help='Caminho para salvar a imagem com detecções')
    return parser.parse_args()

def draw_prediction(image, face_coords, prediction, probability):
    """
    Desenha a predição na imagem.
    
    Args:
        image (numpy.ndarray): Imagem original
        face_coords (tuple): Coordenadas da face (x, y, w, h)
        prediction (int): Predição (0: com máscara, 1: sem máscara)
        probability (float): Probabilidade da predição
    
    Returns:
        numpy.ndarray: Imagem com as anotações
    """
    x, y, w, h = face_coords
    
    # Define cor baseado na predição
    color = (0, 255, 0) if prediction == 0 else (0, 0, 255)
    
    # Desenha retângulo ao redor da face
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    
    # Prepara texto
    label = 'Com Mascara' if prediction == 0 else 'Sem Mascara'
    text = f'{label}: {probability:.2f}%'
    
    # Adiciona texto
    cv2.putText(image, text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    
    return image

def main():
    """Função principal para predição."""
    args = parse_args()
    
    # Configuração do device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Carrega o modelo
    model = MaskDetector(pretrained=False)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Inicializa o processador de imagens
    processor = ImageProcessor()
    
    # Carrega e processa a imagem
    image = cv2.imread(args.image_path)
    if image is None:
        raise ValueError(f"Não foi possível carregar a imagem: {args.image_path}")
    
    # Detecta faces
    faces = processor.detect_face(image)
    
    # Para cada face detectada
    for face_coords in faces:
        # Extrai e pré-processa a face
        x, y, w, h = face_coords
        face_img = image[y:y+h, x:x+w]
        pil_image = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        
        # Aplica transformações e faz predição
        tensor_image = processor.transform(pil_image)
        tensor_image = tensor_image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(tensor_image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            probability = probabilities[0][prediction].item() * 100
        
        # Desenha predição na imagem
        image = draw_prediction(image, face_coords, prediction, probability)
    
    # Salva ou mostra a imagem
    if args.output_path:
        cv2.imwrite(args.output_path, image)
        print(f'Imagem com detecções salva em: {args.output_path}')
    else:
        cv2.imshow('Detecção de Máscaras', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
