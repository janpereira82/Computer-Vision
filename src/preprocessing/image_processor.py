import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

class ImageProcessor:
    """
    Classe responsável pelo pré-processamento das imagens para detecção de máscaras.
    """
    
    def __init__(self, image_size=(224, 224)):
        """
        Inicializa o processador de imagens.
        
        Args:
            image_size (tuple): Dimensões desejadas para redimensionar as imagens
        """
        self.image_size = image_size
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Transformações padrão para as imagens
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def detect_face(self, image):
        """
        Detecta faces em uma imagem.
        
        Args:
            image (numpy.ndarray): Imagem de entrada em formato BGR
            
        Returns:
            list: Lista de coordenadas (x, y, w, h) das faces detectadas
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces

    def preprocess_image(self, image_path):
        """
        Realiza o pré-processamento completo de uma imagem.
        
        Args:
            image_path (str): Caminho para a imagem
            
        Returns:
            torch.Tensor: Tensor da imagem processada
        """
        # Carrega a imagem
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Não foi possível carregar a imagem: {image_path}")
            
        # Detecta faces
        faces = self.detect_face(image)
        
        if len(faces) == 0:
            # Se nenhuma face for detectada, processa a imagem inteira
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            # Usa a primeira face detectada
            x, y, w, h = faces[0]
            face_img = image[y:y+h, x:x+w]
            pil_image = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        
        # Aplica as transformações
        return self.transform(pil_image)

    def augment_image(self, image):
        """
        Aplica técnicas de data augmentation em uma imagem.
        
        Args:
            image (PIL.Image): Imagem de entrada
            
        Returns:
            list: Lista de imagens aumentadas
        """
        augmented_images = []
        
        # Rotação
        for angle in [-15, 15]:
            rotated = transforms.functional.rotate(image, angle)
            augmented_images.append(rotated)
        
        # Espelhamento horizontal
        flipped = transforms.functional.hflip(image)
        augmented_images.append(flipped)
        
        # Ajuste de brilho e contraste
        brightness_contrast = transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2
        )(image)
        augmented_images.append(brightness_contrast)
        
        return augmented_images
