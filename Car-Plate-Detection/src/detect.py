import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
import os
from pathlib import Path

class DetectorPlacas:
    def __init__(self):
        """
        Inicializa o detector de placas com os modelos necessários
        """
        # Inicializa o leitor OCR para português
        self.reader = easyocr.Reader(['pt'])
        
        # Carrega o modelo YOLO (será treinado posteriormente)
        self.modelo_yolo = YOLO('models/best.pt')
        
        # Define o diretório base do projeto
        self.base_dir = Path(__file__).parent.parent

    def preprocessar_imagem(self, imagem):
        """
        Realiza o pré-processamento da imagem
        
        Args:
            imagem: Imagem de entrada em formato BGR
            
        Returns:
            Imagem pré-processada
        """
        # Converte para escala de cinza
        cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        
        # Aplica equalização de histograma para melhorar o contraste
        equalizada = cv2.equalizeHist(cinza)
        
        # Aplica suavização para reduzir ruído
        suavizada = cv2.GaussianBlur(equalizada, (5, 5), 0)
        
        return suavizada

    def detectar_placa_yolo(self, imagem):
        """
        Detecta a localização da placa usando YOLO
        
        Args:
            imagem: Imagem de entrada
            
        Returns:
            Lista de coordenadas das placas detectadas
        """
        resultados = self.modelo_yolo(imagem)
        return resultados[0].boxes.xyxy.cpu().numpy()

    def reconhecer_texto(self, imagem, bbox):
        """
        Realiza o reconhecimento do texto da placa
        
        Args:
            imagem: Imagem original
            bbox: Coordenadas da placa [x1, y1, x2, y2]
            
        Returns:
            Texto reconhecido da placa
        """
        # Extrai a região da placa
        x1, y1, x2, y2 = map(int, bbox)
        roi = imagem[y1:y2, x1:x2]
        
        # Realiza o reconhecimento de texto
        resultados = self.reader.readtext(roi)
        
        if resultados:
            # Retorna o texto com maior confiança
            return max(resultados, key=lambda x: x[2])[1]
        return None

    def processar_imagem(self, caminho_imagem):
        """
        Processa uma única imagem para detectar e reconhecer a placa
        
        Args:
            caminho_imagem: Caminho para a imagem
            
        Returns:
            Tupla (imagem processada, texto da placa)
        """
        # Carrega a imagem
        imagem = cv2.imread(caminho_imagem)
        if imagem is None:
            raise ValueError(f"Não foi possível carregar a imagem: {caminho_imagem}")
        
        # Pré-processa a imagem
        imagem_proc = self.preprocessar_imagem(imagem)
        
        # Detecta a placa
        bboxes = self.detectar_placa_yolo(imagem)
        
        resultados = []
        imagem_anotada = imagem.copy()
        
        for bbox in bboxes:
            # Reconhece o texto da placa
            texto = self.reconhecer_texto(imagem_proc, bbox)
            
            if texto:
                # Desenha o retângulo e o texto na imagem
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(imagem_anotada, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(imagem_anotada, texto, (x1, y1-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                resultados.append(texto)
        
        return imagem_anotada, resultados

def main():
    """
    Função principal para demonstração do detector
    """
    detector = DetectorPlacas()
    
    # Define os diretórios
    dir_teste = Path(detector.base_dir) / 'data' / 'test'
    dir_resultados = Path(detector.base_dir) / 'results'
    
    # Cria o diretório de resultados se não existir
    dir_resultados.mkdir(exist_ok=True)
    
    # Processa todas as imagens no diretório de teste
    for imagem_path in dir_teste.glob('*.jpg'):
        try:
            # Processa a imagem
            imagem_anotada, placas = detector.processar_imagem(str(imagem_path))
            
            # Salva a imagem processada
            nome_saida = dir_resultados / f'resultado_{imagem_path.name}'
            cv2.imwrite(str(nome_saida), imagem_anotada)
            
            # Imprime os resultados
            print(f"Imagem: {imagem_path.name}")
            print(f"Placas detectadas: {placas}")
            print("-" * 50)
            
        except Exception as e:
            print(f"Erro ao processar {imagem_path}: {str(e)}")

if __name__ == "__main__":
    main()
