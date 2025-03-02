from ultralytics import YOLO
from pathlib import Path
import logging
import json
import time
import cv2
import numpy as np
from tqdm import tqdm

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)

class AvaliadorModelo:
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.model_path = self.base_dir / 'models' / 'best.pt'
        self.test_dir = self.base_dir / 'data' / 'test' / 'images'
        self.valid_dir = self.base_dir / 'data' / 'valid' / 'images'
        self.results_dir = self.base_dir / 'results'
        self.results_dir.mkdir(exist_ok=True)

    def avaliar_conjunto_dados(self, conjunto='valid'):
        """
        Avalia o modelo em um conjunto de dados específico
        """
        try:
            # Carrega o modelo
            modelo = YOLO(str(self.model_path))
            diretorio = self.valid_dir if conjunto == 'valid' else self.test_dir
            
            logging.info(f"Iniciando avaliação no conjunto {conjunto}")
            
            # Métricas de tempo
            tempos_inferencia = []
            
            # Resultados por imagem
            resultados_detalhados = []
            
            # Executa validação
            for img_path in tqdm(list(diretorio.glob('*.jpg'))):
                inicio = time.time()
                resultados = modelo.predict(
                    source=str(img_path),
                    conf=0.25,
                    iou=0.45,
                    imgsz=416
                )
                fim = time.time()
                
                # Tempo de inferência
                tempo_inferencia = fim - inicio
                tempos_inferencia.append(tempo_inferencia)
                
                # Processa resultados
                for resultado in resultados:
                    boxes = resultado.boxes
                    confs = []
                    coords = []
                    
                    if len(boxes) > 0:
                        confs = boxes.conf.cpu().numpy().tolist()
                        coords = boxes.xyxy.cpu().numpy().tolist()
                    
                    resultados_detalhados.append({
                        'imagem': img_path.name,
                        'deteccoes': len(boxes),
                        'confiancas': confs,
                        'coordenadas': coords,
                        'tempo_inferencia': tempo_inferencia
                    })
            
            # Calcula métricas
            metricas = {
                'conjunto': conjunto,
                'total_imagens': len(resultados_detalhados),
                'tempo_medio_inferencia': np.mean(tempos_inferencia),
                'tempo_maximo_inferencia': np.max(tempos_inferencia),
                'tempo_minimo_inferencia': np.min(tempos_inferencia),
                'imagens_com_deteccao': sum(1 for r in resultados_detalhados if r['deteccoes'] > 0),
                'media_deteccoes_por_imagem': np.mean([r['deteccoes'] for r in resultados_detalhados]),
                'confianca_media': np.mean([conf for r in resultados_detalhados for conf in r['confiancas']]) if any(r['confiancas'] for r in resultados_detalhados) else 0
            }
            
            # Salva resultados
            resultados_file = self.results_dir / f'resultados_{conjunto}.json'
            with open(resultados_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'metricas': metricas,
                    'resultados_detalhados': resultados_detalhados
                }, f, indent=4)
            
            logging.info(f"Avaliação concluída. Resultados salvos em {resultados_file}")
            return metricas
            
        except Exception as e:
            logging.error(f"Erro durante a avaliação: {str(e)}")
            raise

    def avaliar_tempo_real(self, num_frames=100):
        """
        Avalia o desempenho do modelo em tempo real usando a webcam
        """
        try:
            modelo = YOLO(str(self.model_path))
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                logging.error("Não foi possível acessar a webcam")
                return
            
            tempos_inferencia = []
            logging.info(f"Iniciando teste de tempo real com {num_frames} frames")
            
            for _ in tqdm(range(num_frames)):
                ret, frame = cap.read()
                if not ret:
                    break
                
                inicio = time.time()
                modelo.predict(
                    source=frame,
                    conf=0.25,
                    iou=0.45,
                    imgsz=416
                )
                fim = time.time()
                
                tempos_inferencia.append(fim - inicio)
            
            cap.release()
            
            # Calcula métricas de tempo real
            metricas_tempo_real = {
                'fps_medio': 1.0 / np.mean(tempos_inferencia),
                'fps_minimo': 1.0 / np.max(tempos_inferencia),
                'fps_maximo': 1.0 / np.min(tempos_inferencia),
                'tempo_medio_inferencia': np.mean(tempos_inferencia),
                'tempo_total': sum(tempos_inferencia)
            }
            
            # Salva resultados
            resultados_file = self.results_dir / 'resultados_tempo_real.json'
            with open(resultados_file, 'w', encoding='utf-8') as f:
                json.dump(metricas_tempo_real, f, indent=4)
            
            logging.info(f"Avaliação em tempo real concluída. Resultados salvos em {resultados_file}")
            return metricas_tempo_real
            
        except Exception as e:
            logging.error(f"Erro durante avaliação em tempo real: {str(e)}")
            raise
        finally:
            if 'cap' in locals() and cap.isOpened():
                cap.release()

def main():
    """
    Função principal para executar a avaliação completa
    """
    try:
        avaliador = AvaliadorModelo()
        
        # Avalia no conjunto de validação
        logging.info("Iniciando avaliação no conjunto de validação")
        metricas_valid = avaliador.avaliar_conjunto_dados('valid')
        
        # Avalia no conjunto de teste
        logging.info("Iniciando avaliação no conjunto de teste")
        metricas_test = avaliador.avaliar_conjunto_dados('test')
        
        # Avalia em tempo real
        logging.info("Iniciando avaliação em tempo real")
        metricas_tempo_real = avaliador.avaliar_tempo_real(num_frames=50)
        
        # Exibe resumo dos resultados
        logging.info("\nResumo da Avaliação:")
        logging.info("\nMétricas do Conjunto de Validação:")
        for k, v in metricas_valid.items():
            logging.info(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
        
        logging.info("\nMétricas do Conjunto de Teste:")
        for k, v in metricas_test.items():
            logging.info(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
        
        logging.info("\nMétricas de Tempo Real:")
        for k, v in metricas_tempo_real.items():
            logging.info(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
        
    except Exception as e:
        logging.error(f"Erro durante o processo de avaliação: {str(e)}")
        raise

if __name__ == "__main__":
    main()
