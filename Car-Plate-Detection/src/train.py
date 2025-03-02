from ultralytics import YOLO
from pathlib import Path
import yaml
import logging
import sys

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class TreinadorYOLO:
    def __init__(self):
        """
        Inicializa o treinador do modelo YOLO
        """
        self.base_dir = Path(__file__).parent.parent
        self.config_path = self.base_dir / 'data' / 'dataset.yaml'
        
    def criar_arquivo_config(self):
        """
        Cria o arquivo de configuração YAML para treinamento
        """
        config = {
            'path': str(self.base_dir / 'data'),
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images',
            'names': {
                0: 'placa'
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)
        
        logging.info(f"Arquivo de configuração criado em {self.config_path}")

    def treinar_modelo(self, epochs=30, batch_size=16, imgsz=416):
        """
        Treina o modelo YOLO com configurações otimizadas
        
        Args:
            epochs: Número de épocas de treinamento
            batch_size: Tamanho do batch
            imgsz: Tamanho das imagens de entrada
        """
        try:
            self.criar_arquivo_config()
            
            # Inicializa o modelo YOLO
            modelo = YOLO('yolov8n.pt')
            
            logging.info("Iniciando treinamento do modelo...")
            
            # Treina o modelo com configurações otimizadas
            resultados = modelo.train(
                data=str(self.config_path),
                epochs=epochs,
                batch=batch_size,
                imgsz=imgsz,
                patience=10,           # Early stopping mais agressivo
                save=True,
                device='cpu',
                workers=4,             # Número de workers para carregamento de dados
                mosaic=0.5,           # Reduz augmentation para acelerar
                mixup=0.3,            # Reduz mixup para acelerar
                degrees=10.0,         # Reduz rotação máxima
                scale=0.4,            # Reduz escala de augmentation
                optimizer='AdamW',     # Otimizador mais eficiente
                lr0=0.001,            # Learning rate inicial maior
                lrf=0.01,             # Learning rate final maior
                warmup_epochs=1.0,    # Warmup mais curto
                cos_lr=True,          # Scheduler cosseno
                overlap_mask=False,    # Desativa overlap mask para acelerar
                val=True,             # Mantém validação para monitorar qualidade
                plots=False           # Desativa plots para economizar tempo
            )
            
            # Salva o modelo final
            modelo_final = self.base_dir / 'models' / 'best.pt'
            modelo.save(str(modelo_final))
            
            logging.info(f"Treinamento concluído. Modelo salvo em {modelo_final}")
            
            # Avalia o modelo
            logging.info("Avaliando modelo no conjunto de teste...")
            resultados_teste = modelo.val()
            
            metricas = {
                'mAP50': resultados_teste.box.map50,
                'mAP50-95': resultados_teste.box.map,
                'Precisão': resultados_teste.box.precision,
                'Recall': resultados_teste.box.recall
            }
            
            logging.info("Métricas de avaliação:")
            for metrica, valor in metricas.items():
                logging.info(f"{metrica}: {valor:.4f}")
            
        except Exception as e:
            logging.error(f"Erro durante o treinamento: {str(e)}")
            raise

def main():
    """
    Função principal para executar o treinamento
    """
    try:
        treinador = TreinadorYOLO()
        
        # Configurações otimizadas para treinamento rápido
        config_treinamento = {
            'epochs': 30,             # Reduzido de 100 para 30
            'batch_size': 16,         # Mantido em 16 para melhor convergência
            'imgsz': 416             # Reduzido de 640 para 416 para acelerar
        }
        
        logging.info("Iniciando processo de treinamento com configurações:")
        for param, valor in config_treinamento.items():
            logging.info(f"{param}: {valor}")
        
        treinador.treinar_modelo(**config_treinamento)
        
        logging.info("Processo de treinamento concluído com sucesso!")
        
    except Exception as e:
        logging.error(f"Erro no processo de treinamento: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
