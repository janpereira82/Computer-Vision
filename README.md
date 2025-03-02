# Projeto de Detecção de Máscaras Faciais

## Descrição do Projeto
Este projeto implementa um sistema de detecção de máscaras faciais utilizando técnicas de Deep Learning e Visão Computacional. O objetivo é identificar automaticamente se uma pessoa está ou não usando máscara facial em imagens.

## Dataset
- **Fonte**: [Face Mask 12K Images Dataset (Kaggle)](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset)
- **Tamanho**: Aproximadamente 12.000 imagens (328,92 MB)
- **Classes**: 
  - Com máscara (WithMask)
  - Sem máscara (WithoutMask)
- **Distribuição**:
  - Treino: 10.000 imagens (5.000 por classe)
  - Validação: 800 imagens (400 por classe)
  - Teste: 992 imagens (483 com máscara, 509 sem máscara)

## Instalação

1. Clone o repositório:
```bash
git clone [URL_DO_REPOSITÓRIO]
cd Face-Mask-Detection
```

2. Crie um ambiente virtual e ative-o:
```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Instale o PyTorch com suporte a CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Estrutura do Projeto
```
Face Mask Detection/
│
├── data/                    # Dataset
│   ├── Train/              # 10.000 imagens
│   │   ├── WithMask/      # 5.000 imagens
│   │   └── WithoutMask/   # 5.000 imagens
│   ├── Test/              # 992 imagens
│   │   ├── WithMask/      # 483 imagens
│   │   └── WithoutMask/   # 509 imagens
│   └── Validation/        # 800 imagens
│       ├── WithMask/      # 400 imagens
│       └── WithoutMask/   # 400 imagens
│
├── src/                    # Código fonte
│   ├── preprocessing/      # Scripts de pré-processamento
│   │   └── image_processor.py
│   ├── models/            # Modelos de deep learning
│   │   └── mask_detector.py
│   ├── utils/             # Funções utilitárias
│   │   └── dataset.py
│   ├── train.py          # Script de treinamento
│   └── predict.py        # Script de predição
│
├── results/              # Resultados e métricas
│   ├── mask_detector.pth # Modelo treinado
│   └── training_history.png
│
├── requirements.txt      # Dependências do projeto
└── README.md            # Este arquivo
```

## Metodologia

### Pré-processamento
- Detecção facial usando OpenCV Haar Cascades
- Redimensionamento das imagens para 224x224 pixels
- Normalização dos valores dos pixels
- Data augmentation:
  - Rotação aleatória (±15°)
  - Espelhamento horizontal
  - Ajuste de brilho e contraste

### Arquitetura do Modelo
- Backbone: ResNet50 pré-treinado no ImageNet
- Camadas adicionais:
  - Linear(2048, 512)
  - ReLU
  - Dropout(0.5)
  - Linear(512, 2)

### Treinamento
- Otimizador: Adam
- Learning rate: 0.001
- Batch size: 32
- Épocas: 10
- Data augmentation em tempo real
- Device: CUDA (GPU)

## Resultados

### Métricas Finais
- **Acurácia no Conjunto de Teste**: 98.49%
- **Perda no Conjunto de Teste**: 0.0314

### Progresso do Treinamento
- **Época 1**: Acurácia Val = 99.50%, Perda Val = 0.0129
- **Época 5**: Acurácia Val = 98.88%, Perda Val = 0.0234
- **Época 10**: Acurácia Val = 99.25%, Perda Val = 0.0169

### Destaques
- O modelo alcançou 100% de acurácia no conjunto de validação durante as épocas 7-9
- A perda de treinamento diminuiu consistentemente ao longo das épocas
- Não houve sinais significativos de overfitting, com performance similar nos conjuntos de validação e teste

## Como Usar

### Treinamento
```bash
python src/train.py --data_dir data --epochs 10 --batch_size 32 --output_dir results
```

### Predição
```bash
python src/predict.py --image_path caminho/para/imagem.jpg --model_path results/mask_detector.pth --output_path resultado.jpg
```

## Próximos Passos
1. Implementar técnicas de ensemble para melhorar ainda mais a performance
2. Experimentar com diferentes arquiteturas (EfficientNet, Vision Transformer)
3. Otimizar hiperparâmetros usando técnicas como grid search ou Optuna
4. Desenvolver uma interface web para demonstração
5. Adicionar suporte para detecção em tempo real via webcam
6. Realizar testes de robustez com diferentes tipos de máscaras e condições de iluminação

## Contribuição
Contribuições são bem-vindas! Por favor, sinta-se à vontade para abrir issues e pull requests.

## Licença
Este projeto está sob a licença MIT.
