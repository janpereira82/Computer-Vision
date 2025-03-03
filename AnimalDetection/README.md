# Sistema de Detecção de Animais

## Descrição
Sistema de detecção e classificação de animais usando deep learning, focado em identificar diferentes espécies como aranhas, cobras, escorpiões, lagartas e mariposas.

## Estrutura do Projeto
```
AnimalDetection/
├── data/
│   ├── animais/           # Dataset original
│   └── dataset_balanceado/ # Dataset balanceado (~200 imagens/classe)
├── checkpoints/           # Modelos salvos durante o treinamento
├── src/                   # Código fonte do projeto
├── balance_dataset.py     # Script para equilibrar o dataset
├── download_dataset.py    # Script para download de imagens
├── train.py              # Script de treinamento do modelo
└── README.md
```

## Tecnologias Utilizadas
- Python 3.8+
- TensorFlow 2.x
- MobileNetV2 (modelo base)
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

## Dataset
- Total de imagens: ~1000 imagens balanceadas
- Classes: 5 (aranha, cobra, escorpião, lagarta, mariposa)
- Resolução das imagens: 224x224 pixels
- Augmentation aplicado: rotação, zoom, flips, brilho

## Resultados do Treinamento

### Métricas Gerais
- Acurácia geral: 83%
- Macro avg F1-score: 0.83
- Weighted avg F1-score: 0.83

### Performance por Classe
| Classe    | Precisão | Recall | F1-score |
|-----------|----------|--------|----------|
| Aranha    | 1.00     | 0.79   | 0.88     |
| Cobra     | 1.00     | 0.57   | 0.73     |
| Escorpião | 0.67     | 0.93   | 0.78     |
| Lagarta   | 0.80     | 0.98   | 0.88     |
| Mariposa  | 0.84     | 0.90   | 0.87     |

### Análise dos Resultados
- Excelente precisão na detecção de aranhas e cobras (100%)
- Alto recall para lagartas (98%) e escorpiões (93%)
- Mariposas apresentam bom equilíbrio entre precisão (84%) e recall (90%)
- Dataset balanceado melhorou significativamente a performance geral

## Como Usar
1. Clone o repositório
2. Instale as dependências: `pip install -r requirements.txt`
3. Execute o download do dataset: `python download_dataset.py`
4. Balance o dataset: `python balance_dataset.py`
5. Treine o modelo: `python train.py`

## Otimizações Implementadas
1. **Balanceamento de Dataset**:
   - Geração de imagens sintéticas para classes minoritárias
   - ~200 imagens por classe após balanceamento
   
2. **Arquitetura do Modelo**:
   - MobileNetV2 como backbone (otimizado para CPU)
   - Camadas de BatchNormalization para melhor generalização
   - Dropout progressivo (0.5, 0.4, 0.3) para reduzir overfitting
   
3. **Estratégia de Treinamento**:
   - Learning rate reduzido (0.0005)
   - Data augmentation mais agressivo
   - Early stopping com maior paciência
   - Fine-tuning em mais camadas do modelo base

## Próximos Passos
1. Implementar técnicas de data cleaning mais avançadas
2. Adicionar validação cruzada para avaliação mais robusta
3. Experimentar com ensemble de modelos
4. Implementar interface de usuário para teste do modelo
5. Adicionar detecção em tempo real via webcam

## Autor
[Jan Pereira](https://github.com/janpereira82)
