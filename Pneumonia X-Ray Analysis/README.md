# Pneumonia Detection from Chest X-Ray Images

Este projeto implementa um modelo de deep learning para detectar pneumonia em imagens de raio-X do tórax. O modelo utiliza uma arquitetura CNN (Convolutional Neural Network) para classificar as imagens em duas categorias: Normal e Pneumonia.

## Estrutura do Projeto

```
Pneumonia X-Ray Analysis/
├── data/
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── test/
│       ├── NORMAL/
│       └── PNEUMONIA/
├── models/
│   ├── best_model.h5
│   ├── modelo_final.h5
│   └── resultados/
│       ├── confusion_matrix.png
│       └── evaluation_report.txt
├── src/
│   ├── data_preparation.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
└── requirements.txt
```

## Dataset

O dataset contém imagens de raio-X do tórax divididas em duas categorias:
- **Normal**: Raio-X de pulmões saudáveis
- **Pneumonia**: Raio-X com indicações de pneumonia

Distribuição dos dados:
- **Treino**: 5216 imagens
  - 20% usado para validação durante o treinamento
- **Teste**: 624 imagens

## Modelo

O modelo utiliza uma arquitetura CNN com as seguintes características:
- Três blocos convolucionais com MaxPooling
- Camadas de Dropout para redução de overfitting
- Otimizador Adam
- Binary Cross-Entropy como função de perda

## Resultados

### Métricas de Treinamento
- Acurácia de validação: 95.97%
- Early stopping implementado para evitar overfitting

### Avaliação no Conjunto de Teste
- **Acurácia**: 76.44%
- **Precisão**: 73.23%
- **Recall**: 98.21%

Análise detalhada por classe:
```
              precision    recall  f1-score   support
      Normal       0.93      0.40      0.56       234
   Pneumonia       0.73      0.98      0.84       390
```

### Interpretação dos Resultados
- O modelo tem alta sensibilidade (recall) para casos de pneumonia (98.21%)
- Apresenta boa precisão para casos normais (93%)
- Existe um trade-off entre precisão e recall, com o modelo tendendo a favorecer a detecção de pneumonia
- O F1-score para pneumonia (0.84) é melhor que para casos normais (0.56)

## Como Usar

1. **Preparação do Ambiente**
```bash
pip install -r requirements.txt
```

2. **Treinamento do Modelo**
```bash
python src/train.py
```

3. **Avaliação do Modelo**
```bash
python src/evaluate.py
```

## Dependências Principais

- TensorFlow 2.13.0
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Próximos Passos

1. **Melhorias no Modelo**:
   - Implementar técnicas de balanceamento de classes
   - Experimentar com diferentes arquiteturas
   - Adicionar data augmentation mais robusto

2. **Melhorias na Avaliação**:
   - Adicionar visualização de casos falso-positivos/negativos
   - Implementar interpretabilidade do modelo (e.g., mapas de calor)
   - Adicionar validação cruzada

3. **Melhorias na Usabilidade**:
   - Criar interface para predições em novas imagens
   - Adicionar pipeline de deploy
   - Implementar versionamento de modelos

## Autor
[Jan Pereira](https://github.com/janpereira82)

## Licença
Este projeto está sob a licença MIT.
