# Detecção de Placas de Carros 

## Descrição do Projeto
Este projeto implementa um sistema de detecção de placas de carros utilizando técnicas de Visão Computacional e Aprendizado de Máquina. O sistema é capaz de identificar e extrair informações de placas de carros em imagens.

## Estrutura do Projeto
```
Car-Plate-Detection/
│
├── data/               # Diretório com as imagens
│   ├── train/         # Imagens de treinamento (400 imagens)
│   ├── valid/         # Imagens de validação (261 imagens)
│   └── test/          # Imagens de teste (2 imagens)
│
├── src/               # Códigos fonte
│   ├── detect.py      # Script principal de detecção
│   ├── train.py       # Script de treinamento
│   └── utils.py       # Funções auxiliares
│
├── models/            # Modelos treinados
│
├── results/           # Resultados e métricas
│
└── requirements.txt   # Dependências do projeto
```

## Tecnologias Utilizadas
- **YOLOv8**: Framework principal para detecção de objetos
- **OpenCV**: Processamento de imagens e visualização
- **EasyOCR**: Reconhecimento óptico de caracteres
- **Python**: Linguagem de programação
- **PyTorch**: Framework de deep learning

## Configurações do Modelo
O modelo foi treinado utilizando as seguintes configurações otimizadas:

### Arquitetura
- Base: YOLOv8n (nano)
- Tamanho de entrada: 416x416 pixels
- Classes: 1 (placa)

### Hiperparâmetros
- **Treinamento**:
  - Épocas: 30
  - Batch size: 16
  - Otimizador: AdamW
  - Learning rate inicial: 0.001
  - Learning rate final: 0.01
  - Scheduler: Cosseno com warmup

- **Data Augmentation**:
  - Mosaic: 50% das imagens
  - Mixup: 30% das imagens
  - Rotação máxima: 10 graus
  - Escala: 40%

### Recursos de Otimização
- Early stopping (paciência: 10 épocas)
- Validação contínua
- Salvamento do melhor modelo

## Conjunto de Dados
- **Treino**: 400 imagens
- **Validação**: 261 imagens
- **Teste**: 2 imagens
- **Formato**: YOLO (normalizado)
- **Fonte**: Roboflow Universe - [Car Plate Detection Dataset](https://universe.roboflow.com/trabalho-jnal6/placa-de-carro-oz0eg/dataset/6)

## Resultados
O modelo foi treinado com foco em eficiência e velocidade, mantendo um bom equilíbrio entre performance e tempo de treinamento.

### Métricas de Performance
- **Tempo de Treinamento**: < 2 horas
- **Hardware**: CPU
- **Otimizações**: 
  - Redução do tamanho da imagem
  - Data augmentation moderado
  - Early stopping agressivo

## Resultados da Avaliação

### Métricas no Conjunto de Validação
- **Total de Imagens**: 261
- **Tempo Médio de Inferência**: 76.4ms
- **Confiança Média**: 92.27%
- **Taxa de Detecção**: 100% (261/261 imagens)
- **Média de Detecções por Imagem**: 1.023

### Métricas no Conjunto de Teste
- **Total de Imagens**: 2
- **Tempo Médio de Inferência**: 131.2ms
- **Taxa de Detecção**: 0% (0/2 imagens)
- **Média de Detecções por Imagem**: 0

### Performance em Tempo Real
- **FPS Médio**: 14.45
- **FPS Mínimo**: 7.30
- **FPS Máximo**: 20.63
- **Tempo Médio de Inferência**: 69.2ms
- **Tempo Total de Processamento**: 3.46s (50 frames)

### Análise dos Resultados

1. **Conjunto de Validação**:
   - Excelente taxa de detecção (100%)
   - Alta confiança média nas detecções (92.27%)
   - Tempo de inferência adequado para aplicações não-tempo-real
   - Média de detecções próxima a 1, indicando boa precisão

2. **Conjunto de Teste**:
   - Resultados limitados devido ao pequeno número de amostras
   - Necessidade de expandir o conjunto de teste para melhor avaliação

3. **Performance em Tempo Real**:
   - FPS médio adequado para aplicações em tempo real
   - Variação aceitável entre FPS mínimo e máximo
   - Tempo de inferência consistente com o conjunto de validação

### Pontos de Melhoria
1. Expandir o conjunto de teste para uma avaliação mais robusta
2. Otimizar o tempo de inferência para melhorar o FPS em tempo real
3. Investigar possíveis falsos positivos/negativos
4. Implementar técnicas de pós-processamento para melhorar a precisão

## Uso do Modelo
Para utilizar o modelo treinado:

1. Instale as dependências:
```bash
pip install -r requirements.txt
```

2. Execute a detecção:
```bash
python src/detect.py --source sua_imagem.jpg
```

## Próximos Passos
- [ ] Implementar detecção em tempo real
- [ ] Otimizar para diferentes condições de iluminação
- [ ] Adicionar suporte para múltiplas placas
- [ ] Integrar OCR para leitura dos caracteres
- [ ] Desenvolver interface gráfica

## Licença
Este projeto está sob a licença MIT.

## Autor
[Jan Pereira](https://github.com/janpereira82)

## Agradecimentos
- Roboflow Universe pela disponibilização do dataset
- Comunidade YOLOv8 pelo framework de detecção
- Comunidade de Visão Computacional pelos recursos
