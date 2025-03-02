# Dados do Projeto

Os dados de treinamento não estão incluídos diretamente no repositório devido ao seu tamanho. Para obter os dados:

1. Acesse o dataset no Roboflow Universe:
   [Car Plate Detection Dataset](https://universe.roboflow.com/trabalho-jnal6/placa-de-carro-oz0eg/dataset/6)

2. Faça o download do dataset e extraia nas seguintes pastas:
   - `data/train/` - Para imagens e labels de treinamento
   - `data/valid/` - Para imagens e labels de validação
   - `data/test/` - Para imagens e labels de teste

## Estrutura de Dados
```
data/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## Formato dos Dados
- Imagens: formato JPG
- Labels: formato YOLO (txt)
- Resolução: variável
- Classes: 1 (placa)
