import cv2
import numpy as np
from pathlib import Path
import re
import logging

def validar_formato_placa(texto):
    """
    Valida se o texto reconhecido segue o formato de placa brasileiro
    
    Args:
        texto: Texto da placa reconhecida
        
    Returns:
        bool: True se o formato é válido, False caso contrário
    """
    # Formato Mercosul: 3 letras, 1 número, 1 letra, 2 números
    padrao_mercosul = re.compile(r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$')
    
    # Formato antigo: 3 letras, 4 números
    padrao_antigo = re.compile(r'^[A-Z]{3}[0-9]{4}$')
    
    texto = texto.upper().replace(' ', '')
    return bool(padrao_mercosul.match(texto) or padrao_antigo.match(texto))

def aumentar_contraste(imagem):
    """
    Aumenta o contraste da imagem usando CLAHE
    
    Args:
        imagem: Imagem em escala de cinza
        
    Returns:
        Imagem com contraste aumentado
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(imagem)

def remover_ruido(imagem):
    """
    Remove ruído da imagem usando filtro bilateral
    
    Args:
        imagem: Imagem de entrada
        
    Returns:
        Imagem com ruído reduzido
    """
    return cv2.bilateralFilter(imagem, 9, 75, 75)

def redimensionar_imagem(imagem, largura=None, altura=None):
    """
    Redimensiona a imagem mantendo a proporção
    
    Args:
        imagem: Imagem de entrada
        largura: Largura desejada
        altura: Altura desejada
        
    Returns:
        Imagem redimensionada
    """
    dim = None
    (h, w) = imagem.shape[:2]

    if largura is None and altura is None:
        return imagem

    if largura is None:
        r = altura / float(h)
        dim = (int(w * r), altura)
    else:
        r = largura / float(w)
        dim = (largura, int(h * r))

    return cv2.resize(imagem, dim, interpolation=cv2.INTER_AREA)

def salvar_resultados(imagem, texto, caminho_saida, nome_arquivo):
    """
    Salva a imagem processada e o texto reconhecido
    
    Args:
        imagem: Imagem processada
        texto: Texto reconhecido
        caminho_saida: Diretório de saída
        nome_arquivo: Nome do arquivo
    """
    try:
        # Cria o diretório se não existir
        Path(caminho_saida).mkdir(parents=True, exist_ok=True)
        
        # Salva a imagem
        caminho_imagem = Path(caminho_saida) / f"{nome_arquivo}_processada.jpg"
        cv2.imwrite(str(caminho_imagem), imagem)
        
        # Salva o texto
        caminho_texto = Path(caminho_saida) / f"{nome_arquivo}_resultado.txt"
        with open(caminho_texto, 'w') as f:
            f.write(f"Placa detectada: {texto}\n")
            
        logging.info(f"Resultados salvos em {caminho_saida}")
        
    except Exception as e:
        logging.error(f"Erro ao salvar resultados: {str(e)}")
        raise

def calcular_metricas(predicoes, ground_truth):
    """
    Calcula métricas de avaliação do modelo
    
    Args:
        predicoes: Lista de placas detectadas
        ground_truth: Lista de placas reais
        
    Returns:
        dict: Dicionário com as métricas calculadas
    """
    total_predicoes = len(predicoes)
    total_gt = len(ground_truth)
    
    # Conta acertos
    acertos = sum(1 for p in predicoes if p in ground_truth)
    
    # Calcula métricas
    precisao = acertos / total_predicoes if total_predicoes > 0 else 0
    recall = acertos / total_gt if total_gt > 0 else 0
    f1 = 2 * (precisao * recall) / (precisao + recall) if (precisao + recall) > 0 else 0
    
    return {
        'precisao': precisao,
        'recall': recall,
        'f1': f1,
        'total_predicoes': total_predicoes,
        'total_ground_truth': total_gt,
        'acertos': acertos
    }

def criar_visualizacao(imagem, bbox, texto, cor=(0, 255, 0)):
    """
    Cria uma visualização da detecção na imagem
    
    Args:
        imagem: Imagem original
        bbox: Coordenadas do retângulo [x1, y1, x2, y2]
        texto: Texto da placa
        cor: Cor do retângulo e texto (B, G, R)
        
    Returns:
        Imagem com anotações
    """
    img_viz = imagem.copy()
    
    # Desenha o retângulo
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(img_viz, (x1, y1), (x2, y2), cor, 2)
    
    # Adiciona o texto
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    tamanho_fonte = 0.9
    espessura = 2
    
    # Calcula o tamanho do texto
    (largura_texto, altura_texto), _ = cv2.getTextSize(texto, fonte, tamanho_fonte, espessura)
    
    # Desenha um fundo para o texto
    cv2.rectangle(img_viz, (x1, y1-altura_texto-10), (x1+largura_texto, y1), cor, -1)
    
    # Adiciona o texto
    cv2.putText(img_viz, texto, (x1, y1-10), fonte, tamanho_fonte, (0, 0, 0), espessura)
    
    return img_viz
