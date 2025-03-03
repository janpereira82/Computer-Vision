import os
import urllib.request
import requests
from tqdm import tqdm
import time
import random
from concurrent.futures import ThreadPoolExecutor
import json

def download_file(url, filename, headers=None):
    """Download file from URL with progress bar"""
    try:
        if headers is None:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as file, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                pbar.update(size)
        return True
    except Exception as e:
        print(f"Erro ao baixar {url}: {str(e)}")
        if os.path.exists(filename):
            os.remove(filename)
        return False

def get_pixabay_images(query, api_key, per_page=50):
    """Get image URLs from Pixabay API"""
    base_url = "https://pixabay.com/api/"
    params = {
        "key": api_key,
        "q": query,
        "image_type": "photo",
        "per_page": per_page
    }
    
    try:
        response = requests.get(base_url, params=params)
        data = response.json()
        return [hit["largeImageURL"] for hit in data.get("hits", [])]
    except Exception as e:
        print(f"Erro ao buscar imagens do Pixabay: {str(e)}")
        return []

def get_unsplash_images(query, client_id, per_page=30):
    """Get image URLs from Unsplash API"""
    base_url = "https://api.unsplash.com/search/photos"
    headers = {"Authorization": f"Client-ID {client_id}"}
    params = {
        "query": query,
        "per_page": per_page
    }
    
    try:
        response = requests.get(base_url, headers=headers, params=params)
        data = response.json()
        return [photo["urls"]["regular"] for photo in data.get("results", [])]
    except Exception as e:
        print(f"Erro ao buscar imagens do Unsplash: {str(e)}")
        return []

def get_wikimedia_images(query, limit=50):
    """Get image URLs from Wikimedia Commons"""
    base_url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": f"{query} filetype:bitmap",
        "srnamespace": "6",
        "srlimit": limit
    }
    
    try:
        response = requests.get(base_url, params=params)
        data = response.json()
        
        image_urls = []
        for item in data.get("query", {}).get("search", []):
            title = item["title"]
            if any(ext in title.lower() for ext in ['.jpg', '.jpeg', '.png']):
                file_name = title.replace("File:", "").replace(" ", "_")
                image_url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{file_name}"
                image_urls.append(image_url)
        
        return image_urls
    except Exception as e:
        print(f"Erro ao buscar imagens do Wikimedia: {str(e)}")
        return []

def download_images_for_category(category, search_terms, base_dir, max_images=1000):
    """Download images for a specific category using multiple sources"""
    print(f"\nBaixando imagens de {category}...")
    category_dir = os.path.join(base_dir, category)
    os.makedirs(category_dir, exist_ok=True)
    
    # APIs (você precisará registrar e obter suas próprias chaves)
    PIXABAY_API_KEY = "SUA_CHAVE_PIXABAY"  # Substitua pela sua chave
    UNSPLASH_CLIENT_ID = "SEU_CLIENT_ID"    # Substitua pela sua chave
    
    image_count = 0
    for term in search_terms:
        print(f"Buscando por: {term}")
        
        # Coletar URLs de diferentes fontes
        urls = []
        urls.extend(get_wikimedia_images(term))
        if PIXABAY_API_KEY != "SUA_CHAVE_PIXABAY":
            urls.extend(get_pixabay_images(term, PIXABAY_API_KEY))
        if UNSPLASH_CLIENT_ID != "SEU_CLIENT_ID":
            urls.extend(get_unsplash_images(term, UNSPLASH_CLIENT_ID))
        
        # Embaralhar URLs para maior diversidade
        random.shuffle(urls)
        
        # Download das imagens
        with ThreadPoolExecutor(max_workers=5) as executor:
            for url in urls:
                if image_count >= max_images:
                    break
                
                filename = os.path.join(category_dir, f"{category}_{image_count+1}.jpg")
                future = executor.submit(download_file, url, filename)
                
                if future.result():
                    image_count += 1
                    print(f"Downloaded ({image_count}/{max_images}): {filename}")
                
                time.sleep(random.uniform(0.1, 0.3))
                
            if image_count >= max_images:
                break
    
    print(f"Total de imagens baixadas para {category}: {image_count}")
    return image_count

def main():
    # Configurações
    base_dir = "data/animais"
    os.makedirs(base_dir, exist_ok=True)
    
    # Termos de busca em português e inglês para maior cobertura
    searches = {
        "aranha": [
            "aranha brasileira", "spider species", 
            "tarantula brasil", "caranguejeira brasil",
            "aranha armadeira", "aranha marrom"
        ],
        "cobra": [
            "cobra brasileira", "snake species brazil",
            "jararaca brasil", "sucuri brasil",
            "cobra coral brasil", "python snake brazil"
        ],
        "escorpiao": [
            "escorpião brasileiro", "scorpion species brazil",
            "escorpião amarelo", "escorpião preto",
            "tityus serrulatus", "escorpião marrom"
        ],
        "lagarta": [
            "lagarta brasileira", "caterpillar brazil",
            "lagarta de mariposa", "lagarta de borboleta",
            "taturana brasil", "lagarta de fogo"
        ]
    }
    
    total_images = 0
    for category, search_terms in searches.items():
        total_images += download_images_for_category(
            category, 
            search_terms, 
            base_dir,
            max_images=1000  # 1000 imagens por categoria
        )
    
    print(f"\nTotal geral de imagens baixadas: {total_images}")

if __name__ == "__main__":
    main()
