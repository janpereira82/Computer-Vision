import os
import shutil
import random
from collections import Counter
import cv2
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

def check_and_remove_corrupt_images(image_path):
    """Verifica se a imagem está corrompida"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Removendo imagem corrompida: {image_path}")
            os.remove(image_path)
            return False
        return True
    except:
        if os.path.exists(image_path):
            print(f"Removendo imagem corrompida: {image_path}")
            os.remove(image_path)
        return False

def generate_augmented_images(img_path, output_dir, num_augmented=5):
    """Gera imagens aumentadas para classes com poucas amostras"""
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    i = 0
    for batch in datagen.flow(x, batch_size=1,
                            save_to_dir=output_dir,
                            save_prefix='aug',
                            save_format='jpg'):
        i += 1
        if i >= num_augmented:
            break

def balance_dataset(src_dir, dst_dir, target_per_class=200):
    """Equilibra o dataset usando augmentation para classes com poucas imagens"""
    print("\nIniciando balanceamento do dataset...")
    os.makedirs(dst_dir, exist_ok=True)
    
    # Listar todas as classes
    classes = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
    print("Classes encontradas:", classes)
    
    # Contar e verificar imagens válidas em cada classe
    class_counts = {}
    for class_name in classes:
        class_dir = os.path.join(src_dir, class_name)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"\nVerificando imagens da classe {class_name}...")
        valid_images = []
        for img in tqdm(images):
            img_path = os.path.join(class_dir, img)
            if check_and_remove_corrupt_images(img_path):
                valid_images.append(img)
        
        class_counts[class_name] = len(valid_images)
        print(f"Imagens válidas em {class_name}: {len(valid_images)}")
    
    # Processar cada classe
    for class_name in classes:
        src_class_dir = os.path.join(src_dir, class_name)
        dst_class_dir = os.path.join(dst_dir, class_name)
        os.makedirs(dst_class_dir, exist_ok=True)
        
        # Listar imagens válidas
        images = [f for f in os.listdir(src_class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Copiar todas as imagens originais
        print(f"\nProcessando classe {class_name}...")
        for img in tqdm(images):
            src_path = os.path.join(src_class_dir, img)
            dst_path = os.path.join(dst_class_dir, img)
            shutil.copy2(src_path, dst_path)
        
        # Se precisar de mais imagens, gerar através de augmentation
        current_count = len(images)
        if current_count < target_per_class:
            augmentation_needed = target_per_class - current_count
            augmentation_per_image = (augmentation_needed // current_count) + 1
            
            print(f"Gerando {augmentation_needed} imagens adicionais para {class_name}...")
            for img in tqdm(images):
                src_path = os.path.join(src_class_dir, img)
                generate_augmented_images(src_path, dst_class_dir, augmentation_per_image)
                
                # Verificar se já atingimos o objetivo
                current_files = len([f for f in os.listdir(dst_class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                if current_files >= target_per_class:
                    break
    
    # Mostrar contagem final
    final_counts = {}
    for class_name in classes:
        class_dir = os.path.join(dst_dir, class_name)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        final_counts[class_name] = len(images)
    
    print("\nContagem final de imagens por classe:")
    for class_name, count in final_counts.items():
        print(f"{class_name}: {count} imagens")

def main():
    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "animais")
    dst_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dataset_balanceado")
    
    # Equilibrar para 200 imagens por classe
    balance_dataset(src_dir, dst_dir, target_per_class=200)
    
    print("\nDataset balanceado criado com sucesso!")

if __name__ == "__main__":
    main()
