import os
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
from collections import Counter
from PIL import Image, ImageOps

# ==========================================
# ⚙️ CONFIGURATION (SENIOR SETUP)
# ==========================================
LOCAL_DATASET_DIR = "dataset"           # Folder sumber (isi folder: annotations/ & images/)
OUTPUT_DIR = "dataset_prepared_final"   # Folder output baru
YAML_PATH = "facemask_final.yaml"

# Target Kelas
CLASS_NAMES = ["with_mask", "mask_weared_incorrect", "without_mask"]

# Target kelas minoritas untuk di-boost (Oversampling)
MINORITY_CLASS = "mask_weared_incorrect"
OVERSAMPLE_MULTIPLIER = 4  # Data minoritas akan dikali 4x lipat (Copy-Paste)

# Rasio Split (70% Train, 20% Val, 10% Test)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

SEED = 42  # Agar hasil split selalu konsisten (Replicable)

# ==========================================
# 1. HELPER: XML TO YOLO CONVERTER
# ==========================================
def convert_voc_to_yolo(xml_file, output_txt_path, class_names):
    """Konversi XML Pascal VOC ke format YOLO .txt"""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        size = root.find("size")
        img_width = int(size.find("width").text)
        img_height = int(size.find("height").text)
        
        label_lines = []

        for obj in root.findall("object"):
            cls_name = obj.find("name").text
            if cls_name not in class_names:
                continue
            cls_id = class_names.index(cls_name)

            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            # Normalize (xywh)
            x_center = ((xmin + xmax) / 2) / img_width
            y_center = ((ymin + ymax) / 2) / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            # Safety check agar tidak keluar batas
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            width = max(0.0, min(1.0, width))
            height = max(0.0, min(1.0, height))

            label_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # Write to file
        with open(output_txt_path, "w") as f:
            f.write("\n".join(label_lines))
            
    except Exception as e:
        print(f"[WARN] Error converting {xml_file}: {e}")

def get_primary_class(xml_file):
    """Mengambil kelas objek pertama untuk keperluan Stratified Split"""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        obj = root.find("object")
        if obj is not None:
            return obj.find("name").text
    except:
        pass
    return "background"

# ==========================================
# 2. DATASET PREPARATION (STRATIFIED & OVERSAMPLING & SANITIZING)
# ==========================================
def prepare_dataset_pro():
    if os.path.exists(OUTPUT_DIR):
        print(f"[INFO] Folder '{OUTPUT_DIR}' sudah ada. Menghapus dan membuat ulang...")
        shutil.rmtree(OUTPUT_DIR) # Bersihkan run sebelumnya agar tidak numpuk
    
    # Buat 3 folder: Train, Val, Test
    os.makedirs(os.path.join(OUTPUT_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "val"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "test"), exist_ok=True)

    print("[INFO] 1. Mengumpulkan data & Labeling...")
    annot_dir = Path(LOCAL_DATASET_DIR) / "annotations"
    image_dir = Path(LOCAL_DATASET_DIR) / "images"
    
    xml_files = sorted(list(annot_dir.glob("*.xml")))
    
    data_pairs = []
    labels = []

    for xml in xml_files:
        # Cek ekstensi gambar (bisa png atau jpg)
        img_path = image_dir / (xml.stem + ".png")
        if not img_path.exists():
            img_path = image_dir / (xml.stem + ".jpg")
        
        if img_path.exists():
            # Ambil label utama untuk stratifikasi
            cls = get_primary_class(xml)
            if cls in CLASS_NAMES: # Hanya ambil jika kelas valid
                data_pairs.append((xml, img_path))
                labels.append(cls)

    print(f"   -> Total Data Valid: {len(data_pairs)}") #Total Data yang Valid
    print(f"   -> Distribusi Awal: {Counter(labels)}") #Distribusi Kelas

    # -------------------------------------------
    # A. STRATIFIED SPLIT (70% Train, 20% Val, 10% Test)
    # -------------------------------------------
    print("[INFO] 2. Melakukan Stratified Split (70/20/10)...")
    
    # Tahap 1: Pisahkan Train (70%) dan Sisa (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        data_pairs, labels, 
        train_size=TRAIN_RATIO, 
        stratify=labels, 
        random_state=SEED
    )

    # Tahap 2: Pisahkan Sisa (30%) menjadi Val (20% total) dan Test (10% total)
    # Validation size relative to temp = 0.2 / 0.3 = 0.666...
    val_relative_size = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        train_size=val_relative_size, 
        stratify=y_temp, 
        random_state=SEED
    )

    print(f"   -> Split Counts: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # -------------------------------------------
    # B. OVERSAMPLING (Perbanyak data 'Incorrect' HANYA DI TRAIN)
    # -------------------------------------------
    print(f"[INFO] 3. Melakukan Oversampling pada kelas '{MINORITY_CLASS}' (x{OVERSAMPLE_MULTIPLIER})...")
    final_train_data = []
    
    # Masukkan data normal dulu
    for pair in X_train:
        final_train_data.append(pair)
    
    # Masukkan data duplikat khusus minoritas
    oversample_count = 0
    for idx, label in enumerate(y_train):
        if label == MINORITY_CLASS:
            pair = X_train[idx]
            # Tambahkan duplikat
            for _ in range(OVERSAMPLE_MULTIPLIER - 1):
                final_train_data.append(pair)
                oversample_count += 1
                
    print(f"   -> Data Training Awal: {len(X_train)}")
    print(f"   -> Data Training Akhir (setelah boost): {len(final_train_data)} (+{oversample_count} duplikat)")

    # -------------------------------------------
    # C. COPY, SANITIZE & CONVERT FILES
    # -------------------------------------------
    def process_files(data_list, subset_name):
        target_dir = os.path.join(OUTPUT_DIR, subset_name)
        
        for i, (xml_src, img_src) in enumerate(data_list):
            # Rename file agar unik (terutama untuk file hasil oversampling)
            unique_name = f"{img_src.stem}_{i:04d}"  #contoh: wajah_01.jpg (asli) disimpan jadi wajah_01_0000.jpg
            
            new_img_path = os.path.join(target_dir, unique_name + img_src.suffix)
            new_txt_path = os.path.join(target_dir, unique_name + ".txt")
            
            # --- [SANITASI GAMBAR] ---
            try:
                with Image.open(img_src) as img:
                    img = img.convert('RGB') #warna cuma rbg, gada background transparan lagi
                    img = ImageOps.exif_transpose(img)
                    img.save(new_img_path, quality=95, icc_profile=None)
            except Exception as e:
                print(f"[WARN] Gagal sanitasi {img_src.name}, fallback ke copy biasa. Error: {e}")
                shutil.copy(img_src, new_img_path)
            
            # Convert XML ke TXT dengan nama baru
            convert_voc_to_yolo(xml_src, new_txt_path, CLASS_NAMES)

    print("[INFO] 4. Menyalin, Sanitasi, dan Mengkonversi file...")
    process_files(final_train_data, "train")
    process_files(X_val, "val")   # Proses Validasi
    process_files(X_test, "test") # Proses Testing
    print("✅ Dataset Preparation Selesai!")

# ==========================================
# 3. CREATE YAML CONFIG
# ==========================================
def create_yaml():
    # Menggunakan abspath agar path absolut dan aman
    train_path = os.path.abspath(os.path.join(OUTPUT_DIR, 'train'))
    val_path = os.path.abspath(os.path.join(OUTPUT_DIR, 'val'))
    test_path = os.path.abspath(os.path.join(OUTPUT_DIR, 'test'))

    yaml_content = f"""
train: {train_path}
val: {val_path}
test: {test_path}

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
    """
    with open(YAML_PATH, "w") as f:
        f.write(yaml_content.strip())
    print(f"📄 YAML Config saved to: {YAML_PATH}")

# ==========================================
# 4. TRAINING ENGINE (OPTIMIZED)
# ==========================================
def run_training():
    print("\n🚀 Starting Training (Bisa bgt inimah ayo)...")
    
    model = YOLO("yolov8s.pt") 
    
    model.train(
        data=YAML_PATH,
        epochs=100,             
        batch=16,               
        imgsz=640,              
        
        # --- OPTIMIZATION ---
        optimizer="AdamW",      
        lr0=0.001,
        patience=20,            
        cos_lr=True,            
        
        # --- AUGMENTATION BALANCED ---
        augment=True,
        mosaic=1.0,        #1 foto 4 grid     
        mixup=0.15,        #tempel 2 foto bersamaan
        copy_paste=0.0,         
        
        degrees=15.0,           
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, 
        
        # --- SYSTEM ---
        workers=4,              
        project="facemask_final",
        name="run_split_70_20_10",
        verbose=True
    )
    return model

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Siapkan Data (70% Train, 20% Val, 10% Test)
    prepare_dataset_pro()
    create_yaml()
    
    # 2. Jalankan Training
    final_model = run_training()