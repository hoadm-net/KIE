"""
Script để download và setup FUNSD dataset
"""
import os
import urllib.request
import zipfile
from pathlib import Path

# Configuration
DATASET_URL = "https://guillaumejaume.github.io/FUNSD/dataset.zip"
DOWNLOAD_DIR = "data"
ZIP_FILE = "dataset.zip"
EXTRACT_DIR = "data/FUNSD"


def download_file(url, destination):
    """Download file với progress bar"""
    print(f"📥 Đang download từ: {url}")
    print(f"📁 Lưu vào: {destination}")
    
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100.0 / total_size, 100)
        bar_length = 50
        filled = int(bar_length * percent / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f'\r[{bar}] {percent:.1f}% ({downloaded/1024/1024:.1f}MB/{total_size/1024/1024:.1f}MB)', end='')
    
    try:
        urllib.request.urlretrieve(url, destination, reporthook=progress_hook)
        print('\n✅ Download hoàn tất!')
        return True
    except Exception as e:
        print(f'\n❌ Lỗi khi download: {e}')
        return False


def extract_zip(zip_path, extract_to):
    """Extract file zip"""
    print(f"\n📦 Đang giải nén: {zip_path}")
    print(f"📁 Vào thư mục: {extract_to}")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # List all files
            file_list = zip_ref.namelist()
            total_files = len(file_list)
            
            # Extract với progress
            for i, file in enumerate(file_list):
                zip_ref.extract(file, extract_to)
                percent = (i + 1) * 100.0 / total_files
                print(f'\rGiải nén: {percent:.1f}% ({i+1}/{total_files} files)', end='')
            
        print('\n✅ Giải nén hoàn tất!')
        
        # Move dataset folder if extracted to nested directory
        extracted_dataset = os.path.join(extract_to, 'dataset')
        target_dir = os.path.join(extract_to, 'FUNSD')
        
        if os.path.exists(extracted_dataset):
            print(f"📂 Di chuyển dataset từ {extracted_dataset} đến {target_dir}")
            import shutil
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            shutil.move(extracted_dataset, target_dir)
            
            # Remove __MACOSX if exists
            macosx_dir = os.path.join(extract_to, '__MACOSX')
            if os.path.exists(macosx_dir):
                shutil.rmtree(macosx_dir)
                print("🧹 Đã xóa thư mục __MACOSX")
        
        return True
    except Exception as e:
        print(f'\n❌ Lỗi khi giải nén: {e}')
        return False


def verify_structure(base_dir):
    """Verify cấu trúc dataset"""
    print(f"\n🔍 Đang kiểm tra cấu trúc dataset...")
    
    # Expected structure
    expected_dirs = [
        'training_data/images',
        'training_data/annotations',
        'testing_data/images',
        'testing_data/annotations'
    ]
    
    all_good = True
    for dir_path in expected_dirs:
        full_path = os.path.join(base_dir, dir_path)
        if os.path.exists(full_path):
            # Count files
            files = os.listdir(full_path)
            print(f"✅ {dir_path}: {len(files)} files")
        else:
            print(f"❌ Không tìm thấy: {dir_path}")
            all_good = False
    
    return all_good


def count_statistics(base_dir):
    """Đếm statistics của dataset"""
    print(f"\n📊 THỐNG KÊ DATASET")
    print("="*60)
    
    splits = ['training_data', 'testing_data']
    total_images = 0
    total_annotations = 0
    
    for split in splits:
        images_dir = os.path.join(base_dir, split, 'images')
        annotations_dir = os.path.join(base_dir, split, 'annotations')
        
        if os.path.exists(images_dir):
            images = [f for f in os.listdir(images_dir) if f.endswith('.png')]
            annotations = [f for f in os.listdir(annotations_dir) if f.endswith('.json')]
            
            print(f"\n{split}:")
            print(f"  - Images: {len(images)}")
            print(f"  - Annotations: {len(annotations)}")
            
            total_images += len(images)
            total_annotations += len(annotations)
    
    print(f"\n{'='*60}")
    print(f"TỔNG:")
    print(f"  - Images: {total_images}")
    print(f"  - Annotations: {total_annotations}")
    print(f"{'='*60}\n")


def cleanup(file_path):
    """Xóa file tạm"""
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"🧹 Đã xóa file tạm: {file_path}")


def main():
    """Main function"""
    print("\n" + "="*60)
    print("🚀 FUNSD DATASET DOWNLOADER & SETUP")
    print("="*60 + "\n")
    
    # Create download directory
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    
    zip_path = os.path.join(DOWNLOAD_DIR, ZIP_FILE)
    
    # Check if FUNSD dataset already exists
    if os.path.exists(EXTRACT_DIR):
        response = input(f"⚠️  Dataset FUNSD đã tồn tại tại {EXTRACT_DIR}. Xóa và download lại? (y/n): ")
        if response.lower() == 'y':
            import shutil
            shutil.rmtree(EXTRACT_DIR)
            print(f"🗑️  Đã xóa dataset FUNSD cũ")
        else:
            print("ℹ️  Giữ dataset hiện tại và thoát.")
            return
    
    # Step 1: Download
    if not download_file(DATASET_URL, zip_path):
        return
    
    # Step 2: Extract
    if not extract_zip(zip_path, DOWNLOAD_DIR):
        return
    
    # Step 3: Verify structure
    if not verify_structure(EXTRACT_DIR):
        print("\n⚠️  Cấu trúc dataset không đúng! Vui lòng kiểm tra lại.")
        return
    
    # Step 4: Count statistics
    count_statistics(EXTRACT_DIR)
    
    # Step 5: Cleanup
    cleanup(zip_path)
    
    # Done
    print("✅ Setup FUNSD dataset hoàn tất!")
    print(f"📂 Dataset location: {os.path.abspath(EXTRACT_DIR)}\n")
    print("🎯 Bạn có thể bắt đầu sử dụng dataset ngay bây giờ!")
    print(f"   - Training data: {EXTRACT_DIR}/training_data")
    print(f"   - Testing data: {EXTRACT_DIR}/testing_data\n")


if __name__ == "__main__":
    main()
