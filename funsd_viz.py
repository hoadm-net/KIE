"""
Visualize FUNSD dataset - Hiển thị form với annotations
"""
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
import argparse

# Màu sắc cho các loại entity
LABEL_COLORS = {
    'question': '#3498db',  # Blue
    'answer': '#2ecc71',    # Green
    'header': '#e74c3c',    # Red/Orange
    'other': '#95a5a6'      # Gray
}

LABEL_NAMES = {
    'question': 'Câu hỏi',
    'answer': 'Câu trả lời',
    'header': 'Tiêu đề',
    'other': 'Khác'
}


def load_funsd_data(image_path, annotation_path):
    """Load image và annotation của FUNSD"""
    # Load image
    img = Image.open(image_path)
    
    # Load annotation
    with open(annotation_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return img, data


def visualize_funsd(image_path, annotation_path, save_path=None):
    """
    Visualize FUNSD document với annotations
    
    Args:
        image_path: Đường dẫn đến ảnh
        annotation_path: Đường dẫn đến file JSON annotation
        save_path: Đường dẫn lưu ảnh (optional)
    """
    # Load data
    img, data = load_funsd_data(image_path, annotation_path)
    
    # Tạo figure với 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))
    
    # Subplot 1: Hiển thị ảnh gốc
    ax1.imshow(img, cmap='gray')
    ax1.set_title('Document gốc', fontsize=16, fontweight='bold', pad=20)
    ax1.axis('off')
    
    # Subplot 2: Hiển thị với annotations
    ax2.imshow(img, cmap='gray')
    ax2.set_title('Document với Annotations', fontsize=16, fontweight='bold', pad=20)
    ax2.axis('off')
    
    # Statistics
    stats = {'question': 0, 'answer': 0, 'header': 0, 'other': 0}
    relations = []
    
    # Vẽ các bounding boxes và text
    for entity in data['form']:
        box = entity['box']  # [x_left, y_top, x_right, y_bottom]
        label = entity['label']
        text = entity['text']
        entity_id = entity['id']
        
        # Update statistics
        stats[label] += 1
        
        # Tính width và height
        x, y = box[0], box[1]
        width = box[2] - box[0]
        height = box[3] - box[1]
        
        # Vẽ bounding box
        color = LABEL_COLORS[label]
        rect = patches.Rectangle(
            (x, y), width, height,
            linewidth=2,
            edgecolor=color,
            facecolor='none',
            alpha=0.8
        )
        ax2.add_patch(rect)
        
        # Vẽ text label (rút gọn nếu quá dài)
        display_text = text[:30] + '...' if len(text) > 30 else text
        ax2.text(
            x, y - 5,
            display_text,
            fontsize=8,
            color=color,
            weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor=color)
        )
        
        # Collect relations
        if entity['linking']:
            for link in entity['linking']:
                relations.append((link[0], link[1]))
    
    # Tạo legend
    legend_elements = [
        patches.Patch(facecolor=LABEL_COLORS[label], edgecolor='black', label=f'{LABEL_NAMES[label]} ({count})')
        for label, count in stats.items()
    ]
    ax2.legend(
        handles=legend_elements,
        loc='upper right',
        fontsize=10,
        framealpha=0.9
    )
    
    # Thêm thông tin tổng quan
    total_entities = sum(stats.values())
    info_text = f'Tổng entities: {total_entities}\nRelations: {len(relations)}'
    fig.text(
        0.5, 0.02,
        info_text,
        ha='center',
        fontsize=12,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    
    # Save nếu có path
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Đã lưu visualization tại: {save_path}")
    
    plt.show()
    
    return stats, relations


def print_questions_and_answers(annotation_path):
    """
    In ra các cặp question-answer từ annotation
    """
    with open(annotation_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Tạo dict để map id -> entity
    entities = {entity['id']: entity for entity in data['form']}
    
    print("\n" + "="*80)
    print("📋 CÁC CẶP QUESTION-ANSWER TRONG DOCUMENT")
    print("="*80 + "\n")
    
    # Tìm và in các cặp question-answer
    qa_count = 0
    for entity in data['form']:
        if entity['label'] == 'question' and entity['linking']:
            question_text = entity['text']
            
            # Tìm answers liên kết
            for link in entity['linking']:
                from_id, to_id = link
                if to_id in entities and entities[to_id]['label'] == 'answer':
                    answer_text = entities[to_id]['text']
                    qa_count += 1
                    print(f"Q{qa_count}: {question_text}")
                    print(f"A{qa_count}: {answer_text}")
                    print("-" * 80)
    
    if qa_count == 0:
        print("❌ Không tìm thấy cặp question-answer nào trong document này.")
    else:
        print(f"\n✅ Tổng cộng: {qa_count} cặp question-answer\n")


def main():
    """Main function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Visualize FUNSD dataset')
    parser.add_argument(
        '--subset',
        type=str,
        default='train',
        choices=['train', 'test'],
        help='Dataset subset to use (train or test). Default: train'
    )
    parser.add_argument(
        '--image',
        type=str,
        default='0000971160',
        help='Image ID to visualize (with or without .png extension). Default: 0000971160'
    )
    args = parser.parse_args()
    
    # Map subset to folder name
    subset_map = {
        'train': 'training_data',
        'test': 'testing_data'
    }
    subset_folder = subset_map[args.subset]
    
    # Remove .png extension if present
    doc_id = args.image.replace('.png', '')
    
    # Đường dẫn đến dữ liệu
    base_path = f"data/FUNSD/{subset_folder}"
    
    image_path = os.path.join(base_path, "images", f"{doc_id}.png")
    annotation_path = os.path.join(base_path, "annotations", f"{doc_id}.json")
    
    # Kiểm tra files có tồn tại không
    if not os.path.exists(image_path):
        print(f"❌ Không tìm thấy image: {image_path}")
        print(f"💡 Hãy kiểm tra ID ảnh trong thư mục: {base_path}/images/")
        return
    
    if not os.path.exists(annotation_path):
        print(f"❌ Không tìm thấy annotation: {annotation_path}")
        return
    
    print(f"📄 Đang visualize document: {doc_id}")
    print(f"📂 Subset: {args.subset} ({subset_folder})")
    print(f"📁 Image: {image_path}")
    print(f"📁 Annotation: {annotation_path}\n")
    
    # Visualize
    viz_dir = "data/FUNSD/visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    save_path = f"{viz_dir}/{doc_id}_annotated.png"
    
    stats, relations = visualize_funsd(image_path, annotation_path, save_path)
    
    # In thống kê
    print("\n" + "="*80)
    print("📊 THỐNG KÊ DOCUMENT")
    print("="*80)
    for label, count in stats.items():
        print(f"{LABEL_NAMES[label]:15s}: {count:3d}")
    print(f"{'Relations':15s}: {len(relations):3d}")
    
    # In các cặp question-answer
    print_questions_and_answers(annotation_path)


if __name__ == "__main__":
    main()
