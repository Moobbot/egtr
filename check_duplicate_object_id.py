import json
from collections import defaultdict

def check_duplicate_object_ids(json_file_path):
    """
    Kiểm tra xem có object nào bị trùng object_id trong từng ảnh không
    """
    try:
        # Đọc file JSON
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Đang kiểm tra file: {json_file_path}")
        print(f"Tổng số ảnh trong dataset: {len(data)}")
        print("=" * 60)
        
        total_duplicates = 0
        images_with_duplicates = 0
        
        # Kiểm tra từng ảnh
        for i, image_data in enumerate(data):
            if not image_data or 'image_id' not in image_data:
                continue
                
            image_id = image_data['image_id']
            objects = image_data.get('objects', [])
            
            if not objects:
                continue
            
            # Đếm số lần xuất hiện của mỗi object_id
            object_id_count = defaultdict(int)
            
            for obj in objects:
                if 'object_id' in obj:
                    object_id_count[obj['object_id']] += 1
            
            # Tìm các object_id bị trùng
            duplicates = {oid: count for oid, count in object_id_count.items() if count > 1}
            
            if duplicates:
                images_with_duplicates += 1
                total_duplicates += sum(duplicates.values()) - len(duplicates)
                
                print(f"Ảnh ID {image_id} (vị trí {i}):")
                print(f"  - Tổng số objects: {len(objects)}")
                for oid, count in duplicates.items():
                    print(f"  - object_id {oid} bị trùng {count} lần")
                
                # Hiển thị chi tiết các object bị trùng
                for oid in duplicates.keys():
                    duplicate_objects = [obj for obj in objects if obj.get('object_id') == oid]
                    print(f"    Chi tiết object_id {oid}:")
                    for j, obj in enumerate(duplicate_objects):
                        names = obj.get('names', ['N/A'])
                        bbox = f"({obj.get('x', 'N/A')}, {obj.get('y', 'N/A')}, {obj.get('w', 'N/A')}, {obj.get('h', 'N/A')})"
                        print(f"      Object {j+1}: {names[0] if names else 'N/A'} - bbox: {bbox}")
                print()
        
        # Tổng kết
        print("=" * 60)
        print("KẾT QUẢ KIỂM TRA:")
        print(f"- Số ảnh có object_id bị trùng: {images_with_duplicates}")
        print(f"- Tổng số object bị trùng: {total_duplicates}")
        print(f"- Tỷ lệ ảnh có lỗi: {images_with_duplicates/len([img for img in data if img]) * 100:.2f}%")
        
        if images_with_duplicates == 0:
            print("✅ Không có object_id nào bị trùng!")
        else:
            print("❌ Phát hiện có object_id bị trùng!")
            
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file: {json_file_path}")
    except json.JSONDecodeError:
        print(f"❌ File JSON không hợp lệ: {json_file_path}")
    except Exception as e:
        print(f"❌ Lỗi khi xử lý file: {e}")

def check_all_json_files():
    """
    Kiểm tra tất cả các file JSON trong thư mục dataset
    """
    import os
    
    json_files = [
        'coco_uitvic_test.json',
        'coco_uitvic_train.json'
    ]
    
    dataset_dir = 'dataset'
    
    for json_file in json_files:
        file_path = os.path.join(dataset_dir, json_file)
        if os.path.exists(file_path):
            print(f"\n{'='*80}")
            print(f"KIỂM TRA FILE: {json_file}")
            print(f"{'='*80}")
            check_duplicate_object_ids(file_path)
        else:
            print(f"⚠️  File không tồn tại: {file_path}")

if __name__ == "__main__":
    # Kiểm tra file test hiện tại
    test_file = "dataset/coco_uitvic_test.json"
    check_duplicate_object_ids(test_file)
    
    # Uncomment dòng dưới để kiểm tra tất cả files
    # check_all_json_files()