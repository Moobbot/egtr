import json
from collections import defaultdict

def fix_duplicate_object_ids(json_file_path, output_file_path=None):
    """
    Sửa lỗi object_id bị trùng bằng cách gán lại ID duy nhất
    """
    try:
        # Đọc file JSON
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if output_file_path is None:
            output_file_path = json_file_path.replace('.json', '_fixed.json')
        
        print(f"Đang sửa file: {json_file_path}")
        print(f"File output: {output_file_path}")
        print("=" * 60)
        
        fixed_images = 0
        
        # Sửa từng ảnh
        for i, image_data in enumerate(data):
            if not image_data or 'image_id' not in image_data:
                continue
                
            image_id = image_data['image_id']
            objects = image_data.get('objects', [])
            relationships = image_data.get('relationships', [])
            
            if not objects:
                continue
            
            # Đếm số lần xuất hiện của mỗi object_id
            object_id_count = defaultdict(list)
            
            for obj_idx, obj in enumerate(objects):
                if 'object_id' in obj:
                    object_id_count[obj['object_id']].append(obj_idx)
            
            # Tìm các object_id bị trùng
            duplicates = {oid: indices for oid, indices in object_id_count.items() if len(indices) > 1}
            
            if duplicates:
                fixed_images += 1
                print(f"Sửa ảnh ID {image_id}:")
                
                # Tìm object_id mới không bị trùng
                used_ids = set(obj.get('object_id') for obj in objects if 'object_id' in obj)
                next_available_id = max(used_ids) + 1 if used_ids else 1
                
                # Mapping từ old_id sang new_id
                id_mapping = {}
                
                for old_id, obj_indices in duplicates.items():
                    # Giữ nguyên object đầu tiên, đổi ID cho các object sau
                    for j, obj_idx in enumerate(obj_indices[1:], 1):
                        old_object_id = objects[obj_idx]['object_id']
                        new_object_id = next_available_id
                        
                        # Cập nhật object_id
                        objects[obj_idx]['object_id'] = new_object_id
                        id_mapping[old_object_id] = new_object_id
                        
                        obj_name = objects[obj_idx].get('names', ['Unknown'])[0]
                        print(f"  - Object '{obj_name}' (idx {obj_idx}): {old_object_id} -> {new_object_id}")
                        
                        next_available_id += 1
                
                # Cập nhật relationships nếu có
                for rel in relationships:
                    if rel.get('subject_id') in id_mapping:
                        old_id = rel['subject_id']
                        rel['subject_id'] = id_mapping[old_id]
                        print(f"  - Cập nhật relationship subject_id: {old_id} -> {id_mapping[old_id]}")
                    
                    if rel.get('object_id') in id_mapping:
                        old_id = rel['object_id']
                        rel['object_id'] = id_mapping[old_id]
                        print(f"  - Cập nhật relationship object_id: {old_id} -> {id_mapping[old_id]}")
                
                print()
        
        # Lưu file đã sửa
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print("=" * 60)
        print("KẾT QUẢ SỬA LỖI:")
        print(f"- Số ảnh đã sửa: {fixed_images}")
        print(f"- File đã được lưu: {output_file_path}")
        
        if fixed_images > 0:
            print("✅ Đã sửa xong tất cả lỗi object_id trùng!")
            
            # Kiểm tra lại file đã sửa
            print("\nKiểm tra lại file đã sửa...")
            check_duplicate_object_ids_simple(output_file_path)
        else:
            print("ℹ️  Không có lỗi nào cần sửa.")
            
    except Exception as e:
        print(f"❌ Lỗi khi sửa file: {e}")

def check_duplicate_object_ids_simple(json_file_path):
    """
    Kiểm tra nhanh xem còn object_id trùng không
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        duplicate_count = 0
        
        for image_data in data:
            if not image_data or 'objects' not in image_data:
                continue
            
            objects = image_data['objects']
            object_ids = [obj.get('object_id') for obj in objects if 'object_id' in obj]
            
            if len(object_ids) != len(set(object_ids)):
                duplicate_count += 1
        
        if duplicate_count == 0:
            print("✅ Kiểm tra hoàn tất: Không còn object_id trùng!")
        else:
            print(f"❌ Vẫn còn {duplicate_count} ảnh có object_id trùng!")
            
    except Exception as e:
        print(f"❌ Lỗi khi kiểm tra: {e}")

if __name__ == "__main__":
    # Sửa file test
    input_file = "dataset/coco_uitvic_test.json"
    output_file = "dataset/coco_uitvic_test_fixed.json"
    
    fix_duplicate_object_ids(input_file, output_file)