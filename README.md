# Checklist
1. Delete frames outside a specific range: `preprocessing/delete_images.py` 
2. Add empty txt-file for frames that have no YOLO predictions within the sequence: `preprocessing/add_empty_textfiles.py`
3. Create pickle file from YOLO predictions: `preprocessing/yolo_to_repp_pickle.py`
