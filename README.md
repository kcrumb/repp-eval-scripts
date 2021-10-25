# Checklist
1. Delete frames outside a specific range: `preprocessing/delete_images.py` 
2. Add empty txt-file for frames that have no YOLO predictions within the sequence: `preprocessing/add_empty_textfiles.py`
3. Create pickle file from YOLO predictions: `preprocessing/yolo_to_repp_pickle.py`


# GIANA processing order
1. Separate frames and GT files and copy into new folder
2. Copy YOLO predictions into new folder and add empty files for empty predictions
3. Duplicate first frame, GT file and YOLO predictions
4. Create pickle file for YOLO predictions
5. Execute REPP RT
6. Execute REPP
7. Filter YOLO predictions that are below threshold
8. Delete first n files from REPP RT, REPP and YOLO predictions
9. Delete all predictions that have no corresponding GT file an add empty files prediction files
10. Execute object detection metric

