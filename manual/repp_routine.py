"""
Executing the original REPP with the pickle file which contains all predictions.
The output is a new bounding box COCO JSON which is then converted into individual files
for each image into an separated folder.
"""

import json
import subprocess
import sys
import os
from pathlib import Path

# config
repp_project_path = Path('D:/Development/Projects/repp')
repp_config = Path(repp_project_path, 'REPP_cfg/yolo_repp_cfg.json')
prediction_file = Path('D:/Development/REPP_videos/uniklinikum-endo_cr_6/endo_cr_6.pckl')
# prediction_file = Path('D:/Development/REPP/test/repp_ci_3.pkl')


# change working directory to REPP project
os.chdir(repp_project_path)

pred_filename = prediction_file.name

# generated coco file
coco_json_file = Path(prediction_file.parent, pred_filename.replace('.pckl', '_repp') + '_coco.json')
# folder for predications files => name of predication file
out_folder = Path(prediction_file.parent, 'repp_predictions(0.2)')
# out_folder = Path(prediction_file.parent, pred_filename[: pred_filename.index('.')])


def execute_repp():
    """Execute original REPP"""
    subprocess.call(
        ['D:/Development/Projects/repp/venv/Scripts/python.exe',
         'D:/Development/Projects/repp/REPP.py',
         '--repp_cfg', str(repp_config),
         '--predictions_file', str(prediction_file),
         '--store_coco'])


def process_json(repp_json: list, output_folder: Path):
    """Convert REPP JSON into individual files for evaluation"""
    predictions = {}
    for p in repp_json:
        value = predictions[p['image_id']] if p['image_id'] in predictions else []
        value.append(p)
        predictions[p['image_id']] = value
    for k in predictions.keys():
        with open(Path(output_folder, '{}.txt'.format(k)), mode='w') as f:
            for p in predictions[k]:
                f.write('{class_name} {conf} {left} {top} {right} {bottom}\n'.format(
                    class_name='polyp', conf=p['score'],
                    left=p['bbox'][0], top=p['bbox'][1],
                    right=p['bbox'][0] + p['bbox'][2],
                    bottom=p['bbox'][1] + p['bbox'][3]))


if __name__ == '__main__':
    # execute
    print('> Execute REPP')
    execute_repp()

    # load coco json
    print('> Open REPP COCO JSON file:', coco_json_file)
    with open(coco_json_file, mode='r') as f:
        repp_org = json.load(f)

    # create output folder and convert
    if not out_folder.exists():
        out_folder.mkdir()
        print('> Folder created', out_folder)
        print('> Converting REPP JSON to individual files')
        process_json(repp_json=repp_org, output_folder=out_folder)
        print('> Done Converting')
    else:
        print('{} already exists'.format(out_folder), file=sys.stderr)
