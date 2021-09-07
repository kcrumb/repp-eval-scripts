"""
Convert YOLO predictions to a pickle file for REPP.
"""

from ast import literal_eval
from pathlib import Path
from tqdm import tqdm
import pickle

# config
video_name = 'endo_cr_6'
pickle_output = Path('D:/Development/REPP_videos/uniklinikum-{}'.format(video_name))
predications_folder = Path(pickle_output, 'yolo_predictions')
image_height = 1080
image_width = 1920


def convert_line_to_tuple(line: str) -> tuple:
    """Converts a string line into a tuple of number values."""
    converted_line = []
    for value in line.split(' '):
        if isinstance(literal_eval(value), int):
            converted_line.append(int(value))
        elif isinstance(literal_eval(value), float):
            converted_line.append(float(value))
        else:
            converted_line.append(value)
    return tuple(converted_line)
            

if __name__ == '__main__':
    predictions_video = {}
    
    prediction_files = [f for f in predications_folder.iterdir()]
    
    for file in tqdm(prediction_files):
        frame_number = int(file.name[:file.name.rindex('.')])

        # open file and get yolo predictions
        with open(file, mode='r') as f:
            predication_lines = [convert_line_to_tuple(line) for line in f.read().splitlines()]

        # convert to coco box format
        prediction_frames = []
        for line in predication_lines:
            class_id, rel_x_center, rel_y_center, rel_box_width, rel_box_height, confidence = line
            box_x = (rel_x_center * image_width) - ((rel_box_width * image_width) / 2)
            box_y = (rel_y_center * image_height) - ((rel_box_height * image_height) / 2)
            box_width = rel_box_width * image_width
            box_height = rel_box_height * image_height

            detection = {'image_id': frame_number,
                         'bbox': [box_x, box_y, box_width, box_height],
                         'bbox_center': [rel_x_center, rel_y_center],
                         'scores': [confidence],
                         'category_id': 0}
            prediction_frames.append(detection)
        predictions_video[frame_number] = prediction_frames
        
    # video finished, dumping pickle file
    with open(Path(pickle_output, '{}.pckl'.format(video_name)), mode='wb') as file_writer:
        pickle.dump((video_name, predictions_video), file_writer)