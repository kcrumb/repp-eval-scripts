"""
Filter YOLO predictions an copy them into a new folder
"""

from ast import literal_eval
from pathlib import Path
from tqdm import tqdm


threshold = 0.2
video_yolo_predictions = [
    Path('D:/Development/REPP_videos/uniklinikum-endo_ci_3/yolo_predictions'),
    Path('D:/Development/REPP_videos/uniklinikum-endo_cj_4/yolo_predictions'),
    Path('D:/Development/REPP_videos/uniklinikum-endo_co_3/yolo_predictions'),
    Path('D:/Development/REPP_videos/uniklinikum-endo_co_5/yolo_predictions'),
    Path('D:/Development/REPP_videos/uniklinikum-endo_co_6/yolo_predictions'),
    Path('D:/Development/REPP_videos/uniklinikum-endo_cr_1/yolo_predictions'),
    Path('D:/Development/REPP_videos/uniklinikum-endo_cr_3/yolo_predictions'),
    Path('D:/Development/REPP_videos/uniklinikum-endo_cr_4/yolo_predictions'),
    Path('D:/Development/REPP_videos/uniklinikum-endo_cr_5/yolo_predictions'),
    Path('D:/Development/REPP_videos/uniklinikum-endo_cr_6/yolo_predictions')
]
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
    for prediction_folder in tqdm(video_yolo_predictions):
        # create new folder
        threshold_folder = Path(prediction_folder.parent, 'yolo_predictions({})'.format(threshold))
        threshold_folder.mkdir()

        for prediction_file in tqdm(prediction_folder.iterdir()):
            # open file and get yolo predictions
            with open(prediction_file, mode='r') as f:
                predication_lines = [convert_line_to_tuple(line) for line in f.read().splitlines()]

            lines = []
            for line in predication_lines:
                class_id, rel_x_center, rel_y_center, rel_box_width, rel_box_height, confidence = line
                # skip if box is below threshold
                if confidence < threshold:
                    continue
                lines.append('polyp {confidence} {left} {top} {right} {bottom}'.format(
                    confidence=confidence,
                    left=(rel_x_center * image_width) - ((rel_box_width * image_width) / 2),
                    top=(rel_y_center * image_height) - ((rel_box_height * image_height) / 2),
                    right=(rel_x_center * image_width) + ((rel_box_width * image_width) / 2),
                    bottom=(rel_y_center * image_height) + ((rel_box_height * image_height) / 2)))

            # write filtered and converted yolo predictions to new folder
            with open(Path(threshold_folder, prediction_file.name), mode='w') as file:
                file.write('\n'.join(lines))

