"""
Copy and separate ground truth files into the videos folders.
"""

from ast import literal_eval
from tqdm import tqdm
from pathlib import Path

gt_folder = Path('D:/Development/REPP_videos/validation_1000')
root_video_folder = Path('D:/Development/REPP_videos')
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


def convert_gt_data(gt_file: Path) -> str:
    # open file and get gt data
    with open(gt_file, mode='r') as f:
        gt_lines = [convert_line_to_tuple(line) for line in f.read().splitlines()]
    lines = []
    for line in gt_lines:
        class_id, rel_x_center, rel_y_center, rel_box_width, rel_box_height, = line
        lines.append('polyp {left} {top} {right} {bottom}'.format(
            left=(rel_x_center * image_width) - ((rel_box_width * image_width) / 2),
            top=(rel_y_center * image_height) - ((rel_box_height * image_height) / 2),
            right=(rel_x_center * image_width) + ((rel_box_width * image_width) / 2),
            bottom=(rel_y_center * image_height) + ((rel_box_height * image_height) / 2)))
    return '\n'.join(lines)


if __name__ == '__main__':
    # get video names
    for gt_file in tqdm(gt_folder.iterdir()):
        if gt_file.name == 'classes.txt' or not gt_file.name.endswith('.txt'):
            continue

        video_gt_folder = Path(root_video_folder, gt_file.name[:21], 'gt')
        if not video_gt_folder.exists():
            video_gt_folder.mkdir()

        frame_number_filename = gt_file.name[gt_file.name.rindex('_') + 1:]
        with open(Path(video_gt_folder, frame_number_filename), mode='w') as video_gt_file:
            video_gt_file.write(convert_gt_data(gt_file))
