"""
Make a copy of your data before running this script. This script will delete and add files from the folder structure.

----------------------------------------------------------------------------------------------------

Expected structure:
    - root_video_folder/frames

    - root_video_folder/gt

      - expected format: <class-id> <rel-x-center> <rel-y-center> <rel-width> <rel-height> <confidence>

    - root_video_folder/yolo_predictions(raw)

      - expected format: <class-id> <rel-x-center> <rel-y-center> <rel-width> <rel-height> <confidence>

----------------------------------------------------------------------------------------------------

Preprocessing:
    1) Separate frames and ground truth files into evaluation folder
    2) Coping YOLO predictions into evaluation folder

"""
import json
import os
import pickle
import shutil
import subprocess
import sys
from ast import literal_eval
from pathlib import Path
# REPP config
from typing import Tuple

from PIL import Image
from tqdm import tqdm

# ==============================
# =========== CONFIG ===========
# ==============================

repp_project_path = Path('D:/Development/Projects/repp')
repp_python_executable = Path(repp_project_path, 'venv/Scripts/python.exe')
# Object Detection Metrics config
odm_project_path = Path('D:/Development/Projects/Object-Detection-Metrics-master')
odm_python_executable = Path(odm_project_path, 'venv/Scripts/python.exe')
# Video evaluation config
sequence_numbers = [1]
repp_rt_window_sizes = [100, 200]
excel_file_output = Path('D:/Development/REPP_TEST')
giana_source = Path('D:/Development/GIANA_videos/GIANA_frames_gt')
giana_yolo_predictions = Path('D:/Development/GIANA_videos/GIANA_yolo_predictions')
first_frame_duplicates = 1000


def separate_frames_gt(source_sequence_folder: Path) -> Tuple[bool, bool]:
    """Separate the frame and the gt-files into two folder named 'frames' and 'gt'.
    """
    print('> Separate frames and ground truth files')
    frame_folder = Path(giana_source.parent, 'GIANA_evaluation', source_sequence_folder.name, 'frames')
    print('>> New frame folder:', str(frame_folder))
    gt_folder = Path(giana_source.parent, 'GIANA_evaluation', source_sequence_folder.name, 'gt')
    print('>> New GT folder:', str(gt_folder))

    frame_folder_exists = frame_folder.exists()
    if not frame_folder_exists:
        frame_folder.mkdir(parents=True)
    else:
        print('>> Skip: Frame folder already exists.')
    gt_folder_exists = gt_folder.exists()
    if not gt_folder_exists:
        gt_folder.mkdir(parents=True)
    else:
        print('>> Skip: GT folder already exists.')

    if frame_folder_exists and gt_folder_exists:
        return frame_folder_exists, gt_folder_exists

    for file in tqdm(source_sequence_folder.iterdir()):
        if file.suffix == '.png' and not frame_folder_exists:
            shutil.copy(file, Path(frame_folder, file.name))
        elif file.suffix == '.txt' and not gt_folder_exists:
            # TODO: convert into VOC format
            shutil.copy(file, Path(gt_folder, file.name))
        elif (not file.suffix == '.png' and not frame_folder_exists) and (
                not file.suffix == '.txt' and not gt_folder_exists):
            print('The following file cannot be mapped', str(file), file=sys.stderr)
    print('>> Done separating')
    return frame_folder_exists, gt_folder_exists


def duplicate_first(folder: Path):
    """Duplicates the first frame n-times and renumbering the rest
    """
    print('> Duplicate first frame {} times'.format(first_frame_duplicates))

    print('>> Renumber frames by adding ', first_frame_duplicates)
    # rename frames
    for frame_file in folder.iterdir():
        frame_number = int(frame_file.name[:frame_file.name.rindex('.')])
        frame_number += first_frame_duplicates
        frame_file.rename(Path(frame_file.parent, str(frame_number) + frame_file.suffix))
    print('>> Done renumbering')

    print('>> Duplicate first frame')
    frames = [f for f in folder.iterdir()]
    frames.sort(key=lambda f: int(f.name[:f.name.rindex('.')]))
    first_frame = frames[0]
    for i in tqdm(range(1, first_frame_duplicates + 1)):
        shutil.copy(first_frame, Path(first_frame.parent, str(i) + first_frame.suffix))
    print('>> Done duplicating')


def delete_first(folder: Path):
    """Deletes the first n files and renumbering the rest
    """
    print('> Delete first {} files'.format(first_frame_duplicates))

    files = [f for f in folder.iterdir()]
    files.sort(key=lambda f: int(f.name[:f.name.rindex('.')]))
    for f in files:
        filename_number = int(f.name[:f.name.rindex('.')])
        if filename_number <= first_frame_duplicates:
            f.unlink()


def copy_yolo_predictions(sequence_folder: Path) -> bool:
    """Get yolo predictions and copy them into the video sequence folder and add missing prediction files
    """
    print('> Start coping YOLO predictions for', str(sequence_folder))
    # copy files with specific video number into separate folders for each video
    sequence_number = sequence_folder.name
    video_sequence_predictions = [f for f in giana_yolo_predictions.iterdir() if
                                  f.name.split('-')[0] == sequence_number]
    sequence_prediction_folder = Path(sequence_folder, 'yolo_predictions(raw)')

    yolo_raw_folder_exists = sequence_prediction_folder.exists()
    if yolo_raw_folder_exists:
        print('>> Skip: YOLO predictions folder already exists.')
        return yolo_raw_folder_exists

    print('>> Copy YOLO predictions into sequence folder')
    sequence_prediction_folder.mkdir()
    for file in tqdm(video_sequence_predictions):
        filename = file.name.split('-')[1]
        shutil.copy(file, Path(sequence_prediction_folder, filename))

    print('>> Add empty txt-files for missing predictions')
    # add empty txt if there is no prediction file for a frame
    video_frames_folder = Path(sequence_folder, 'frames')
    video_frames = set([int(f.name[:f.name.rindex('.')]) for f in video_frames_folder.iterdir()])
    prediction_files = set([int(f.name[:f.name.rindex('.')]) for f in sequence_prediction_folder.iterdir()])
    missing_files = video_frames - prediction_files
    for file_number in tqdm(missing_files, position=0):
        Path(sequence_prediction_folder, '{}.txt'.format(file_number)).touch()
    print('>> DONE coping')
    return yolo_raw_folder_exists


def pickle_predictions(sequence_folder: Path):
    """Takes all prediction files from YOLO, converts them into COCO-format and saves them into a pickle file.
    """

    def convert_line_to_tuple(line: str) -> tuple:
        """Converts a string line into a tuple of number values.
        """
        converted_line = []
        for value in line.split(' '):
            if isinstance(literal_eval(value), int):
                converted_line.append(int(value))
            elif isinstance(literal_eval(value), float):
                converted_line.append(float(value))
            else:
                converted_line.append(value)
        return tuple(converted_line)

    print('> Create pickle from YOLO predictions')
    pickle_output = Path(sequence_folder, '{}.pckl'.format(sequence_folder.name))

    # check if pickle file already exists#
    if pickle_output.exists():
        print('>> Skip: pickle file already exists', str(pickle_output))
        return

    # get image resolution by looking at the fir image in folder, which then counts for all frames/images
    print('>> Reading resolution of first frame image...')
    frames_folder = Path(sequence_folder, 'frames')
    first_frame = Image.open(next(frames_folder.iterdir()))
    image_width, image_height = first_frame.size
    print('\t>> {} x {}'.format(image_width, image_height))

    # go through all predictions txt-files and save them into a pickle file
    predications_folder = Path(sequence_folder, 'yolo_predictions(raw)')
    predictions_video = {}
    prediction_files = [f for f in predications_folder.iterdir()]

    for file in tqdm(prediction_files, position=0):
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
    with open(pickle_output, mode='wb') as file_writer:
        pickle.dump((sequence_folder.name, predictions_video), file_writer)
    print('>> Done! Saved pickle file to', str(pickle_output))


def execute_repp_rt(sequence_folder: Path):
    """Execute real-time REPP with different window sizes.
    """
    print('> Start executing REPP RT')
    # repp_config = Path(repp_project_path, 'REPP_cfg/yolo_repp_cfg.json')
    repp_config = Path('yolo_repp_cfg.json').resolve()
    repp_rt_script = Path(repp_project_path, 'REPP_RT.py')
    pickle_prediction_file = Path(sequence_folder, '{}.pckl'.format(sequence_folder.name))
    frames_folder = Path(sequence_folder, 'frames')

    # change working directory to REPP project
    this_working_dir = os.getcwd()
    print('>> Change working directory to REPP project', repp_project_path)
    os.chdir(repp_project_path)

    # define repp_rt_predictions path, create and get defined predictions score
    repp_rt_prediction_folder = Path(sequence_folder, 'repp_rt_predictions')
    repp_rt_prediction_folder.mkdir(parents=True, exist_ok=True)
    with open(repp_config, mode='r') as f:
        repp_config_json = json.load(f)
    min_pred_score = repp_config_json['min_pred_score']
    repp_rt_prediction_folder = Path(repp_rt_prediction_folder, 'threshold_{}'.format(min_pred_score))
    repp_rt_prediction_folder.mkdir(parents=True, exist_ok=True)

    # iterate over all defined window sizes
    for window_size in repp_rt_window_sizes:
        print('>> Execute REPP RT with window size', window_size)
        repp_rt_window_prediction_folder = Path(repp_rt_prediction_folder, 'ws_{}'.format(window_size))

        # check if existing
        if repp_rt_window_prediction_folder.exists():
            print('>> Delete window prediction folder for new predictions:', str(repp_rt_window_prediction_folder))
            for f in repp_rt_window_prediction_folder.iterdir():
                f.unlink()
            repp_rt_window_prediction_folder.rmdir()

        # execute REPP_RT script
        subprocess.call([str(repp_python_executable),
                         str(repp_rt_script),
                         '--repp_cfg', str(repp_config),
                         '--predictions_file', str(pickle_prediction_file),
                         '--store_coco',
                         '--store', str(repp_rt_window_prediction_folder),
                         '--window_size', str(window_size),
                         '--frames', str(frames_folder)])

    # change working directory back
    print('>> Done. Change working directory back to', str(this_working_dir))
    os.chdir(this_working_dir)


def execute_repp(sequence_folder: Path):
    """Execute original REPP
    """

    def convert_json_to_individual_files(coco_json_file: Path, repp_config_path: Path):
        """Convert REPP JSON into individual files for evaluation
        """
        print('>> Load REPP COCO JSON file and convert it to individual files')

        # get minimum prediction score
        with open(repp_config_path, mode='r') as f:
            repp_config_json = json.load(f)
        min_pred_score = repp_config_json['min_pred_score']

        # create folder to save individual file or delete folder if existing
        repp_predictions_output_folder = Path(sequence_folder, 'repp_predictions',
                                              'threshold_{}'.format(min_pred_score))
        if repp_predictions_output_folder.exists():
            print('>> Delete REPP prediction folder for new predictions:', str(repp_predictions_output_folder))
            for f in repp_predictions_output_folder.iterdir():
                f.unlink()
            repp_predictions_output_folder.rmdir()
        repp_predictions_output_folder.mkdir(parents=True)

        # get generate json coco file and load it
        with open(coco_json_file, mode='r') as f:
            repp_json = json.load(f)

        predictions = {}
        for p in repp_json:
            value = predictions[p['image_id']] if p['image_id'] in predictions else []
            value.append(p)
            predictions[p['image_id']] = value
        for k in predictions.keys():
            with open(Path(repp_predictions_output_folder, '{}.txt'.format(k)), mode='w') as f:
                for p in predictions[k]:
                    f.write('{class_name} {conf} {left} {top} {right} {bottom}\n'.format(
                        class_name='polyp', conf=p['score'],
                        left=p['bbox'][0], top=p['bbox'][1],
                        right=p['bbox'][0] + p['bbox'][2],
                        bottom=p['bbox'][1] + p['bbox'][3]))
        print('>> Done converting')

    print('> Start executing REPP')
    repp_script = Path(repp_project_path, 'REPP.py')
    repp_config = Path('yolo_repp_cfg.json').resolve()
    pickle_prediction_file = Path(sequence_folder, '{}.pckl'.format(sequence_folder.name))

    # change working directory to REPP project
    this_working_dir = os.getcwd()
    print('>> Change working directory to REPP project', repp_project_path)
    os.chdir(repp_project_path)

    # execute REPP script
    subprocess.call([str(repp_python_executable),
                     str(repp_script),
                     '--repp_cfg', str(repp_config),
                     '--predictions_file', str(pickle_prediction_file),
                     '--store_coco'])

    repp_coco_json_file = Path(sequence_folder, sequence_folder.name + '_repp_coco.json')
    # convert repp predictions to files
    convert_json_to_individual_files(repp_coco_json_file, repp_config)
    # delete json
    print('>> Delete REPP JSON', str(repp_coco_json_file))
    repp_coco_json_file.unlink()

    # change working directory back
    print('>> Done. Change working directory back to', str(this_working_dir))
    os.chdir(this_working_dir)


def filter_raw_yolo_predictions(sequence_folder: Path):
    """Goes through the raw YOLO predictions and filters out all predicted boxes that are below the minimum prediction
    score (threshold)
    """

    def convert_line_to_tuple(line: str) -> tuple:
        """Converts a string line into a tuple of number values.
        """
        converted_line = []
        for value in line.split(' '):
            if isinstance(literal_eval(value), int):
                converted_line.append(int(value))
            elif isinstance(literal_eval(value), float):
                converted_line.append(float(value))
            else:
                converted_line.append(value)
        return tuple(converted_line)

    print('> Start filtering YOLO predictions')

    yolo_prediction_folder = Path(sequence_folder, 'yolo_predictions(raw)')

    # get image resolution by looking at the fir image in folder, which then counts for all frames/images
    print('>> Reading resolution of first frame image...')
    frames_folder = Path(sequence_folder, 'frames')
    first_frame = Image.open(next(frames_folder.iterdir()))
    image_width, image_height = first_frame.size
    print('\t>> {} x {}'.format(image_width, image_height))

    # get minimum prediction score
    min_pred_score = get_min_prediction_score()

    # create new folder or delete if it exists
    threshold_folder = Path(sequence_folder, 'yolo_predictions', 'threshold_{}'.format(min_pred_score))
    if threshold_folder.exists():
        print('>> Delete YOLO predictions threshold folder for new filtering:', str(threshold_folder))
        for f in threshold_folder.iterdir():
            f.unlink()
        threshold_folder.rmdir()
    threshold_folder.mkdir(parents=True)

    for prediction_file in tqdm(yolo_prediction_folder.iterdir(), position=0):
        # open file and get yolo predictions
        with open(prediction_file, mode='r') as f:
            predication_lines = [convert_line_to_tuple(line) for line in f.read().splitlines()]

        lines = []
        for line in predication_lines:
            class_id, rel_x_center, rel_y_center, rel_box_width, rel_box_height, confidence = line
            # skip if box is below threshold
            if confidence < min_pred_score:
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
    print('>> Done filtering YOLO predictions')


def get_min_prediction_score() -> float:
    """Helper Function: Get defined minimum prediction score in REPP JSON.
    """
    repp_config = Path('yolo_repp_cfg.json').resolve()
    with open(repp_config, mode='r') as file:
        repp_config_json = json.load(file)
    return repp_config_json['min_pred_score']


def duplicate_first_files(eval_sequence_folder: Path, frame_folder_exists: bool, gt_folder_exists: bool,
                          yolo_raw_folder_exists: bool):
    # duplicate first frame
    if not frame_folder_exists:
        print('>> Duplicate first frame')
        duplicate_first(folder=Path(eval_sequence_folder, 'frames'))
    # duplicate first gt file
    if not gt_folder_exists:
        print('>> Duplicate first GT file')
        duplicate_first(folder=Path(eval_sequence_folder, 'gt'))
    # duplicate first yolo raw prediction file
    if not yolo_raw_folder_exists:
        print('>> Duplicate first raw YOLO prediction file')
        duplicate_first(folder=Path(eval_sequence_folder, 'yolo_predictions(raw)'))


def delete_first_files_from_folders(eval_sequence_folder: Path):
    print('> Delete files until', first_frame_duplicates)
    min_pred_score = get_min_prediction_score()
    repp_rt_threshold_folder = Path(eval_sequence_folder, 'repp_rt_predictions', 'threshold_{}'.format(min_pred_score))
    for ws_folder in repp_rt_threshold_folder.iterdir():
        print('>> Delete files from', str(ws_folder))
        delete_first(folder=ws_folder)

    repp_threshold_folder = Path(eval_sequence_folder, 'repp_predictions', 'threshold_{}'.format(min_pred_score))
    print('>> Delete files from', str(repp_threshold_folder))
    delete_first(folder=repp_threshold_folder)

    yolo_threshold_folder = Path(eval_sequence_folder, 'yolo_predictions', 'threshold_{}'.format(min_pred_score))
    print('>> Delete files from', str(yolo_threshold_folder))
    delete_first(folder=yolo_threshold_folder)


def main():
    for seq_number in sequence_numbers:
        source_sequence_folder = Path(giana_source, str(seq_number))
        eval_sequence_folder = Path(giana_source.parent, 'GIANA_evaluation', str(seq_number))

        frame_folder_exists, gt_folder_exists = separate_frames_gt(source_sequence_folder=source_sequence_folder)
        yolo_raw_folder_exists = copy_yolo_predictions(sequence_folder=eval_sequence_folder)

        duplicate_first_files(eval_sequence_folder, frame_folder_exists, gt_folder_exists, yolo_raw_folder_exists)

        pickle_predictions(sequence_folder=eval_sequence_folder)
        execute_repp_rt(sequence_folder=eval_sequence_folder)
        execute_repp(sequence_folder=eval_sequence_folder)
        filter_raw_yolo_predictions(sequence_folder=eval_sequence_folder)

        delete_first_files_from_folders(eval_sequence_folder)


if __name__ == '__main__':
    main()
