"""
Make a copy of your data before running this script. This script will delete and add files from the folder structure.

----------------------------------------------------------------------------------------------------

Expected structure:
    - root_video_folder/frames
    - root_video_folder/gt
    - root_video_folder/yolo_predictions(raw)

Folder explanation:
    - frames: contains all frames; some will be deleted if frame does not have a prediction

    - gt: contains ground truth files; empty txt-file for frames that have no YOLO predictions within the sequence will be added

    - yolo_predictions(raw): contains txt-files with the predictions from the YOLO network; should be on sequence with consecutive frames numbers

----------------------------------------------------------------------------------------------------

Preprocessing:
    1) Delete frames that are outside from the specific range where no predictions exists
    2) Add empty txt-file for frames that have no YOLO predictions within the sequence
    3) Create pickle file from YOLO predictions for REPP and REPP_RT

Execution:
    1) REPP_RT with all defined windows sizes
    2) REPP
    3) Pure YOLO

Postprocessing:
    1) ...

"""

import json
import os
import pickle
import subprocess
from ast import literal_eval
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from tqdm import tqdm

# ========== CONFIG ==========
# REPP config
repp_project_path = Path('D:/Development/Projects/repp')
repp_python_executable = Path(repp_project_path, 'venv/Scripts/python.exe')
# Object Detection Metrics config
odm_project_path = Path('D:/Development/Projects/Object-Detection-Metrics-master')
odm_python_executable = Path(odm_project_path, 'venv/Scripts/python.exe')
# Video evaluation config
video_folders = [
    'D:/Development/REPP_TEST/uniklinikum-endo_ci_3',
    'D:/Development/REPP_TEST/uniklinikum-endo_cj_4'
]
repp_rt_window_sizes = [100, 200]
excel_file_output = Path('D:/Development/REPP_TEST')


def delete_frames_outside_range(video_folder: Path):
    """Deletes frames that are outside from the predictions frame sequence.
    """
    print('> Delete frame files outside of range')
    frame_folder = Path(video_folder, 'frames')
    # get range from prediction folder
    prediction_folder = Path(video_folder, 'yolo_predictions(raw)')
    predicted_frame_numbers = [int(f.name[:f.name.rindex('.')]) for f in prediction_folder.iterdir()]
    predicted_frame_numbers.sort()
    range_start = predicted_frame_numbers[0]
    range_stop = predicted_frame_numbers[-1]
    # get all files and delete all files that should stay
    frames = [f for f in frame_folder.iterdir() if
              int(f.name[:f.name.rindex('.')]) not in range(range_start, range_stop + 1)]
    # delete files that are outside of range
    for f in tqdm(frames, position=0):
        f.unlink()
    print('>> Done! Deleted', len(frames), 'frames')


def add_empty_txt_to_predictions(video_folder: Path):
    """Adds empty txt-files to predictions folder to fill the gaps where no prediction is present within the sequence.
    """
    print('> Create empty files for missing numbers')
    predictions_folder = Path(video_folder, 'yolo_predictions(raw)')
    # get all frame numbers that have a prediction and sort them
    file_numbers = [int(f.name[:f.name.rindex('.')]) for f in predictions_folder.iterdir() if f.name.endswith('.txt')]
    file_numbers.sort()
    # generate a set with all number within the range of frame numbers
    all_numbers = set([i for i in range(file_numbers[0], file_numbers[-1] + 1)])
    file_numbers = set(file_numbers)
    # get all numbers that are missing from prediction set
    missing_numbers = sorted(list(all_numbers - file_numbers))
    # create empty txt-files for missing numbers
    for m in tqdm(missing_numbers, position=0):
        Path(predictions_folder, '{}.txt'.format(m)).touch()
    print('>> Done! Added', len(missing_numbers), 'files')


def pickle_predictions(video_folder: Path):
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
    pickle_output = Path(video_folder, '{}.pckl'.format(video_folder.name))

    # check if pickle file already exists#
    if pickle_output.exists():
        print('>> Skip: pickle file already exists', str(pickle_output))
        return

    # get image resolution by looking at the fir image in folder, which then counts for all frames/images
    print('>> Reading resolution of first frame image...')
    frames_folder = Path(video_folder, 'frames')
    first_frame = Image.open(next(frames_folder.iterdir()))
    image_width, image_height = first_frame.size
    print('\t>> {} x {}'.format(image_width, image_height))

    # go through all predictions txt-files and save them into a pickle file
    predications_folder = Path(video_folder, 'yolo_predictions(raw)')
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
        pickle.dump((video_folder.name, predictions_video), file_writer)
    print('>> Done! Saved pickle file to', str(pickle_output))


def execute_repp_rt(video_folder: Path):
    """Execute real-time REPP with different window sizes.
    """
    print('> Start executing REPP RT')
    # repp_config = Path(repp_project_path, 'REPP_cfg/yolo_repp_cfg.json')
    repp_config = Path('yolo_repp_cfg.json').resolve()
    repp_rt_script = Path(repp_project_path, 'REPP_RT.py')
    pickle_prediction_file = Path(video_folder, '{}.pckl'.format(video_folder.name))
    frames_folder = Path(video_folder, 'frames')

    # change working directory to REPP project
    this_working_dir = os.getcwd()
    print('>> Change working directory to REPP project', repp_project_path)
    os.chdir(repp_project_path)

    # define repp_rt_predictions path, create and get defined predictions score
    repp_rt_prediction_folder = Path(video_folder, 'repp_rt_predictions')
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


def execute_repp(video_folder: Path):
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
        repp_predictions_output_folder = Path(video_folder, 'repp_predictions', 'threshold_{}'.format(min_pred_score))
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
    pickle_prediction_file = Path(video_folder, '{}.pckl'.format(video_folder.name))

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

    repp_coco_json_file = Path(video_folder, video_folder.name + '_repp_coco.json')
    # convert repp predictions to files
    convert_json_to_individual_files(repp_coco_json_file, repp_config)
    # delete json
    print('>> Delete REPP JSON', str(repp_coco_json_file))
    repp_coco_json_file.unlink()

    # change working directory back
    print('>> Done. Change working directory back to', str(this_working_dir))
    os.chdir(this_working_dir)


def filter_raw_yolo_predictions(video_folder: Path):
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

    yolo_prediction_folder = Path(video_folder, 'yolo_predictions(raw)')

    # get image resolution by looking at the fir image in folder, which then counts for all frames/images
    print('>> Reading resolution of first frame image...')
    frames_folder = Path(video_folder, 'frames')
    first_frame = Image.open(next(frames_folder.iterdir()))
    image_width, image_height = first_frame.size
    print('\t>> {} x {}'.format(image_width, image_height))

    # get minimum prediction score
    repp_config = Path('yolo_repp_cfg.json').resolve()
    with open(repp_config, mode='r') as f:
        repp_config_json = json.load(f)
    min_pred_score = repp_config_json['min_pred_score']

    # create new folder or delete if it exists
    threshold_folder = Path(video_folder, 'yolo_predictions', 'threshold_{}'.format(min_pred_score))
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


def filter_relevant_predictions(video_folder: Path):
    """Delete all predictions files that do not have a correspondent GT file.
    Additionally add empty txt-file if there is an GT file but no prediction file
    """
    print('> Start filtering predictions')

    gt_folder = Path(video_folder, 'gt')
    gt_files = set([f.name for f in gt_folder.iterdir()])
    min_pred_score = get_min_prediction_score()
    prediction_folders = get_prediction_folders(video_folder=video_folder, min_pred_score=min_pred_score)

    print('>> Delete frame files that hat no ground truth')
    # go through each folder and delete file if there is no corresponding GT file
    for folder in tqdm(prediction_folders, position=0):
        # iterate over all predictions and check if a ground truth exists
        for prediction_file in tqdm(folder.iterdir(), position=0):
            # delete file if no ground truth for that frame exists
            if prediction_file.name not in gt_files:
                prediction_file.unlink()

    print('>> Add empty files to prediction if missing')
    # add empty txt file if there was no prediction file to a GT file
    for folder in tqdm(prediction_folders, position=0):
        prediction_files = set([f.name for f in folder.iterdir()])
        missing_prediction_files = gt_files - prediction_files
        for file in tqdm(missing_prediction_files, position=0):
            Path(folder, file).touch()

    print('>> Done filtering predictions')


def execute_object_detection_metrics(video_folder: Path):
    """Execute evaluation of REPP, REPP RT and YOLO (with threshold)
    """

    print('> Start metric calculation of REPP, REPP RT and YOLO')
    odm_script = Path(odm_project_path, 'pascalvoc.py')
    gt_path = Path(video_folder, 'gt')

    # get minimum prediction score, which is used the to find the correct prediction folder
    min_pred_score = get_min_prediction_score()
    prediction_folders = get_prediction_folders(video_folder=video_folder, min_pred_score=min_pred_score)

    print('>> Calculate metrics for following folders (threshold {}):'.format(min_pred_score))
    for folder in prediction_folders:
        print('\t', str(folder))

    # change working directory to ODM project
    this_working_dir = os.getcwd()
    print('>> Change working directory to REPP project', odm_project_path)
    os.chdir(odm_project_path)

    # iterate over the folders with predictions to evaluate
    for detection_path in prediction_folders:
        # create folder for evaluation result
        eval_save_path = Path(video_folder, 'eval', detection_path.relative_to(video_folder))
        eval_save_path.mkdir(parents=True, exist_ok=True)
        # check delete folder contents if evaluation files already exist
        print('>> Delete evaluation folder content for new evaluation:', str(eval_save_path))
        for eval_file in eval_save_path.iterdir():
            eval_file.unlink()

        subprocess.call([str(odm_python_executable),
                         str(odm_script),
                         '-det', str(detection_path),
                         '-gt', str(gt_path),
                         '-t', '0.5',
                         '--savepath', str(eval_save_path),
                         '--noplot'])

    # change working directory back
    print('>> Done. Change working directory back to', str(this_working_dir))
    os.chdir(this_working_dir)


def gather_evaluation_results():
    """Gathers all evaluation results from a specified threshold, calculate F1-score
    and generates a table within a excel file.
    """

    def get_map_from_file(results_file: Path) -> float:
        """Search for the mAP value in the evalu results file (mostly at the bottom), parses the value
        and converts the percentage value in to a value between 0 and 1.
        """
        with open(results_file, mode='r') as f:
            lines = f.readlines()
        map_line = ''
        for line in reversed(lines):
            line = line.strip()
            if line.startswith('mAP'):
                map_line = line
                break
        # parse mAP value
        return float(map_line.split(':')[1].strip()[:-1]) / 100.0

    def calculate_f1(results_file: Path) -> float:
        """Calculate the F1 score with the precision and recall values from the results file.
        """
        with open(results_file, mode='r') as f:
            lines = f.readlines()
        # read precision and recall
        precision, recall = -1, -1
        for line in lines:
            if line.startswith('Precision:'):
                line = line.replace('Precision: ', '')
                precision = np.asarray(literal_eval(line), dtype=np.double)
            if line.startswith('Recall:'):
                line = line.replace('Recall: ', '')
                recall = np.asarray(literal_eval(line), dtype=np.double)

        precision = precision + 0.00000000001
        recall = recall + 0.00000000001
        return max(2 * (precision * recall) / (precision + recall))

    import xlsxwriter

    min_pred_score = get_min_prediction_score()

    excel_file_path = Path(excel_file_output, 'repp_eval(threshold_{}).xlsx'.format(min_pred_score))
    workbook = xlsxwriter.Workbook(str(excel_file_path))
    worksheet = workbook.add_worksheet('threshold {}'.format(min_pred_score))

    # create header
    algo_header_format = workbook.add_format({'bold': True, 'align': 'center', 'left': 1})
    map_header_format = workbook.add_format({'bold': True, 'align': 'center', 'bottom': 1, 'left': 1})
    f1_header_format = workbook.add_format({'bold': True, 'align': 'center', 'bottom': 1})
    worksheet.merge_range('B1:C1', 'YOLO', algo_header_format)
    worksheet.write_string('B2', 'mAP', map_header_format)
    worksheet.write_string('C2', 'F1', f1_header_format)
    worksheet.merge_range('D1:E1', 'REPP', algo_header_format)
    worksheet.write_string('D2', 'mAP', map_header_format)
    worksheet.write_string('E2', 'F1', f1_header_format)

    ws_col_index = 5
    ws_col_indices = {}
    for ws_size in repp_rt_window_sizes:
        worksheet.merge_range(0, ws_col_index, 0, ws_col_index + 1,
                              'REPP RT (WS {})'.format(ws_size), algo_header_format)
        worksheet.write_string(1, ws_col_index, 'mAP', map_header_format)
        worksheet.write_string(1, ws_col_index + 1, 'F1', f1_header_format)
        ws_col_indices[ws_size] = ws_col_index
        ws_col_index += 2

    video_name_format = workbook.add_format({'bold': True})
    map_value_format = workbook.add_format({'num_format': '0.00%', 'align': 'center', 'left': 1})
    f1_value_format = workbook.add_format({'num_format': '0.00%', 'align': 'center'})

    max_video_name_characters = 9

    video_folders.sort()
    for index, video_folder in enumerate(video_folders, start=3):
        folder = Path(video_folder)
        worksheet.write_string('A' + str(index), folder.name, video_name_format)

        if max_video_name_characters < len(folder.name):
            max_video_name_characters = len(folder.name)

        # write YOLO values
        eval_result_folder = Path(video_folder, 'eval/yolo_predictions/threshold_{}'.format(min_pred_score))
        if eval_result_folder.exists():
            eval_results_file = Path(eval_result_folder, 'results.txt')
            map_value = get_map_from_file(results_file=eval_results_file)
            f1_value = calculate_f1(results_file=eval_results_file)
            worksheet.write_number('B' + str(index), map_value, map_value_format)
            worksheet.write_number('C' + str(index), f1_value, f1_value_format)

        # write REPP values
        eval_result_folder = Path(video_folder, 'eval/repp_predictions/threshold_{}'.format(min_pred_score))
        if eval_result_folder.exists():
            eval_results_file = Path(eval_result_folder, 'results.txt')
            map_value = get_map_from_file(results_file=eval_results_file)
            f1_value = calculate_f1(results_file=eval_results_file)
            worksheet.write_number('D' + str(index), map_value, map_value_format)
            worksheet.write_number('E' + str(index), f1_value, f1_value_format)

        # write REPP RT values
        eval_result_folder = Path(video_folder, 'eval/repp_rt_predictions/threshold_{}'.format(min_pred_score))
        if eval_result_folder.exists():
            ws_eval_folders = [f for f in eval_result_folder.iterdir() if f.is_dir()]
            ws_eval_folders.sort(key=lambda n: int(n.name.split('_')[1]))
            for ws_result_folder in ws_eval_folders:
                if ws_result_folder.is_dir():
                    ws_size = int(ws_result_folder.name.split('_')[1].strip())
                    ws_col_index = ws_col_indices[ws_size]
                    eval_results_file = Path(ws_result_folder, 'results.txt')
                    map_value = get_map_from_file(results_file=eval_results_file)
                    f1_value = calculate_f1(results_file=eval_results_file)
                    worksheet.write_number(index - 1, ws_col_index, map_value, map_value_format)
                    worksheet.write_number(index - 1, ws_col_index + 1, f1_value, f1_value_format)

    # fit column width of video names
    worksheet.set_column(0, 0, max_video_name_characters)
    workbook.close()


def get_min_prediction_score() -> float:
    """Helper Function: Get defined minimum prediction score in REPP JSON.
    """
    repp_config = Path('yolo_repp_cfg.json').resolve()
    with open(repp_config, mode='r') as file:
        repp_config_json = json.load(file)
    return repp_config_json['min_pred_score']


def get_prediction_folders(video_folder: Path, min_pred_score: float) -> List[Path]:
    """Helper Function:
    Get all folders that should be evaluated (folder name: threshold_XX).
    For REPP RT the structure is threshold_XX/ws_XXX for different window sizes.
    """

    def threshold_folder_lookup(prediction_folder: Path, threshold: float) -> Path:
        for file in prediction_folder.iterdir():
            if file.is_dir():
                score = file.name[file.name.rindex('_') + 1:]
                if score == str(threshold):
                    return file

    prediction_folders = []

    repp_rt_prediction_folder = Path(video_folder, 'repp_rt_predictions')
    if repp_rt_prediction_folder.exists():
        threshold_folder = threshold_folder_lookup(repp_rt_prediction_folder, min_pred_score)
        if threshold_folder is not None:
            for ws_folder in threshold_folder.iterdir():
                if ws_folder.is_dir() and ws_folder.name[:ws_folder.name.rindex('_')] == 'ws':
                    prediction_folders.append(ws_folder)

    repp_prediction_folder = Path(video_folder, 'repp_predictions')
    if repp_prediction_folder.exists():
        threshold_folder = threshold_folder_lookup(repp_prediction_folder, min_pred_score)
        if threshold_folder is not None:
            prediction_folders.append(threshold_folder)

    yolo_prediction_folder = Path(video_folder, 'yolo_predictions')
    if yolo_prediction_folder.exists():
        threshold_folder = threshold_folder_lookup(yolo_prediction_folder, min_pred_score)
        if threshold_folder is not None:
            prediction_folders.append(threshold_folder)

    return prediction_folders


def ask_confirmation(message: str) -> bool:
    reply = ''
    while reply not in ['y', 'n']:
        reply = str(input(message)).lower().strip()
        if reply[:1] == 'y':
            return True
        if reply[:1] == 'n':
            return False


def ask_execution_choice() -> str:
    reply = ''
    while reply not in ['1', '2', '3']:
        user_input = str(input('Full execution (1); Metric calculation (2); Generate Excel file (3): ')).lower().strip()
        reply = user_input[:1]
    return reply


def execute_full():
    # iterate over all defined video folders and execute all steps
    for folder in video_folders:
        video_folder = Path(folder)
        print('\n===========================================================================\n')
        print('Start evaluation for', str(video_folder))
        delete_frames_outside_range(video_folder)
        print()
        add_empty_txt_to_predictions(video_folder)
        print()
        pickle_predictions(video_folder)
        print()
        execute_repp_rt(video_folder)
        print()
        execute_repp(video_folder)
        print()
        filter_raw_yolo_predictions(video_folder)
        print()
        filter_relevant_predictions(video_folder)
        print()
        execute_object_detection_metrics(video_folder)
        print('\n===========================================================================\n')


def execute_only_metrics():
    # iterate over all defined video folders and calculate metrics
    for folder in video_folders:
        video_folder = Path(folder)
        print('\n===========================================================================\n')
        print('Start metric calculations for', str(video_folder))
        execute_object_detection_metrics(video_folder)
        print('\n===========================================================================\n')


def main():
    print('Defined video folders:')
    for f in video_folders:
        print('\t', f)

    if ask_confirmation('Video folders will be modified. Does a backup copy exists? (y/n): '):
        choice = ask_execution_choice()

        if choice == '1':
            print('Start evaluation...')
            execute_full()
            print('Evaluation finished')
        elif choice == '2':
            print('Start metric calculation...')
            execute_only_metrics()
            print('Metric calculation finished')

        if choice == '3' or choice == '1' or choice == '2':
            print('Gather evaluation results and generate Excel file')
            gather_evaluation_results()
            print('Excel file generation finished')
    else:
        print('Evaluation canceled')


if __name__ == '__main__':
    main()
