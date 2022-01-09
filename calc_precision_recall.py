import ast
from pathlib import Path

import numpy as np

video_folders = [
    Path('D:/Development/REPP_videos/uniklinikum-endo_ci_3'),
    Path('D:/Development/REPP_videos/uniklinikum-endo_cj_4'),
    Path('D:/Development/REPP_videos/uniklinikum-endo_co_3'),
    Path('D:/Development/REPP_videos/uniklinikum-endo_co_5'),
    Path('D:/Development/REPP_videos/uniklinikum-endo_co_6'),
    Path('D:/Development/REPP_videos/uniklinikum-endo_cr_1'),
    Path('D:/Development/REPP_videos/uniklinikum-endo_cr_3'),
    Path('D:/Development/REPP_videos/uniklinikum-endo_cr_4'),
    Path('D:/Development/REPP_videos/uniklinikum-endo_cr_5'),
    Path('D:/Development/REPP_videos/uniklinikum-endo_cr_6'),
    Path('D:/Development/GIANA_videos/GIANA_evaluation/1'),
    Path('D:/Development/GIANA_videos/GIANA_evaluation/2'),
    Path('D:/Development/GIANA_videos/GIANA_evaluation/3'),
    Path('D:/Development/GIANA_videos/GIANA_evaluation/4'),
    Path('D:/Development/GIANA_videos/GIANA_evaluation/5'),
    Path('D:/Development/GIANA_videos/GIANA_evaluation/6'),
    Path('D:/Development/GIANA_videos/GIANA_evaluation/7'),
    Path('D:/Development/GIANA_videos/GIANA_evaluation/8'),
    Path('D:/Development/GIANA_videos/GIANA_evaluation/9'),
    Path('D:/Development/GIANA_videos/GIANA_evaluation/10'),
    Path('D:/Development/GIANA_videos/GIANA_evaluation/11'),
    Path('D:/Development/GIANA_videos/GIANA_evaluation/12'),
    Path('D:/Development/GIANA_videos/GIANA_evaluation/13'),
    Path('D:/Development/GIANA_videos/GIANA_evaluation/14'),
    Path('D:/Development/GIANA_videos/GIANA_evaluation/15'),
    Path('D:/Development/GIANA_videos/GIANA_evaluation/16'),
    Path('D:/Development/GIANA_videos/GIANA_evaluation/17'),
    Path('D:/Development/GIANA_videos/GIANA_evaluation/18'),
]


def print_csv_line(video_name: str, algo: str, threshold: str, ws: str, precision: str, recall: str, f1: str):
    def round_float(x: str):
        return '{:.4f}'.format(float(x))
    print(video_name + ';' + algo.ljust(7) + ';' + threshold + ';' + ws.ljust(4) + ';' + round_float(precision) + ';' +
          round_float(recall) + ';' + round_float(f1))


def yolo_repp_eval(video: Path, eval_folder: Path):
    algo_string = eval_folder.name.split('_')[0].strip()
    yolo_eval_threshold_folder = Path(video, eval_folder)
    thresholds = [t for t in yolo_eval_threshold_folder.iterdir() if t.is_dir()]
    for threshold_folder in thresholds:
        threshold_value = threshold_folder.name[threshold_folder.name.rindex('_') + 1:]
        eval_results = Path(threshold_folder, 'results.txt')
        # open file
        with open(eval_results, mode='r') as f:
            lines = f.readlines()
        # read precision and recall
        precision, recall = -1, -1
        for line in lines:
            if line.startswith('Precision:'):
                line = line.replace('Precision: ', '')
                precision = np.asarray(ast.literal_eval(line), dtype=np.double)
            if line.startswith('Recall:'):
                line = line.replace('Recall: ', '')
                recall = np.asarray(ast.literal_eval(line), dtype=np.double)

        precision = precision + 0.00000000001
        recall = recall + 0.00000000001
        f1_list = 2 * (precision * recall) / (precision + recall)
        f1_index = np.argmax(f1_list)
        f1_max = np.max(f1_list)
        precision_max = precision[f1_index]
        recall_max = recall[f1_index]
        print_csv_line(video.name, algo_string, threshold_value, '----', str(precision_max), str(recall_max), str(f1_max))


def rt_repp_eval(video: Path):
    yolo_eval_threshold_folder = Path(video, 'eval/repp_rt_predictions')
    thresholds = [t for t in yolo_eval_threshold_folder.iterdir() if t.is_dir()]
    for threshold_folder in thresholds:
        threshold_value = threshold_folder.name[threshold_folder.name.rindex('_') + 1:]

        ws_folders = [f for f in threshold_folder.iterdir() if f.is_dir()]
        ws_folders.sort(key=lambda d: int(d.name.split('_')[1]))
        for ws_folder in ws_folders:
            ws_value = ws_folder.name.split('_')[1]

            eval_results = Path(ws_folder, 'results.txt')
            # open file
            with open(eval_results, mode='r') as f:
                lines = f.readlines()
            # read precision and recall
            precision, recall = -1, -1
            for line in lines:
                if line.startswith('Precision:'):
                    line = line.replace('Precision: ', '')
                    precision = np.asarray(ast.literal_eval(line), dtype=np.double)
                if line.startswith('Recall:'):
                    line = line.replace('Recall: ', '')
                    recall = np.asarray(ast.literal_eval(line), dtype=np.double)

            precision = precision + 0.00000000001
            recall = recall + 0.00000000001
            f1_list = 2 * (precision * recall) / (precision + recall)
            f1_index = np.argmax(f1_list)
            f1_max = np.max(f1_list)
            precision_max = precision[f1_index]
            recall_max = recall[f1_index]
            print_csv_line(video.name, 'repp_rt', threshold_value, ws_value, str(precision_max), str(recall_max),
                           str(f1_max))


def main():
    print('name;algo;threshold;ws;precision;recall;f1')
    for video in video_folders:
        yolo_repp_eval(video, Path('eval/yolo_predictions'))
        yolo_repp_eval(video, Path('eval/repp_predictions'))
        rt_repp_eval(video)


if __name__ == '__main__':
    main()
    # print('name;algo;threshold;precision;recall;f1')
    # for video in video_folders:
        # yolo_eval_threshold_folder = Path(video, yolo_eval_sub_path)

        # thresholds = [t for t in yolo_eval_threshold_folder.iterdir() if t.is_dir()]
        # for threshold_folder in thresholds:
        #     threshold_value = threshold_folder.name[threshold_folder.name.rindex('_') + 1:]
        #     eval_results = Path(threshold_folder, 'results.txt')
        #     # open file
        #     with open(eval_results, mode='r') as f:
        #         lines = f.readlines()
        #     # read precision and recall
        #     precision, recall = -1, -1
        #     for line in lines:
        #         if line.startswith('Precision:'):
        #             line = line.replace('Precision: ', '')
        #             precision = np.asarray(ast.literal_eval(line), dtype=np.double)
        #         if line.startswith('Recall:'):
        #             line = line.replace('Recall: ', '')
        #             recall = np.asarray(ast.literal_eval(line), dtype=np.double)
        #
        #     precision = precision + 0.00000000001
        #     recall = recall + 0.00000000001
        #     f1_list = 2 * (precision * recall) / (precision + recall)
        #     f1_index = np.argmax(f1_list)
        #     f1_max = np.max(f1_list)
        #     precision_max = precision[f1_index]
        #     recall_max = recall[f1_index]
        #     print(video.name + ';' + threshold_value + ';' + str(precision_max) + ';' + str(recall_max) + ';' + str(f1_max))