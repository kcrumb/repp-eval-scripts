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
    Path('D:/Development/REPP_videos/uniklinikum-endo_cr_6')
]
# eval_subpath = 'repp_predictions(0.2)-eval'
# eval_subpath = 'rt_predictions(0.2)/ws-400-eval'
eval_subpath = 'yolo_predictions(0.2)-eval'

if __name__ == '__main__':
    for video in video_folders:
        eval_folder = Path(video, 'eval', eval_subpath)
        eval_results = Path(eval_folder, 'results.txt')
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
        f1 = max(2 * (precision * recall) / (precision + recall))
        print('{} {:.4f}'.format(video.name, f1))
        # print(np.where(f1 == np.amax(f1)))
