"""
Performs evaluation with Object-Detection-Metrics.
"""

from pathlib import Path
import subprocess
import os

odm_project_path = Path('D:/Development/Projects/Object-Detection-Metrics-master')
# eval_subpath = 'rt_predictions(0.2)/ws-400-eval'
eval_subpath = 'repp_predictions(0.2)-eval'
videos = [
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


def execute_eval(video_folder: Path):
    det_path = Path(video_folder, eval_subpath)
    gt_path = Path(video_folder, 'gt')
    savepath = Path(video_folder, 'eval', eval_subpath)
    savepath.mkdir(parents=True)
    subprocess.call(
        ['D:/Development/Projects/Object-Detection-Metrics-master/venv/Scripts/python.exe',
         'D:/Development/Projects/Object-Detection-Metrics-master/pascalvoc.py',
         '-det', str(det_path),
         '-gt', str(gt_path),
         '-t', '0.5',
         '--savepath', str(savepath),
         '--noplot'])


if __name__ == '__main__':
    # change working directory to ODM project
    os.chdir(odm_project_path)
    for v in videos:
        execute_eval(video_folder=v)

