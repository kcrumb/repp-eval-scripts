import os
import subprocess
from pathlib import Path
from tqdm import tqdm

# config
repp_project_path = Path('D:/Development/Projects/repp')
repp_config = Path(repp_project_path, 'REPP_cfg/yolo_repp_cfg.json')
window_sizes = [400]
# tuples of videos (0: video name, 1: pickled predictions, 2: RT predictions folder, 3: video frames)
root = Path('D:/Development/REPP_videos')
videos = [
    ('endo_ci_3', Path(root, 'uniklinikum-endo_ci_3/endo_ci_3.pckl'), Path(root, 'uniklinikum-endo_ci_3/rt_predictions(0.2)'), Path(root, 'uniklinikum-endo_ci_3/frames')),
    ('endo_cj_4', Path(root, 'uniklinikum-endo_cj_4/endo_cj_4.pckl'), Path(root, 'uniklinikum-endo_cj_4/rt_predictions(0.2)'), Path(root, 'uniklinikum-endo_cj_4/frames')),
    ('endo_co_3', Path(root, 'uniklinikum-endo_co_3/endo_co_3.pckl'), Path(root, 'uniklinikum-endo_co_3/rt_predictions(0.2)'), Path(root, 'uniklinikum-endo_co_3/frames')),
    ('endo_co_5', Path(root, 'uniklinikum-endo_co_5/endo_co_5.pckl'), Path(root, 'uniklinikum-endo_co_5/rt_predictions(0.2)'), Path(root, 'uniklinikum-endo_co_5/frames')),
    ('endo_co_6', Path(root, 'uniklinikum-endo_co_6/endo_co_6.pckl'), Path(root, 'uniklinikum-endo_co_6/rt_predictions(0.2)'), Path(root, 'uniklinikum-endo_co_6/frames')),
    ('endo_cr_1', Path(root, 'uniklinikum-endo_cr_1/endo_cr_1.pckl'), Path(root, 'uniklinikum-endo_cr_1/rt_predictions(0.2)'), Path(root, 'uniklinikum-endo_cr_1/frames')),
    ('endo_cr_3', Path(root, 'uniklinikum-endo_cr_3/endo_cr_3.pckl'), Path(root, 'uniklinikum-endo_cr_3/rt_predictions(0.2)'), Path(root, 'uniklinikum-endo_cr_3/frames')),
    ('endo_cr_4', Path(root, 'uniklinikum-endo_cr_4/endo_cr_4.pckl'), Path(root, 'uniklinikum-endo_cr_4/rt_predictions(0.2)'), Path(root, 'uniklinikum-endo_cr_4/frames')),
    ('endo_cr_5', Path(root, 'uniklinikum-endo_cr_5/endo_cr_5.pckl'), Path(root, 'uniklinikum-endo_cr_5/rt_predictions(0.2)'), Path(root, 'uniklinikum-endo_cr_5/frames')),
    ('endo_cr_6', Path(root, 'uniklinikum-endo_cr_6/endo_cr_6.pckl'), Path(root, 'uniklinikum-endo_cr_6/rt_predictions(0.2)'), Path(root, 'uniklinikum-endo_cr_6/frames'))
]


def execute_repp_rt(window_size: int, prediction_file: Path, prediction_folder: Path, video_frames_folder: Path):
    """Execute real-time REPP with different window sizes"""
    subprocess.call(
        ['D:/Development/Projects/repp/venv/Scripts/python.exe',
         'D:/Development/Projects/repp/REPP_RT.py',
         '--repp_cfg', str(repp_config),
         '--predictions_file', str(prediction_file),
         '--store_coco',
         '--store', str(prediction_folder),
         '--window_size', str(window_size),
         '--frames', str(video_frames_folder)])


if __name__ == '__main__':
    # change working directory to REPP project
    os.chdir(repp_project_path)
    # execute
    print('> Execute REPP RT')
    # iterate over videos
    for video in tqdm(videos):
        # create predications folder for video
        if not video[2].exists():
            video[2].mkdir(parents=True)
        # iterate over windows sizes
        for ws in tqdm(window_sizes):
            # create predication folder for specific window size
            ws_folder = Path(video[2], 'ws-{}'.format(ws))
            execute_repp_rt(window_size=ws, prediction_file=video[1], prediction_folder=ws_folder, video_frames_folder=video[3])
