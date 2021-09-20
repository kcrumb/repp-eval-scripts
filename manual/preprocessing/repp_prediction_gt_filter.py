"""
Only copy those txt files that also have a GT file.
"""

from tqdm import tqdm
from pathlib import Path
import shutil

subpath = 'repp_predictions(0.2)'
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

if __name__ == '__main__':
    for predictions_folder in tqdm(video_folders):
        # get frame numbers das have a predictions
        predicted_frames = [int(f.name[:f.name.rindex('.')]) for f in Path(predictions_folder, subpath).iterdir()]
        # copy files into a separate eval folder
        eval_folder = Path(predictions_folder, '{}-eval'.format(subpath))
        eval_folder.mkdir()
        gt_folder = Path(predictions_folder, 'gt')
        # iterate over all GT numbers an copy the relevant predictions file
        for gt_file in tqdm(gt_folder.iterdir()):
            gt_frame_number = int(gt_file.name[:gt_file.name.rindex('.')])
            # if there is a prediction file with same frame number, then copy from predictions folder to eval folder
            if gt_frame_number in predicted_frames:
                shutil.copy(Path(predictions_folder, subpath, '{}.txt'.format(gt_frame_number)),
                            Path(eval_folder, '{}.txt'.format(gt_frame_number)))
            else:
                Path(eval_folder, '{}.txt'.format(gt_frame_number)).touch()
