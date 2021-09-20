import shutil
import sys
from pathlib import Path

from tqdm import tqdm

giana_source = Path('D:/Development/GIANA_videos/GIANA_original')
first_frame_duplicates = 1000


def separate_frames_gt(source_sequence_folder: Path):
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
        print('>> Skip: GT folder aleady exists.')

    if frame_folder_exists and gt_folder_exists:
        return

    for file in tqdm(source_sequence_folder.iterdir()):
        if file.suffix == '.png' and not frame_folder_exists:
            shutil.copy(file, Path(frame_folder, file.name))
        elif file.suffix == '.txt' and not gt_folder_exists:
            shutil.copy(file, Path(gt_folder, file.name))
        elif (not file.suffix == '.png' and not frame_folder_exists) \
                and (not file.suffix == '.txt' and not gt_folder_exists):
            print('The following file cannot be mapped', str(file), file=sys.stderr)
    print('>> Done separating')


def duplicate_first_frame(sequence_folder: Path):
    """Duplicates the first frame n-times and renumbering the rest
    """
    print('> Duplicate first frame {} times'.format(first_frame_duplicates))
    frame_folder = Path(sequence_folder, 'frames')

    print('>> Renumber frames by adding ', first_frame_duplicates)
    # rename frames
    for frame_file in frame_folder.iterdir():
        frame_number = int(frame_file.name[:frame_file.name.rindex('.')])
        frame_number += first_frame_duplicates
        frame_file.rename(Path(frame_file.parent, str(frame_number) + frame_file.suffix))
    print('>> Done renumbering')

    print('>> Duplicate first frame')
    frames = [f for f in frame_folder.iterdir()]
    frames.sort(key=lambda f: int(f.name[:f.name.rindex('.')]))
    first_frame = frames[0]
    for i in tqdm(range(1, first_frame_duplicates + 1)):
        shutil.copy(first_frame, Path(first_frame.parent, str(i) + first_frame.suffix))
    print('>> Done duplicating')


if __name__ == '__main__':
    # separate_frames_gt(source_sequence_folder=Path(giana_source, '1'))
    duplicate_first_frame(sequence_folder=Path(giana_source.parent, 'GIANA_evaluation', '1'))
