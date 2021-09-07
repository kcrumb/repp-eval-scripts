"""
Add empty txt-files to the folder with YOLO predictions.
Because if a frame has no prediction then no txt-file exists.
"""

from pathlib import Path
from tqdm import tqdm

# config
folder = Path('D:/Development/REPP_videos/uniklinikum-endo_cr_6/yolo_predictions')

if __name__ == '__main__':
    # get all frame numbers that have a prediction and sort them
    file_numbers = [int(f.name[:f.name.rindex('.')]) for f in folder.iterdir() if f.name.endswith('.txt')]
    file_numbers.sort()
    # generate a set with all number within the range of frame numbers
    all_numbers = set([i for i in range(file_numbers[0], file_numbers[-1] + 1)])
    file_numbers = set(file_numbers)
    # get all numbers that are missing from prediction set
    missing_numbers = sorted(list(all_numbers - file_numbers))
    # create empty txt-files for missing numbers
    print('> Create empty files for missing numbers')
    for m in tqdm(missing_numbers):
        Path(folder, '{}.txt'.format(m)).touch()
