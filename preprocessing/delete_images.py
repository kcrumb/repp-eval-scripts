"""
Delete all images outside of a specific range
"""

from tqdm import tqdm
from pathlib import Path

frame_folder = Path('D:/Development/REPP_videos/uniklinikum-endo_cr_6/frames')
range_start = 10000
range_stop = 14999

if __name__ == '__main__':
    # get all files and delete all files that should stay
    frames = [f for f in frame_folder.iterdir() if int(f.name[:f.name.rindex('.')]) not in range(range_start, range_stop + 1)]
    # delete files that are outside of range
    for f in tqdm(frames):
        f.unlink()
