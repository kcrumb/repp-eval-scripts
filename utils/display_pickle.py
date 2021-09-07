from pathlib import Path
import pickle

pickle_file = Path('D:/Development/REPP_videos/uniklinikum-endo_ci_3/endo_ci_3.pckl')
count = 3  # show the first n entries

if __name__ == '__main__':
    with open(pickle_file, mode='rb') as file:
        content = pickle.load(file)
    print('Name:', content[0])
    frame_boxes: dict = content[1]
    # iterate through the first n frames
    for idx, key in enumerate(sorted(frame_boxes.keys())):
        if idx > count - 1:
            break
        # iterate through boxes of frame
        for box in frame_boxes[key]:
            print(box)
