import time
import argparse
import pathlib

import cv2
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


def main():

    log_path = r"lightning_logs\version_49\events.out.tfevents.1619672114.DESKTOP-CEDVQMN.8692.0"
    
    out_dir = "output/49"

    t0 = time.time()

    event_acc = event_accumulator.EventAccumulator(log_path, size_guidance={'images': 0})
    print("event_acc :", event_acc)
    event_acc.Reload()

    print("Reload :", time.time()-t0)

    outdir = pathlib.Path(out_dir)
    outdir.mkdir(exist_ok=True, parents=True)

    for tag in event_acc.Tags()['images']:
        events = event_acc.Images(tag)

        tag_name = tag.replace('/', '_')
        dirpath = outdir / tag_name
        dirpath.mkdir(exist_ok=True, parents=True)
        print("dirpath :", dirpath)

        videopath = outdir / 'video.avi'
        print("videopath :", videopath)

        video = None  # Will be the video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = 30.0

        for index, event in enumerate(events):
            s = np.frombuffer(event.encoded_image_string, dtype=np.uint8)
            image = cv2.imdecode(s, cv2.IMREAD_COLOR)
            if index == 0:
                height, width, c = image.shape
                video = cv2.VideoWriter(str(videopath), fourcc, fps, (width, height))

            video.write(image)

            outpath = dirpath / '{:05}.jpg'.format(index)
            cv2.imwrite(outpath.as_posix(), image)
    
    video.release()


if __name__ == '__main__':
    main()
