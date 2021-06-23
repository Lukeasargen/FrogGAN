import pathlib

import cv2


def main():

    names = [
        "16/videos.avi",
        "17/videos.avi",
        "18/videos.avi",
        "19/videos.avi",
        "21/videos.avi",
        "22/videos.avi",
        "24/videos.avi",
        "25/videos.avi",
        "26/videos.avi",
    ]

    rootdir = "output"
    outname = "square_v1.avi"
    
    rootdir = pathlib.Path(rootdir)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 30.0

    started = False

    c = 0

    for f in names:
        print(rootdir / f)
        cap = cv2.VideoCapture(str(rootdir / f))

        frame = None
        while(cap.isOpened()):
            prev = frame
            ret, frame = cap.read()

            if ret:
                if not started:
                    height, width, c = frame.shape
                    videopath = rootdir / outname
                    print(videopath)
                    video = cv2.VideoWriter(str(videopath), fourcc, fps, (width, height))
                    started = True

                if started:
                    video.write(frame)
                    c +=1

                if c%100==0:
                    print(c)
            else:
                for i in range(int(4.0*fps)):
                    video.write(prev)

                break

        cap.release()

    video.release()


if __name__ == '__main__':
    main()
