import cv2 as cv
import numpy as np
import skvideo.io as io

class VideoWriterOpenCV():
    """
    """

    def __init__(self, filename=None, shape=(720, 1280), fps=60, vflip=False):
        """
        """

        if filename is None:
            self._writer = None
            return

        height, width = shape
        codec = cv.VideoWriter_fourcc(*'mp4v')
        self._writer = cv.VideoWriter(
            filename,
            codec,
            fps,
            (width, height),
            False
        )
        self._vflip = vflip

        return

    def write(self, frame):
        """
        """

        stack = np.dstack([frame, frame, frame])

        if self._vflip:
            stack = cv.flip(stack, 0)

        if self._writer is not None:
            self._writer.write(stack)

        return

    def close(self):
        """
        """

        if self._writer is not None:
            self._writer.release()

        return

class VideoWriterSkvideo():
    """
    """

    def __init__(self, filename=None, shape=(720, 1280), fps=60, crf=17, vflip=False):
        """
        """

        if filename is None:
            self._writer = None
            return

        height, width = shape
        idct = {
            '-r': f'{fps}',
            '-s': f'{width}x{height}'
        }
        odct = {
            '-r'      : f'{fps}',
            '-c:v'    : f'libx264',
            '-crf'    : f'{crf}',
            '-preset' : f'ultrafast',
            '-pix_fmt': f'yuv444p'
        }

        #
        if vflip:
            odct['-vf'] = f'vflip'

        self._writer = io.FFmpegWriter(
            filename,
            inputdict=idct,
            outputdict=odct,
        )

        return

    def write(self, array):
        """
        """

        if self._writer is not None:
            self._writer.writeFrame(array)

        return

    def close(self):
        """
        """

        if self._writer is not None:
            self._writer.close()

        return