import numpy as np
import skvideo.io as io

class SKVideoVideoWriterWrapper():
    """
    """

    def __init__(self, display, filename=None, crf=17, vflip=False):
        """
        """

        if filename is None:
            self._writer = None
            return

        idct = {
            '-r': f'{display.fps}',
            '-s': f'{display.width}x{display.height}'
        }
        odct = {
            '-r'      : f'{display.fps}',
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
            outputdict=odct
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
    
class VideoBuffer():
    """
    TODO: Finish coding this
    """

    def __init__(self):
        """
        """

        self._array = None
        self._index = None

        return
    
    def open(self, nFrames, shape):
        """
        """

        self._array = np.full([nFrames, *shape], 0).astype(np.uint8)
        self._index = 0

        return
    
    def write(self):
        """
        """

        return
    
    def close(self, filename):
        """
        """

        return