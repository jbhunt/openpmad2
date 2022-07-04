import skvideo.io as io

class SKVideoVideoWriterWrapper():
    """
    """

    def __init__(self, display, filename, crf=17):
        """
        """

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

        self._writer = io.FFmpegWriter(
            filename,
            inputdict=idct,
            outputdict=odct
        )

        return

    def write(self, array):
        """
        """

        self._writer.writeFrame(array)

        return

    def close(self):
        """
        """

        self._writer.close()

        return
