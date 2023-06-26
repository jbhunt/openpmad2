import pathlib as pl

headerBreakLine = '-' * 40 + '\n'

class StimulusBase():
    """
    """

    def __init__(self, display):
        self.display = display
        self.metadata = None
        self.header = None
        return

    def prepareMetadataStream(self, sessionFolder, filename, header):
        """
        """

        if self.metadata is None:
            return

        #
        sessionFolderPath = pl.Path(sessionFolder)
        if sessionFolderPath.exists() == False:
            sessionFolderPath.mkdir()

        #
        fullFilePath = sessionFolderPath.joinpath(f'{filename}.txt')

        #
        data = None
        if fullFilePath.exists():
            with open(fullFilePath, 'r') as stream:
                lines = stream.readlines()
            if len(lines) == 0:
                pass
            else:
                for lineIndex, line in enumerate(lines):
                    if line == headerBreakLine:
                        break
                data = lines[lineIndex + 1:]

        #
        stream = open(fullFilePath, 'w')
        for key, value in header.items():
            stream.write(f'{key}: {value}\n')
        stream.write(headerBreakLine)

        #
        if data is not None:
            for datum in data:
                stream.write(datum)

        return stream
