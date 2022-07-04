class Stimulus():
    """
    """

    def __init__(self, display):
        """
        """

        self._display = display
        self._ppda = self.display.width / self.display.azimuth
        self._ppde = self.display.height / self.display.elevation
        self._movie = None

        return

    @property
    def display(self):
        return self._display

    @property
    def ppda(self):
        return self._ppda

    @property
    def ppde(self):
        return self._ppde

    @property
    def movie(self):
        return self._movie
