import numpy as np
import pathlib as pl
from PIL import Image
import matplotlib as mpl
from matplotlib import pylab as plt
from matplotlib.widgets import Button
from . import warping
from psychopy import visual, event

# Unbind the S key
try:
    mpl.rcParams['keymap.save'].remove('s')
    mpl.rcParams['keymap.back'].remove('left')
    mpl.rcParams['keymap.forward'].remove('right')
except ValueError as error:
    pass

def collectGridPoints(
    screen=1,
    size=[1280,720],
    shiftGain=10,
    markerInitPosition=(100, -140),
    gridShape=(11, 19)
    ):
    """
    Define the physical grid in pixel space

    Instructions
    ------------
    For each point of intersection within the grid, center the crosshair marker
    over the intersection, then press enter. If you'd like to reposition the
    marker, you can press Ctrl+Z to undo your selection. When you are done
    collecting the position of each grid point, press Shift+Enter to exit the
    function and return the grid (sorted by y position, then x position).
    """

    # create the crosshair marker
    win = visual.Window(screen=screen, size=size, fullscr=True)
    marker = visual.ShapeStim(
        win,
        vertices=[(-5,0), (5,0), (0,0), (0,-5), (0,5)],
        closeShape=False,
        lineWidth=1,
        lineColor='black',
        fillColor='black',
        units='pix'
    )
    marker.pos = markerInitPosition
    marker.draw()
    win.flip()

    # initialize empty lists
    dots = list()
    points = list()

    # main loop
    while True:

        # read keyboard buffer
        keys = event.getKeys(keyList=['left', 'right', 'up', 'down', 'return', 'z'], modifiers=True)

        # wait for input
        if len(keys) > 0:
            for (key, modifiers) in keys:

                # move the marker
                if key in ['left', 'right', 'up', 'down']:

                    # define the translation
                    dxy = 1 * shiftGain if modifiers['ctrl'] else 1
                    translation = [0, 0]
                    if key == 'left'  : translation[0] += dxy
                    if key == 'right' : translation[0] -= dxy
                    if key == 'up'    : translation[1] += dxy
                    if key == 'down'  : translation[1] -= dxy

                    # compute the new center coordinates
                    cold = marker.pos
                    cnew = (np.array(cold) + np.array(translation)).tolist()
                    marker.pos = cnew
                    print(f'Marker position: {marker.pos}')

                    # draw all elements
                    for dot in dots:
                        dot.draw()
                    marker.draw()
                    win.flip()

                # undo the last selection
                if key == 'z' and modifiers['ctrl']:
                    try:
                        point = points.pop(-1)
                        dot = dots.pop(-1)
                        for dot in dots:
                            dot.draw()
                        marker.draw()
                        win.flip()
                        print(f'Point removed: {point}')
                    except:
                        pass

                # save point
                if key == 'return' and not modifiers['ctrl']:
                    dot = visual.Circle(win, pos=marker.pos, fillColor='Red', lineColor='Red', size=3, units='pix')
                    dots.append(dot)
                    points.append(marker.pos)
                    for dot in dots:
                        dot.draw()
                    marker.draw()
                    win.flip()
                    print(f'Point saved: {marker.pos}')

                # exit and return the grid
                if key == 'return' and modifiers['shift']:
                    while True:
                        answer = input(f'Are you sure you want to exit? (y/n)\n')
                        if answer not in ('y', 'n'):
                            print(f'{answer} is not a valid answer')
                            continue
                        elif answer == 'n':
                            break
                        else:
                            nPoints = gridShape[0] * gridShape[1]
                            win.close()
                            points = np.array(points)
                            if points.shape[0] != gridShape[0] * gridShape[1]:
                                for i in range(int(nPoints - points.shape[0])):
                                    points = np.vstack([points, [np.nan, np.nan]])
                            return points    
                            grid = points.reshape(*gridShape, 2)
                            return grid

    return

def createWarpfile(destinationCoordinates, filepath, displayShape=(720, 1280)):
    """
    """

    displayHeight, displayWidth = displayShape
    aspectRatio = displayWidth / displayHeight
    nRows, nColumns = destinationCoordinates.shape[:2]
    nPoints = int(nRows * nColumns)

    # Create the grid of source coordinates
    sourceCoordinates = np.dstack(np.meshgrid(
        np.arange(nColumns),
        np.arange(nRows)
    )).reshape(nPoints, 2)
    x = sourceCoordinates[:, 0] / sourceCoordinates[:, 0].max()
    y = sourceCoordinates[:, 1] / sourceCoordinates[:, 1].max()
    sourceCoordinatesNormed = np.hstack([x.reshape(-1, 1), y.reshape(-1, 1)])

    #
    destinationCoordinates = destinationCoordinates.reshape(nPoints, 2)
    x = np.interp(destinationCoordinates[:, 0], (-1 * displayWidth / 2, displayWidth / 2), (-aspectRatio, aspectRatio)) / aspectRatio
    y = np.interp(destinationCoordinates[:, 1], (-1 * displayHeight / 2, displayHeight / 2), (-1, 1))
    destinationCoordinatesNormed = np.hstack([
        x.reshape(-1, 1),
        y.reshape(-1, 1)
    ])

    #
    header = (
        f'2\n',
        f'{nColumns:.0f} {nRows:.0f}\n'
    )
    data = np.ones([nPoints, 5])
    data[:, 0:2] = destinationCoordinatesNormed
    data[:, 2:4] = sourceCoordinatesNormed
    with open(filepath, 'w') as stream:
        for line in header:
            stream.write(line)
        for x1, y1, x2, y2, a in data:
            line = f'{x1} {y1} {x2} {y2} {a}\n'
            stream.write(line)
            
    return

def loadWarpfile(filepath):
    """
    """

    with open(filepath, 'r') as stream:
        lines = stream.readlines()

    data = list()
    for line in lines[2:]:
        x1, y1, x2, y2, a = line.split()
        data.append(list(map(float, [x1, y1, x2, y2])))

    return np.around(np.array(data), 3)

def load_sample_image(tag='horizontal-sinusoid-grating'):
    """
    """

    cwd = pl.Path(__file__)
    folder = cwd.parent.joinpath('data', 'images')
    result = list(folder.rglob(f'*{tag}*'))
    if len(result) == 0:
        raise Exception('No image file found')

    image = np.array(
        Image.open(str(result.pop())).convert('L')
    )

    return image

class ManualAdjustmentFigure():
    """
    """

    def __init__(self, grid, proximity=3, allow_online_warping=True, mc='k'):
        """
        """

        self.fig, self.ax = plt.subplots()
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.markers = self.ax.scatter(
            grid[:,0],
            grid[:,1],
            color=mc,
            s=20
        )

        #
        self.proximity = proximity
        self.imarker = None
        self.mc = mc
        self.colors = np.full(self.grid.shape[0], self.mc)
        self.dragging = False
        self.tuning = False

        #
        self.fig.canvas.mpl_connect('key_press_event', self.onKeyPress)
        self.fig.canvas.mpl_connect('button_press_event', self.onMouseClick)
        self.fig.canvas.mpl_connect('button_release_event', self.onMouseRelease)
        self.fig.canvas.mpl_connect('motion_notify_event', self.onMouseDrag)
        self.fig.canvas.mpl_connect('close_event', self.onFigureClose)

        #
        self.image = None
        self.button = None
        self.allow_online_warping = allow_online_warping
        if self.allow_online_warping:
            self._setup_online_warping()

        return

    def _setup_online_warping(self, margin=20):
        """
        """

        self.image = load_sample_image()
        warped = warping.warp(
            load_sample_image()
        )
        self.background = self.ax.imshow(warped, cmap='binary_r', zorder=-1, alpha=0.5)
        xlim = self.grid[:, 0].min() - margin, self.grid[:, 0].max() + margin
        ylim = self.grid[:, 1].min() - margin, self.grid[:, 1].max() + margin
        self.ax.set_ylim(ylim)
        self.ax.set_xlim(xlim)
        self.fig.subplots_adjust(bottom=0.2)
        self.button = plt.axes([0.05, 0.05, 0.1, 0.075])
        self.obj = Button(self.button, 'Warp', hovercolor='y')
        self.cid = self.obj.on_clicked(self.onWarpClick)

    def onWarpClick(self, event, margin=20):
        """
        """

        # Load the custom transformation
        warping.load_tform_data(dst=self.grid)

        # Update the background image
        warped = warping.warp(self.image)
        self.background.remove()
        self.background = self.ax.imshow(warped, cmap='binary_r', zorder=-1, alpha=0.5)
        xlim = self.grid[:, 0].min() - margin, self.grid[:, 0].max() + margin
        ylim = self.grid[:, 1].min() - margin, self.grid[:, 1].max() + margin
        self.ax.set_ylim(ylim)
        self.ax.set_xlim(xlim)

        return

    def onMouseClick(self, event):
        """
        """

        if self.locked:
            return

        if event.xdata is None and event.ydata is None:
            return

        # Find the closes node
        mouse = (event.xdata, event.ydata)
        dists = np.linalg.norm(self.grid - mouse, axis=1)
        close = dists <= self.proximity
        if close.sum() == 0:
            located = False
        else:
            located = True
            imarker = close.nonzero()[0].item()

        # Left click (Select grid node)
        if event.button == 1 and located:

            # Shift modifier (Hard select)
            if event.key == 'shift':
                self.imarker = imarker
                self.tuning = not self.tuning
                if self.tuning:
                    self.colors[self.imarker] = 'r'
                else:
                    self.colors[:] = self.mc
                self.markers.set_color(self.colors)
                self.fig.canvas.draw()

            # No shift modifier (Soft select)
            else:
                self.imarker = close.nonzero()[0].item()
                self.dragging = True

        # Right click (Add or remove grid node)
        elif event.button == 3:

            # Exit tuning mode
            if self.tuning:
                self.tuning = False
                self.colors = np.full(self.grid.shape[0], self.mc)
                self.markers.set_color(self.colors)

            # Shift modifier (Remove node)
            if event.key == 'shift' and located:
                nodes = np.delete(self.grid, imarker, axis=0)
                self.colors = np.full(nodes.shape[0], self.mc)
                self.markers.remove()
                self.markers = self.ax.scatter(
                    nodes[:,0],
                    nodes[:,1],
                    color=self.colors,
                    s=20
                )
                self.fig.canvas.draw()

            # No shift modifier (Add node)
            else:
                nodes = np.vstack([
                    self.grid,
                    np.array(mouse)
                ])
                self.colors = np.full(nodes.shape[0], self.mc)
                self.markers.remove()
                self.markers = self.ax.scatter(
                    nodes[:,0],
                    nodes[:,1],
                    color=self.colors,
                    s=20
                )
                self.fig.canvas.draw()

        return

    def onKeyPress(self, event):
        """
        """

        if self.locked:
            return

        if self.tuning:

            # Define the offset
            if event.key in ['up', 'down', 'left', 'right']:
                offset = 0.1
            elif event.key in ['a', 's', 'w', 'd']:
                offset = 0.5
            else:
                return

            # Compute the new node position
            point = self.grid[self.imarker]
            if event.key in ['up', 'w']:
                point[1] += offset
            elif event.key in ['down', 's']:
                point[1] -= offset
            elif event.key in ['left', 'a']:
                point[0] -= offset
            else:
                point[0] += offset

            # Update the grid
            nodes = np.copy(self.grid)
            nodes[self.imarker] = point
            self.markers.set_offsets(nodes)
            self.fig.canvas.draw()

        return

    def onMouseDrag(self, event):
        """
        """

        if self.dragging:

            # update positions
            offsets = self.markers.get_offsets()
            offsets[self.imarker, :] = np.array([event.xdata, event.ydata])
            self.markers.set_offsets(offsets)

            # highlight the selected marker
            self.colors = np.full(self.grid.shape[0], self.mc)
            self.colors[self.imarker] = 'r'
            self.markers.set_color(self.colors)

            # draw
            self.ax.figure.canvas.draw()

        return

    def onMouseRelease(self, event):
        """
        """

        if self.dragging:

            # update states
            self.imarker = None
            self.dragging = False

            # reset all marker colors to black
            self.colors = np.full(self.grid.shape[0], self.mc)
            self.markers.set_color(self.colors)
            self.fig.canvas.draw()

        return

    def onFigureClose(self, event):
        """
        """

        # Reload the default transformation
        warping.load_tform_data()

        return

    @property
    def grid(self):
        return np.array(self.markers.get_offsets())

    @property
    def locked(self):
        if self.fig.canvas.manager.toolbar.mode == '':
            return False
        else:
            return True