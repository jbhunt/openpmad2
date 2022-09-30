from doctest import master
import numpy as np
from psychopy.visual import ShapeStim
from shapely.geometry import LineString, Polygon
from openpmad2.bases import StimulusBase

def findBoundaryCrossingIndex(display, edge='leading', stepSize=1, motionAxisOrientation=0, motionAxisLength=1, barWidthInPixels=0):
    """
    """

    displayBoundaryLine = LineString([
        (     display.width / 2,      display.height / 2),
        (-1 * display.width / 2,      display.height / 2),
        (-1 * display.width / 2, -1 * display.height / 2),
        (     display.width / 2, -1 * display.height / 2),
        (     display.width / 2,      display.height / 2)
    ])

    motionAxisPoint1 = (
            0 - (motionAxisLength / 2) * np.cos(np.deg2rad(motionAxisOrientation)),
            0 - (motionAxisLength / 2) * np.sin(np.deg2rad(motionAxisOrientation))
        )
    motionAxisPoint2 = (0, 0)
    motionAxisLine = LineString([
        motionAxisPoint1,
        motionAxisPoint2
    ])

    if motionAxisLine.intersects(displayBoundaryLine):
        result = motionAxisLine.intersection(displayBoundaryLine)
        coords = np.array(result.coords.xy).flatten()
    else:
        raise Exception('No intersection detected')

    #
    innerLineSegment = LineString([
        coords,
        (0, 0)
    ])
    outerLineSegmentLength = (motionAxisLength / 2) - innerLineSegment.length - (barWidthInPixels / 2)
    targetFrameIndex = int(outerLineSegmentLength // stepSize + 1)

    return targetFrameIndex

class MovingBars(StimulusBase):
    """
    """

    def present(
        self,
        width=90,
        velocity=50,
        orientations=(0, 45, 90),
        itiDuration=3,
        randomize=False,
        repeats=1,
        marginSize=20,
        brightness=1,
        ): 
        """
        """

        self.header = {
            'Width': f'{width} (degrees)',
            'Velocity': f'{velocity} (degrees/second)',
            'Intensity': f'{brightness} (0, 1)'
        }

        #
        orientations = np.repeat(orientations, repeats)
        if randomize:
            np.random.shuffle(orientations)

        #
        bar = ShapeStim(self.display, lineWidth=0, units='pix')
        vertices = np.array([
            [-1 * self.display.ppd * width / 2, -1 * np.sqrt(self.display.width ** 2 + self.display.width ** 2) / 2],
            [-1 * self.display.ppd * width / 2,      np.sqrt(self.display.width ** 2 + self.display.width ** 2) / 2],
            [     self.display.ppd * width / 2,      np.sqrt(self.display.width ** 2 + self.display.width ** 2) / 2],
            [     self.display.ppd * width / 2, -1 * np.sqrt(self.display.width ** 2 + self.display.width ** 2) / 2]
        ])
        bar.vertices = vertices
        bar.setFillColor((1, 1, 1, brightness), colorSpace='rgba')

        #
        ppf = velocity * self.display.ppd / self.display.fps
        diameter = np.sqrt(self.display.width ** 2 + self.display.width ** 2) + (width / 2 * self.display.ppd) + (marginSize * self.display.ppd)
        remainder = diameter % ppf
        offset = remainder / 2
        stepCount = int(diameter // ppf)
        stepValues = np.linspace(0 - offset, diameter + offset, stepCount) - (diameter / 2)

        #
        self.metadata = np.full((10000, 3), np.nan)
        eventIndex = 0
        eventID = np.nan
        recordEvent = False

        #
        displayPolygon = Polygon([
            (     self.display.width / 2,      self.display.height / 2),
            (-1 * self.display.width / 2,      self.display.height / 2),
            (-1 * self.display.width / 2, -1 * self.display.height / 2),
            (     self.display.width / 2, -1 * self.display.height / 2),
            (     self.display.width / 2,      self.display.height / 2)
        ])

        #
        frameIndexCenterCrossed = int(np.ceil(stepValues.size / 2))

        #
        self.display.idle(3, units='seconds')

        #
        for orientation in orientations:

            # Change the bar orientation and define the axis of motion
            bar.ori = orientation
            theta = np.deg2rad(180 - orientation)
            barVisible = False

            #
            for frameIndex, stepValue in enumerate(stepValues):

                # Update the bar position
                bar.pos = (
                    stepValue * np.cos(theta),
                    stepValue * np.sin(theta)
                )
                barPolygon = Polygon(bar.verticesPix)

                # Look for the appearance of the leading edge
                if barVisible == False:
                    if displayPolygon.intersects(barPolygon) == True:
                        self.display.signalEvent(0.05, units='seconds')
                        barVisible = True
                        recordEvent = True
                        eventID = 1

                # Look for the disappearance of the trailing edge
                else:
                    if displayPolygon.intersects(barPolygon) == False:
                        self.display.signalEvent(0.05, units='seconds')
                        barVisible = False
                        recordEvent = True
                        eventID = 3

                #
                # if frameIndex == frameIndexCenterCrossed:
                #     self.display.signalEvent(0.05, units='seconds')
                #     recordEvent = True
                #     eventID = 2

                #
                self.display.drawBackground()
                bar.draw()
                timestamp = self.display.flip()
                
                #
                if recordEvent:
                    self.metadata[eventIndex, :] = (eventID, orientation, timestamp)
                    recordEvent = False
                    eventIndex += 1

            # ITI
            self.display.idle(itiDuration)

        return

    def saveMetadata(self, sessionFolder):
        """
        """

        self.header.update({
            'Columns': 'Event (1=Onset, 2=Centered, 3=Offset), Orientation (degrees), Timestamp (seconds)'
        })
        stream = super().prepareMetadataStream(sessionFolder, 'movingBarsMetadata', self.header)
        for line in self.metadata:
            if np.isnan(line).all():
                continue
            eventID, orientation, timestamp = line
            stream.write(f'{eventID:.0f}, {orientation:.0f}, {timestamp:.3f}\n')

        stream.close()

        return