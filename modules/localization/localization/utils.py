# ------------------------------------------------------------

from shapely.geometry import Polygon

from matplotlib import pyplot as plt
from matplotlib import patches as patches


# ------------------------------------------------------------

# Calculates localization Accuracy by means of intersection / union
# using the ground truth and predicted points of the quadrilateral.

def localizationAccuracy(truth, pred):
    truth = Polygon(truth)
    pred = Polygon(pred)

    inter = truth.intersection(pred)
    union = truth.union(pred)
    lr = inter.area / union.area

    print ''
    print 'tr points -> {0}'.format(truth)
    print 'pr points -> {0}'.format(pred)
    print ''
    print 'inter {0}'.format(inter.area)
    print 'union {0}'.format(union.area)
    print 'IR {0}'.format(lr)
    print ''

    return lr


# ------------------------------------------------------------

# Verifies if the license plate size is good comparing its area to
# the image area.

def checkPlateSize(img_resized, scale, localization_truth):
    img_height = img_resized.shape[0]
    img_width = img_resized.shape[1]
    img_area = img_height * img_width

    points = [(x * scale, y * scale) for (x, y) in localization_truth]
    plate = Polygon(points)
    plate_area = plate.area
    ratio = plate_area / img_area

    print ''
    print 'img area {0}'.format(img_area)
    print 'plate area {0}'.format(plate_area)
    print 'ratio {0}'.format(ratio)
    print ''

    if ratio < 0.02 or ratio > 0.125:
        print '   --> ratio not ok'
        print ''
        return True

    return False


# ------------------------------------------------------------

# Create plot object 

def createPlot(img, file_name):
    fig, ax = plt.subplots()
    fig.canvas.set_window_title(file_name)
    ax.imshow(img, cmap=plt.cm.gray)
    return ax


# ------------------------------------------------------------

# Show plot object

def showPlot():
    plt.show()


# ------------------------------------------------------------

# Draw a quadilateral region to plot. If scale is not 1, points will
# be rescaled to fit the image

def drawQuadrilateralRegion(ax, localization_truth, scale, color):
    if scale != 1:
        points = [(x * scale, y * scale) for (x, y) in localization_truth]
    else:
        points = localization_truth
    rect = patches.Polygon(points, fill=False, edgecolor=color, linewidth=2)
    ax.add_patch(rect)


# ------------------------------------------------------------

# Draw a rectangle region to plot.

def drawRectangleRegions(ax, regions, color):
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        w = maxc - minc
        h = maxr - minr
        rect = patches.Rectangle((minc, minr), w, h, fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(rect)

        # ------------------------------------------------------------
