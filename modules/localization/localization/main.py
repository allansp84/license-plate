from .functions import *

def locatePlate(img, file_path, plot=False):

    scale, img_resized = resizeImage(img)
    regions = getRegions(img_resized)
    regions_filtered = filterRegions(img_resized, regions)
    if len(regions_filtered) == 1:
        points = refineBoundaries(img_resized, regions_filtered[0])
        points = [(int(x / scale), int(y / scale)) for (x,y) in points]
    else:
        points = [(0, 0), (0, 0), (0, 0), (0, 0)]

    if plot is True:
        from itertools import izip
        from .utils import createPlot
        from .utils import checkPlateSize
        from .utils import drawQuadrilateralRegion
        from .utils import drawQuadrilateralRegion
        from .utils import showPlot

        ax = createPlot(img_resized, file_path)
        ann = np.loadtxt(file_path.replace('.jpg', '.txt'), dtype=str, delimiter=',')
        if ann.ndim == 1:
            ground_truth = [(int(x), int(y)) for x, y in izip(* [iter(ann[0:8])] * 2)]
            # checkPlateSize(img_resized, scale, ground_truth)
            drawQuadrilateralRegion(ax, ground_truth, scale, 'green')

        # drawRectangleRegions(ax, regions, 'yellow')
        # drawRectangleRegions(ax, regions_filtered, 'orange')
        drawQuadrilateralRegion(ax, points, scale, 'red')
        showPlot()

    return points
