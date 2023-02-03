# GAIN and RON from Nick's Github (unsure if these are correct?)
ARIES_GAIN = 4.0
ARIES_RON = 10.
ARIES_NX, ARIES_NY = (1024, 1024)
ARIES_BADCOLUMNS = (0, 512)
ARIES_NORDERS = 26
ARIES_SPECTRAL_RESOLUTION = 3e4

def scale_to_rgb(colors):
    for i in range(len(colors)):
        r, g, b = colors[i]
        colors[i] = (r / 255., g / 255., b / 255.)
    return colors

TABLEAU20 = scale_to_rgb(
    [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
    (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
    (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
    (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
    (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
)

def get_tableau20_colors(n):
    """Returns a list of n distinct tableau20 colors."""
    return [TABLEAU20[i%20] for i in range(n+1)]