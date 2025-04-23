from matplotlib.colors import ListedColormap

def cmap_confidence():
    # Create a colormap with 4 distinct colors
    colors = [(0.5, 0.5, 0.5),    # grey
              (1.0, 1.0, 0.0),    # yellow
              (1.0, 0.5, 0.0),    # orange
              (1.0, 0.0, 0.0)]    # red
    
    return ListedColormap(colors, name='Confidence')

def cmap_classification():
    # 0 = unevaluated (light grey)
    # 1 = noise (no-signal) (black)
    # 2 = water surface (blue)
    # 3 = subaqueous unclassified (grey)
    # 4 = subaqueous noise (green)
    # 5 = bathymetry (red)

    colors = [(0.2, 0.2, 0.2),    # light grey
              (0.0, 0.0, 0.0),    # black
              (0.0, 0.0, 1.0),    # blue
              (0.5, 0.5, 0.5),    # grey
              (0.0, 1.0, 0.0),    # green
              (1.0, 0.0, 0.0)]    # red

    label_map = {
        0: 'unevaluated',
        1: 'noise (no-signal)',
        2: 'water surface',
        3: 'subaqueous unclassified',
        4: 'subaqueous noise',
        5: 'bathymetry'
    }
              
    return ListedColormap(colors, name='Classification'), label_map