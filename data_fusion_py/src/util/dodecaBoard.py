
# Pentagon formulas: https://www.calctool.org/math-and-statistics/pentagon
# Dodecahedron formulas: https://www.treenshop.com/Treenshop/ArticlesPages/FiguresOfInterest_Article/The%20Dodecahedron.htm#
import numpy as np

# m is marker length in mm
# l is edge length of the pentagonal faces
# offset is x, y, z tuple defining distance to tool tip
# alignment of marker on pentagon face can be 'centre' (default) or 'edge'
# edge alignment currently only works if m = l
def generate(m, l, offset=(0,0,0), alignment='centre'):

    pentaAngle = 3*np.pi/5  # Angle between each edge within a pentagonal face (rad), = 108 deg
    adjAngle = np.arccos(-np.sqrt(5)/5)  # Angle between adjacent faces across their connecting edge, AKA dihedral angle (rad), = ~116.565 deg
    cosAdj = np.cos(np.pi-adjAngle)
    sinAdj = np.sin(adjAngle)

    theta = 2*np.pi/5 # Relative angle of adjacent faces about the central axis (i.e. of pentagon edges about centre) (rad)
    s = np.sin(theta)
    c = np.cos(theta)
    rotMat = np.array([[c, -s, 0],
                    [s, c, 0],
                    [0, 0, 1]], dtype=np.float32)

    h = l*np.sqrt(5+2*np.sqrt(5))/2  # Pentagon height
    r = l/10*np.sqrt(25+10*np.sqrt(5))  # Pentagon incircle radius (centre to edge distance)
    R = l/10*np.sqrt(50+10*np.sqrt(5))  # Pentagon circumcircle radius (centre to vertex distance)
    H = l/10*np.sqrt(250+110*np.sqrt(5))  # Height of dodecahedron, ie diameter of inscribed sphere
    
    margin = r-m/2  # distance from parallel pentagon edge to marker edge
    
    # Tags are defined by 3D coordinates of top left, top right, bottom right, bottom left corners in that order
    # Lower ID of each tag belongs to reference, higher belongs to target
    tags = np.zeros((11, 4, 3), dtype=np.float32)
    
    # Top tag, ID 0/11
    if alignment == 'edge':
        tags[0] = [[l/2, r-l, H], [-l/2, r-l, H], [-l/2, r, H], [l/2, r, H]]
    else:
        tags[0] = [[m/2, -m/2, H], [-m/2, -m/2, H], [-m/2, m/2, H], [m/2, m/2, H]]

    # Top-adjacent tags, IDs 1-5/11-16
    if alignment == 'edge':
        tags[1] = [[l/2, r, H], [-l/2, r, H], 
                [-l/2, r+l*cosAdj, H-l*sinAdj], 
                [l/2, r+l*cosAdj, H-l*sinAdj]]
    else:
        tags[1] = [[m/2, r+margin*cosAdj, H-margin*sinAdj], [-m/2, r+margin*cosAdj, H-margin*sinAdj], 
                [-m/2, r+(margin+m)*cosAdj, H-(margin+m)*sinAdj], 
                [m/2, r+(margin+m)*cosAdj, H-(margin+m)*sinAdj]]

    tags[2:6] = [(np.linalg.matrix_power(rotMat, i) @ tags[1].T).T for i in range(1,5)]

    # Bottom-adjacent tags, IDs 6-10/17-21
    if alignment == 'edge':
        tags[6] = [[-l/2, -r-l*cosAdj, l*sinAdj], 
                [l/2, -r-l*cosAdj, l*sinAdj], 
                [l/2, -r, 0], [-l/2, -r, 0]]
    else:
        tags[6] = [[-m/2, -r-(margin+m)*cosAdj, (margin+m)*sinAdj], 
                [m/2, -r-(margin+m)*cosAdj, (margin+m)*sinAdj], 
                [m/2, -r-margin*cosAdj, margin*sinAdj], [-m/2, -r-margin*cosAdj, margin*sinAdj]]

    tags[7:11] = [(np.linalg.matrix_power(rotMat, i) @ tags[6].T).T for i in range(1,5)]
    
    tags += offset

    return tags

# %%
