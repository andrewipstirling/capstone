import numpy as np

def get_dodecahedron_vertices(edge_length):
    phi =(1 + np.sqrt(5))/2

    vertices = np.array(
        [[phi,1/phi,0], # pink
        [-phi,1/phi,0],
        [phi,-1/phi,0],
        [-phi,-1/phi,0],
        [1/phi,0,phi], # blue
        [-1/phi,0,phi],
        [1/phi,0,-phi],
        [-1/phi,0,-phi],
        [0,phi,1/phi], # green
        [0,phi,-1/phi],
        [0,-phi,1/phi],
        [0,-phi,-1/phi],
        [1,1,1], # orange
        [1,1,-1],
        [1,-1,1],
        [1,-1,-1],
        [-1,1,1],
        [-1,1,-1],
        [-1,-1,1],
        [-1,-1,-1],
        ]
    )
    vertices *= (edge_length) / (2/phi)

    return vertices
