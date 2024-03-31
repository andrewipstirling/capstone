# import click
import cv2
from cv2 import aruco
import os
import numpy as np
import yaml
from PIL import Image

class MarkerFactory:

    @staticmethod
    def create_marker(tile_size, marker_size, id, margin, diam_size, penta_rotation):
        # aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_100)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_100)

        # white background
        img = 255 * np.ones((tile_size, tile_size), dtype=np.uint8)
        # img_marker = aruco.drawMarker(aruco_dict, id, tile_size - 2 * margin,borderBits=1)
        img_marker = aruco.generateImageMarker(aruco_dict, id, marker_size)

        # Calculate border and vertical position based on ID
        #y_shift = 21 # For 40mm marker/pentagon
        y_shift = 15 # For 27.5mm marker/pentagon
        border = int(margin + ((diam_size - marker_size) / 2))
        if id == 0 or (6 <= id <= 11):  # For markers with bottom alignment
            # img[y_shift+border:y_shift-border, border:-border] = img_marker # For 40mm
            img[y_shift+border:y_shift-border-1, border:-border-1] = img_marker # For 27.5mm add -1 to border to deal with even marker size (pixel)
        else:  # For markers with top alignment
            # img[border-y_shift:-border-y_shift, border:-border] # For 40mm
            img[border-y_shift:-border-y_shift-1, border:-border-1] = img_marker # For 27.5mm add -1 to border to deal with even marker size (pixel)

        # Calculate pentagon vertices
        # https://math.stackexchange.com/questions/1990504/how-to-find-the-coordinates-of-the-vertices-of-a-pentagon-centered-at-the-origin
        center = (tile_size / 2, tile_size / 2)
        radius = (tile_size - 2 * margin) / 2
        pentagon_pts = []
        for i in range(5):
            x = int(center[0] + radius * np.cos(i * 2 * np.pi / 5 + np.radians(penta_rotation)))
            y = int(center[1] + radius * np.sin(i * 2 * np.pi / 5 + np.radians(penta_rotation)))
            pentagon_pts.append((x, y))
        pentagon_pts = np.array(pentagon_pts, np.int32)

        # Draw pentagon around the marker
        cv2.polylines(img, [pentagon_pts], isClosed=True, color=(0, 0, 255), thickness=1)

        # print("Marker Size: ", img_marker.size)
        # Marker Size scaled by 0.02
        return img.T


class TileMap:
    _map: np.ndarray

    def __init__(self, tile_size):
        self._map = 255 * np.ones((4, 3, tile_size, tile_size), dtype=np.uint8)

    def set_tile(self, pos: tuple, img: np.ndarray):
        assert np.all(self._map[pos[0], pos[1]].shape == img.shape)
        self._map[pos[0], pos[1]] = img

    def get_map_image(self):
        """ Merges the tile map into a single image """

        img = np.concatenate(self._map, axis=-1)
        img = np.concatenate(img, axis=-2)

        img = img.T

        return img



def main():
    marker_mm = 27.5  # Marker side length in mm
    penta_mm = 27.5 # Pentagon side length in mm
    filePath = f'../dodecahedron/{marker_mm}mm_marker_{penta_mm}mm_penta.png'
    marker_id = 0  # id of first tag (0 for reference, 11 for target)
    marker_in = marker_mm/25.4  # Marker side length in inches
    penta_in = penta_mm/25.4  # Pentagon side length in inches
    dpi = 72  # Page resolution in pixels per inch
    page_size = (8.5, 11)  # Page size in inches
    diam = 2 * (penta_in / 10 * np.sqrt(50 + 10*np.sqrt(5))) # Pentagon circumcircle diameter
    margin_in = 0.01  # Page margins in inches
    marker_size = round(dpi*marker_in)
    
    margin = round(dpi*margin_in)
    diam_size = round(dpi*diam)
    # tile_size = round(dpi*marker_in) + margin*2
    tile_size = diam_size + margin*2
    page_dims = (round(dpi*page_size[1]), round(dpi*page_size[0]))  # page size in rows x cols of pixels
    
    # tile_size = 255  # old
    # margin = int(0.1 * tile_size)  # old

    marker_factory = MarkerFactory()
    tile_map = TileMap(tile_size)

    # order = ['left', 'bottom', 'front', 'top' , 'back', 'right']

    # Define rotation angles for each position
    rotation_matrix = np.array([[-90, 90, 90],
                            [90, 90, 90],
                            [-90, -90, -90],
                            [-90, -90, 0]])
    ids = []
    for i in range(4):
        for j in range(3):
            if (i==3 and j==2):
                break
            
            angle = rotation_matrix[i][j]
            marker_img = marker_factory.create_marker(tile_size, marker_size, marker_id, margin, diam_size, penta_rotation=angle)
            tile_map.set_tile((i, j), marker_img)
            ids.append(marker_id)

            marker_id += 1

    tile_img = tile_map.get_map_image()

    tile_img_square = np.zeros((tile_size * 4, tile_size*4))
    tile_img_square[:, (tile_size//2):(-tile_size//2)] = tile_img
    
    full_page = 255 * np.ones(page_dims, dtype=np.uint8)
    full_page[:tile_size*4, :tile_size*3] = tile_img
    cv2.imwrite(filePath, full_page)



if __name__ == '__main__':
    main()
