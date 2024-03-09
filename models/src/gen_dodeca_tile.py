import click
import cv2
from cv2 import aruco
import os
import numpy as np
import yaml

class MarkerFactory:

    @staticmethod
    def create_marker(size, id, margin):
        # aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_100)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

        # white background
        img = 255 * np.ones((size, size), dtype=np.uint8)
        # img_marker = aruco.drawMarker(aruco_dict, id, size - 2 * margin,borderBits=1)
        img_marker = aruco.generateImageMarker(aruco_dict, id, size - 2 * margin)
        # add marker centered
        img[margin:-margin, margin:-margin] = img_marker

        print("Marker Size: ", img_marker.size)
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
    tile_size = 255
    path = '/home/astirl/Documents/capstone/models/dodecahedron/'
    
    margin = int(0.1 * tile_size)

    marker_factory = MarkerFactory()
    tile_map = TileMap(tile_size)

    order = ['left', 'botton', 'front', 'top' , 'back', 'right']

    ids = []
    reference_marker = 11
    true_marker = 0
    marker_id = reference_marker
    for i in range(4):
        for j in range(3):
            if (i==3 and j==2):
                break

            marker_img = marker_factory.create_marker(tile_size, marker_id, margin)
            tile_map.set_tile((i, j), marker_img)
            ids.append(marker_id)

            marker_id += 1

    tile_img = tile_map.get_map_image()

    tile_img_square = np.zeros((tile_size * 4, tile_size*4))
    tile_img_square[:, (tile_size//2):(-tile_size//2)] = tile_img
    cv2.imwrite(path+'reference_board.png',tile_img)



if __name__ == '__main__':
    main()
