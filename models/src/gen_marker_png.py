import click
import cv2
import os
import numpy as np

def create_marker(size,id):
    # Margin 10% of size
    margin = int(0.1 * size)
    # print(size - (2*margin))
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    img_marker = cv2.aruco.generateImageMarker(aruco_dict,id,size-(2*margin))

    # white background
    img = size * np.ones((size, size), dtype=np.uint8)

    img[margin:-margin,margin:-margin] = img_marker
    return img

def main():
    # CHANGE THIS
    ID = 0
    marker_name = "aruco_marker_" + str(ID)
    path = os.path.join(os.path.expanduser('~'), 'Documents', 'capstone', 'models',marker_name,'materials','textures',"marker.png")
    
    print(path)
    marker = create_marker(250,ID)
    if not cv2.imwrite(filename=path,img=marker):
        raise Exception("Could not write image")
    else:
        print("Succesfully written to: ", path)


if __name__ == '__main__':
    main()