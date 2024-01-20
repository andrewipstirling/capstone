import click
import cv2
import os

def create_marker(size,id):

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    img_marker = cv2.aruco.generateImageMarker(aruco_dict,id,size)
    return img_marker

def main():
    ID = 0
    marker_name = "aruco_marker_" + str(ID)
    path = os.path.join(os.path.expanduser('~'), 'Documents', 'capstone', 'aruco_box',marker_name,'materials','textures',marker_name+".png")
    
    print(path)
    marker = create_marker(255,ID)
    if not cv2.imwrite(filename=path,img=marker):
        raise Exception("Could not write image")
    else:
        print("Succesfully written to: ", path)


if __name__ == '__main__':
    main()