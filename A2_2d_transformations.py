import cv2
import numpy as np
import math

def get_transformed_image( img, M):
    plane = np.full((801, 801),255).astype(np.uint8)
    plane[400-int(img.shape[0]/2):400+int(img.shape[0]/2)+1, 400-int(img.shape[1]/2):400+int(img.shape[1]/2)+1] = img

    new_plane = np.full((801, 801),255).astype(np.uint8)

    for i in range(int(-img.shape[0]/2), int(img.shape[0]/2)):
        for j in range(int(-img.shape[1]/2), int(img.shape[1]/2)):
            try:
                transformed_idx = np.dot(M, np.array([[i], [j], [1]]))
                x, y = int(transformed_idx[0] + 400), int(transformed_idx[1] + 400)
                new_plane[x][y] = plane[i + 400][j + 400]
            except:
                pass

    return new_plane

if __name__=="__main__":

    img = cv2.imread('./CV_Assignment_2_images/smile.png', cv2.IMREAD_GRAYSCALE)

    I = np.array([[1,0,0],[0,1,0],[0,0,1]])
    a = np.array([[1,0,0],[0,1,-5],[0,0,1]])
    d = np.array([[1,0,0],[0,1,5],[0,0,1]])
    w = np.array([[1, 0, -5], [0, 1, 0], [0, 0, 1]])
    s = np.array([[1, 0, 5], [0, 1, 0], [0, 0, 1]])
    r = np.array([[math.cos(math.pi * (5/180)), -math.sin(math.pi * (5/180)), 0], [math.sin(math.pi * (5/180)), math.cos(math.pi * (5/180)), 0], [-5, 0, 1]])
    R = np.array([[math.cos(math.pi * (-5/180)), -math.sin(math.pi * (-5/180)), 0], [math.sin(math.pi * (-5/180)), math.cos(math.pi * (-5/180)), 0], [-5, 0, 1]])
    f = np.array([[1,0,0],[0,-1,0],[0,0,1]])
    F = np.array([[-1,0,0],[0,1,0],[0,0,1]])
    x = np.array([[1, 0, 0], [0, 0.95, 0], [0, 0, 1]])
    X = np.array([[1, 0, 0], [0, 1.05, 0], [0, 0, 1]])
    y = np.array([[0.95, 0, 0], [0, 1, 0], [0, 0, 1]])
    Y = np.array([[1.05, 0, 0], [0, 1, 0], [0, 0, 1]])

    img = get_transformed_image(img, I)
    init = img.copy()
    img_show = img.copy()
    cv2.arrowedLine(img_show, (400, 800), (400, 0), 0, 2, tipLength=0.01)
    cv2.arrowedLine(img_show, (0, 400), (800, 400), 0, 2, tipLength=0.01)
    cv2.imshow('transform',img_show)

    while(True):
        action = cv2.waitKey()
        cv2.destroyAllWindows()
        if action == ord('a'):
            img = get_transformed_image(img, a)
            img_show = img.copy()
            cv2.arrowedLine(img_show, (400, 800), (400, 0), 0, 2, tipLength=0.01)
            cv2.arrowedLine(img_show, (0, 400), (800, 400), 0, 2, tipLength=0.01)
            cv2.imshow('transform',img_show)
        elif action == ord('d'):
            img = get_transformed_image(img, d)
            img_show = img.copy()
            cv2.arrowedLine(img_show, (400, 800), (400, 0), 0, 2, tipLength=0.01)
            cv2.arrowedLine(img_show, (0, 400), (800, 400), 0, 2, tipLength=0.01)
            cv2.imshow('transform',img_show)
        elif action == ord('w'):
            img = get_transformed_image(img, w)
            img_show = img.copy()
            cv2.arrowedLine(img_show, (400, 800), (400, 0), 0, 2, tipLength=0.01)
            cv2.arrowedLine(img_show, (0, 400), (800, 400), 0, 2, tipLength=0.01)
            cv2.imshow('transform', img_show)
        elif action == ord('s'):
            img = get_transformed_image(img, s)
            img_show = img.copy()
            cv2.arrowedLine(img_show, (400, 800), (400, 0), 0, 2, tipLength=0.01)
            cv2.arrowedLine(img_show, (0, 400), (800, 400), 0, 2, tipLength=0.01)
            cv2.imshow('transform', img_show)
        elif action == ord('r'):
            img = get_transformed_image(img, r)
            img_show = img.copy()
            cv2.arrowedLine(img_show, (400, 800), (400, 0), 0, 2, tipLength=0.01)
            cv2.arrowedLine(img_show, (0, 400), (800, 400), 0, 2, tipLength=0.01)
            cv2.imshow('transform', img_show)
        elif action == ord('R'):
            img = get_transformed_image(img, R)
            img_show = img.copy()
            cv2.arrowedLine(img_show, (400, 800), (400, 0), 0, 2, tipLength=0.01)
            cv2.arrowedLine(img_show, (0, 400), (800, 400), 0, 2, tipLength=0.01)
            cv2.imshow('transform', img_show)
        elif action == ord('f'):
            img = get_transformed_image(img, f)
            img_show = img.copy()
            cv2.arrowedLine(img_show, (400, 800), (400, 0), 0, 2, tipLength=0.01)
            cv2.arrowedLine(img_show, (0, 400), (800, 400), 0, 2, tipLength=0.01)
            cv2.imshow('transform', img_show)
        elif action == ord('F'):
            img = get_transformed_image(img, F)
            img_show = img.copy()
            cv2.arrowedLine(img_show, (400, 800), (400, 0), 0, 2, tipLength=0.01)
            cv2.arrowedLine(img_show, (0, 400), (800, 400), 0, 2, tipLength=0.01)
            cv2.imshow('transform', img_show)
        elif action == ord('x'):
            img = get_transformed_image(img, x)
            img_show = img.copy()
            cv2.arrowedLine(img_show, (400, 800), (400, 0), 0, 2, tipLength=0.01)
            cv2.arrowedLine(img_show, (0, 400), (800, 400), 0, 2, tipLength=0.01)
            cv2.imshow('transform', img_show)
        elif action == ord('X'):
            img = get_transformed_image(img, X)
            img_show = img.copy()
            cv2.arrowedLine(img_show, (400, 800), (400, 0), 0, 2, tipLength=0.01)
            cv2.arrowedLine(img_show, (0, 400), (800, 400), 0, 2, tipLength=0.01)
            cv2.imshow('transform', img_show)
        elif action == ord('y'):
            img = get_transformed_image(img, y)
            img_show = img.copy()
            cv2.arrowedLine(img_show, (400, 800), (400, 0), 0, 2, tipLength=0.01)
            cv2.arrowedLine(img_show, (0, 400), (800, 400), 0, 2, tipLength=0.01)
            cv2.imshow('transform', img_show)
        elif action == ord('Y'):
            img = get_transformed_image(img, Y)
            img_show = img.copy()
            cv2.arrowedLine(img_show, (400, 800), (400, 0), 0, 2, tipLength=0.01)
            cv2.arrowedLine(img_show, (0, 400), (800, 400), 0, 2, tipLength=0.01)
            cv2.imshow('transform', img_show)
        elif action == ord('H'):
            img = init
            img_show = img.copy()
            cv2.arrowedLine(img_show, (400, 800), (400, 0), 0, 2, tipLength=0.01)
            cv2.arrowedLine(img_show, (0, 400), (800, 400), 0, 2, tipLength=0.01)
            cv2.imshow('transform', img_show)
        elif action == ord('Q'):
            break
        else:
            cv2.imshow('transform', img_show)
            continue
