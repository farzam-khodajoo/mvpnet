"""All utilities related to opencv will be here."""
import cv2
import numpy as np

def show(image: np.array, title="CV"):
    cv2.imshow(title, cv2.resize(image, (500, 500)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()