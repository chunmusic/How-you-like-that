#!/usr/bin/env python3

import streamlit as st
import cv2
import numpy as np
from PIL import Image


def main():
    # img_test = cv2.imread("test_img/bp0.jpg")

    pil_image = Image.open('test_img/bp0.jpg')
    # open_cv_image = np.array(pil_image) 
    # # Convert RGB to BGR 
    # open_cv_image = open_cv_image[:, :, ::-1].copy() 

    cv2.cvtColor(img_test,cv2.COLOR_BGR2RGB)
    # img_to_detect = cv2.cvtColor(np.array(img_to_detect),cv2.COLOR_BRG2RGB)

    cv2.imshow("test",img_test)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()