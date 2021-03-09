#!/usr/bin/env python3

import numpy as np
import streamlit as st
import footer
import os, urllib, cv2
from PIL import Image
import sys
print(sys.getrecursionlimit())

def main():

    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}

            .reportview-container .main .block-container{{
                max-width: {max_width}px;
                padding-top: {padding_top}rem;
                padding-right: {padding_right}rem;
                padding-left: {padding_left}rem;
                padding-bottom: {padding_bottom}rem;
            }}
            .reportview-container .main {{
                color: {COLOR};
                background-color: {BACKGROUND_COLOR};
            }}

            </style>
            """

    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
    
    readme_text = st.markdown(get_file_content_as_string("instructions.md"))

    uploaded_file = st.file_uploader("Upload Image")


    if (uploaded_file is not None) and (uploaded_file != uploaded_file):

        img_cv = None

        image = Image.open(uploaded_file)

        img_array = np.array(image)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        for filename in EXTERNAL_DEPENDENCIES.keys():
            download_file(filename)

        img_out, print_label, print_confidence = detection(img_cv)

        img_to_detect = cv2.cvtColor(img_out,cv2.COLOR_BGR2RGB)

        st.image(img_to_detect, use_column_width=True)

        uploaded_file = None

        st.write(" ")

        if len(print_label)!=0:
            if len(print_label) == len(print_confidence):
                
                for i in range(len(print_label)):
                    st.write("You're like: {} {:.2f}%".format(print_label[i],print_confidence[i]*100))

        else:
            st.write("No one is like you")




@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/chunmusic/How-you-like-that/master/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")


# This file downloader demonstrates Streamlit animation.
def download_file(file_path):
    # Don't download the file twice. (If possible, verify the download using the file length.)
    if os.path.exists(file_path):
        if "size" not in EXTERNAL_DEPENDENCIES[file_path]:
            return
        elif os.path.getsize(file_path) == EXTERNAL_DEPENDENCIES[file_path]["size"]:
            return

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % file_path)
        progress_bar = st.progress(0)
        with open(file_path, "wb") as output_file:
            with urllib.request.urlopen(EXTERNAL_DEPENDENCIES[file_path]["url"]) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning("Downloading %s... (%6.2f/%6.2f MB)" %
                        (file_path, counter / MEGABYTES, length / MEGABYTES))
                    progress_bar.progress(min(counter / length, 1.0))

    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()


def detection(image):

    img_to_detect = image
    img_height = img_to_detect.shape[0]
    img_width = img_to_detect.shape[1]

    #Transform image to binary in order to build model
    img_blob = cv2.dnn.blobFromImage(img_to_detect,0.003922,(416,416),swapRB = True, crop=False)


    class_labels = ["jennie","jisoo","lisa","rose"]

    class_colors = ["0,255,0","0,0,255","255,0,0","255,255,0","0,255,255"]
    class_colors = [np.array(every_color.split(",")).astype("int") for every_color in class_colors]
    class_colors = np.array(class_colors)
    class_colors = np.tile(class_colors,(16,1))

    yolo_model = cv2.dnn.readNetFromDarknet("bp_yolov4.cfg",'bp_yolov4_best.weights')
    yolo_layers = yolo_model.getLayerNames()

    yolo_output_layer = [yolo_layers[yolo_layer[0] - 1] for yolo_layer in yolo_model.getUnconnectedOutLayers()]

    yolo_model.setInput(img_blob)

    obj_detection_layers = yolo_model.forward(yolo_output_layer)

    class_ids_list = []
    boxes_list = []
    confidences_list = []

    print_predict_label = []
    print_predict_confidence = []
    
    for object_detection_layer in obj_detection_layers:
        # loop over the detections
        for object_detection in object_detection_layer:
            
            # obj_detections[1 to 4] => will have the two center points, box width and box height
            # obj_detections[5] => will have scores for all objects within bounding box
            all_scores = object_detection[5:]
            predicted_class_id = np.argmax(all_scores)
            prediction_confidence = all_scores[predicted_class_id]
        
            # take only predictions with confidence more than 20%
            if prediction_confidence > 0.90:
                #get the predicted label
                predicted_class_label = class_labels[predicted_class_id]
                #obtain the bounding box co-oridnates for actual image from resized image size
                bounding_box = object_detection[0:4] * np.array([img_width, img_height, img_width, img_height])
                (box_center_x_pt, box_center_y_pt, box_width, box_height) = bounding_box.astype("int")
                start_x_pt = int(box_center_x_pt - (box_width / 2))
                start_y_pt = int(box_center_y_pt - (box_height / 2))
                
                class_ids_list.append(predicted_class_id)
                confidences_list.append(float(prediction_confidence))
                boxes_list.append([start_x_pt,start_y_pt,int(box_width),int(box_height)])
                
    max_value_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)

    # loop through the final set of detections remaining after NMS and draw bounding box and write text
    for max_valueid in max_value_ids:
        max_class_id = max_valueid[0]
        box = boxes_list[max_class_id]
        start_x_pt = box[0]
        start_y_pt = box[1]
        box_width = box[2]
        box_height = box[3]
        
        #get the predicted class id and label
        predicted_class_id = class_ids_list[max_class_id]
        predicted_class_label = class_labels[predicted_class_id]
        prediction_confidence = confidences_list[max_class_id]

        #For printing confidence value:
        print_predict_label.append(predicted_class_label)
        print_predict_confidence.append(prediction_confidence)
        
        end_x_pt = start_x_pt + box_width
        end_y_pt = start_y_pt + box_height
        
        #get a random mask color from the numpy array of colors
        box_color = class_colors[predicted_class_id]
        
        #convert the color numpy array as a list and apply to text and box
        box_color = [int(c) for c in box_color]
        
        # print the prediction in console
        predicted_class_label = "{}: {:.2f}%".format(predicted_class_label, prediction_confidence * 100)
        # print("predicted object {}".format(predicted_class_label))
        
        # draw rectangle and text in the image
        cv2.rectangle(img_to_detect, (start_x_pt, start_y_pt), (end_x_pt, end_y_pt), box_color, 4)
        cv2.putText(img_to_detect, predicted_class_label, (start_x_pt, start_y_pt-5), cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 4)



    return img_to_detect,print_predict_label,print_predict_confidence


EXTERNAL_DEPENDENCIES = {
    "bp_yolov4_best.weights": {
        "url": "https://sandbox.zenodo.org/record/740192/files/bp_yolov4_best.weights",
        "size": 256080600
    },
    "bp_yolov4.cfg": {
        "url": "https://raw.githubusercontent.com/chunmusic/How-you-like-that/master/bp_yolov4.cfg",
        "size": 12223
    }
}

if __name__ == "__main__":

    main()

    footer.footer()
