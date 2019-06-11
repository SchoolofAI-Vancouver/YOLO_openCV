import cv2 as cv
import argparse
import sys
import numpy as np
import os.path

# YOLO Params
conf_threshold = 0.5
nms_threshold = 0.4
input_width = 416
input_height = 416

parser = argparse.ArgumentParser(description='YOLO Object Detection using OpenCV')
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()

# Load label names
classFile = 'coco.names'
classes = None

with open(classFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# get config and weight files
model_config = 'yolov3.cfg'
model_weights = 'yolov3.weights'

# create the network using the loaded files
net = cv.dnn.readNetFromDarknet(model_config, model_weights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL)

# Get label names
def get_output_names(net):
    # Grab the names of all the layers in the network
    layer_names = net.getLayerNames()
    # Get the names from the output layer
    return [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box
def draw_pred(class_id, conf, left, top, right, bot):
    # Draw the bounding box
    cv.rectangle(frame, (left, top), (right, bot), (255,178,50), 3)
    label = '%.2f' % conf

    # Get the label and confidence score

    if classes:
        assert(class_id < len(classes))
        label = f'{classes[class_id]} : {label}'


    # Display the label at the top of the bounding box
    label_size, base_line = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, label_size[1])
    cv.rectangle(frame, (left, top - round(1.5 * label_size[1])),
                        (left + round(1.5 * label_size[0]), top + base_line),
                        (255,255,255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

# remove boxes with low confidence using non-maxima-suppression
def postprocess(frame, outputs):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    '''
    Scan through all the bounding boxes output from the network and 
    keep the ones with a high confidence score. Assign the box's class label as
    the class with the highest score
    '''

    class_ids = []
    confidences = []
    boxes = []

    for out in outputs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > conf_threshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform Non-Maximum Suppression to eliminate redundant overlapping boxes
    indices = cv.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        draw_pred(class_ids[i], confidences[i], left, top, left + width, top + height)

# Process inputs
window_name = 'YOLOv3 Object Detection using OpenCV'
cv.namedWindow(window_name, cv.WINDOW_NORMAL)
cv.resizeWindow(window_name, 1080, 720)

output_file = 'yolo_output.avi'

if (args.image):
    # opens the image file
    if not os.path.isfile(args.image):
        print('Input image file ', args.image, ' wasn\'t found')
        sys.exit(1)

    cap = cv.VideoCapture(args.image)
    output_file = args.image[:-4] + '_yolo_output.jpg'

elif (args.video):
    # open the video file
    if not os.path.isfile(args.video):
        print('Input video file ', args.video, ' wasn\'t found')
        sys.exit(1)

    cap = cv.VideoCapture(args.video)
    output_file = args.video[:-4] + '_yolo_output.avi'

else:
    cap = cv.VideoCapture(0)

if (not args.image):
    vid_writer = cv.VideoWriter(output_file, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, 
                                (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
                                round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

while cv.waitKey(1) < 0:
    # get frame from the video
    has_frame, frame = cap.read()

    # stop if we reached the end of the video
    if not has_frame:
        print('Done processing:')
        print('Output file is stored as: ', output_file)
        cv.waitKey(3000)
        # release device
        cap.release()
        break

    # create a 4d blob to feed into the network
    blob = cv.dnn.blobFromImage(
                frame, 1/255, (input_width, input_height), [0, 0, 0], 1, crop=False)
    # set the input to the network
    net.setInput(blob)
    # run the forward pass to get the outputs
    outs = net.forward(get_output_names(net))
    # remove the bounding boxes with low score
    postprocess(frame, outs)

    # get efficiency information
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    # Write the frame with the detection boxes
    if (args.image):
        cv.imwrite(output_file, frame.astype(np.uint8))
    else:
        vid_writer.write(frame.astype(np.uint8))

    cv.imshow(window_name, frame)