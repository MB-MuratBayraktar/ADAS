import torch
import torchvision
import detectron2
from detectron2.utils.logger import setup_logger

#Note that the below is installed seperately
#!pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
#!pip install pyyaml==5.1

setup_logger()
# import some common libraries
import numpy as np
import tqdm
import cv2
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog
import time
import math

# Extract video properties
video = cv2.VideoCapture("/content/gdrive/MyDrive/test_media/dashcam-3_resized.mp4")
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frames_per_second = video.get(cv2.CAP_PROP_FPS)
num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# Initialize video writer
video_writer = cv2.VideoWriter('out.mp4', fourcc=cv2.VideoWriter_fourcc(*"mp4v"), fps=float(frames_per_second),
                               frameSize=(width, height), isColor=True)

# Initialize predictor
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE = 'cpu'
predictor = DefaultPredictor(cfg)

# Initialize visualizer
v = VideoVisualizer(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), ColorMode.IMAGE)


def runOnVideo(video, maxFrames):
    """ Runs the predictor on every frame in the video (unless maxFrames is given),
    and returns the frame with the predictions drawn.
    """

    readFrames = 0
    while True:
        hasFrame, frame = video.read()

        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        centerCoord = (int(width / 2), height)
        cv2.circle(frame, centerCoord, 25, (0, 0, 255), -1)

        if not hasFrame:
            break

        # detectron2.structures.BoxMode(1)

        # Get prediction results for this frame
        outputs = predictor(frame)
        instances = outputs['instances'][outputs['instances'].pred_classes == 0]
        detected_class_indexes = instances.pred_classes
        prediction_boxes = instances.pred_boxes
        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        class_catalog = metadata.thing_classes

        a = prediction_boxes.tensor.cpu()
        a = a.numpy()

        for i in a:
            lines = []
            cv2.line(frame, centerCoord, (i[2], i[3]), (0, 0, 255), 1)
            distance = cv2.norm((i[2] - i[3]) - centerCoord)
            lines.append(distance)
            print(lines[0])
            for x in lines:
                if x <= 20:
                    cv2.putText(frame, "Warning! a pedestrians is too close", (45, 25), cv2.FONT_HERSHEY_DUPLEX, 1,
                                color=(125, 246, 55), thickness=2)
                else:
                    cv2.putText(frame, "Safe zone", (45, 25), cv2.FONT_HERSHEY_DUPLEX, 1, color=(125, 246, 55),
                                thickness=2)

        # for idx, coordinates in enumerate(prediction_boxes):
        #     class_index = detected_class_indexes[idx]
        #     class_name = class_catalog[class_index]
        #     print(coordinates.item())

        # Make sure the frame is colored
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Draw a visualization of the predictions using the video visualizer
        visualization = v.draw_instance_predictions(frame, instances.to("cpu"))

        # Convert Matplotlib RGB format to OpenCV BGR format
        visualization = cv2.cvtColor(visualization.get_image(), cv2.COLOR_RGB2BGR)
        cv2_imshow(visualization)

        yield visualization

        readFrames += 1
        if readFrames > maxFrames:
            break


# Create a cut-off for debugging
num_frames = 800

# Enumerate the frames of the video
for visualization in tqdm.tqdm(runOnVideo(video, num_frames), total=num_frames):
    # Write test image
    cv2.imwrite('POSE detectron2.png', visualization)

    # Write to video file
    video_writer.write(visualization)

# Release resources
video.release()
video_writer.release()
cv2.destroyAllWindows()
