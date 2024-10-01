import cv2
import numpy as np
import rosbag
from cv_bridge import CvBridge
import json
import openpifpaf

# Initialize the CvBridge
bridge = CvBridge()

# Function to draw ROIs on an image
def draw_rois(image, rois):
    for idx, roi in enumerate(rois):
        roi_type = roi['type']
        roi_coords = roi['coords']
        color = (0, 255, 0)
        if roi_type == 'rect':
            x, y, w, h = roi_coords
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            # Label the ROI
            cv2.putText(image, f"Bin {idx}", (x - 50, y + h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        elif roi_type == 'circle':
            center = (roi_coords['center_x'], roi_coords['center_y'])
            radius = roi_coords['radius']
            cv2.circle(image, center, radius, color, 2)
            # Label the ROI
            cv2.putText(image, f"Bin {idx}", (center[0] - radius - 50, center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return image

# Load the ROIs from the JSON file
with open('rois.json', 'r') as f:
    rois = json.load(f)

# Initialize counters and states
roi_counters = [0 for _ in rois]  # A counter for each ROI
roi_states = ['empty' for _ in rois]  # State per ROI
roi_buffers = [0 for _ in rois]  # Buffer frames per ROI

# Parameters
buffer_frames = 3  # Number of frames to buffer before counting again

# Initialize background subtractor for objects if needed
# bg_subtractor_object = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

# Initialize OpenPifPaf predictor
predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k16')

# Helper functions
def is_hand_in_roi(keypoints, roi):
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]

    roi_type = roi['type']
    roi_coords = roi['coords']

    for wrist in [left_wrist, right_wrist]:
        x, y, conf = wrist
        if conf < 0.5:
            continue
        x, y = int(x), int(y)
        if roi_type == 'rect':
            x_roi, y_roi, w_roi, h_roi = roi_coords
            if x_roi <= x <= x_roi + w_roi and y_roi <= y <= y_roi + h_roi:
                return True
        elif roi_type == 'circle':
            cx, cy = roi_coords['center_x'], roi_coords['center_y']
            r = roi_coords['radius']
            if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                return True
    return False

# Read images from the rosbag file and process them in real-time
bag_file = '9_26_0930am.bag'
bag = rosbag.Bag(bag_file)

# Process each image frame-by-frame
for topic, msg, t in bag.read_messages(topics=['/arena_camera_node_0/image_raw']):
    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    # Perform pose estimation
    predictions, _ = predictor.numpy_image(cv_image)

    # Draw ROIs on the image
    image_with_rois = draw_rois(cv_image.copy(), rois)

    # Process each ROI
    for idx, roi in enumerate(rois):
        # Decrease buffer if needed
        if roi_buffers[idx] > 0:
            roi_buffers[idx] -= 1

        # Initialize flags
        hand_in_roi = False

        # Check for hands in ROI using pose estimation
        for pred in predictions:
            keypoints = pred.data
            if is_hand_in_roi(keypoints, roi):
                hand_in_roi = True
                # Draw skeleton on the image
                openpifpaf.show.annotation_painter.AnnotationPainter().annotations(
                    image_with_rois, [pred]
                )
                break  # We can stop after finding one hand in ROI

        # Update state and counting logic
        if roi_states[idx] == 'empty':
            if hand_in_roi:
                roi_states[idx] = 'hand_in'
        elif roi_states[idx] == 'hand_in':
            if not hand_in_roi:
                if roi_buffers[idx] == 0:
                    roi_counters[idx] += 1
                    roi_buffers[idx] = buffer_frames
                    print(f"Hand action in ROI {idx}, count: {roi_counters[idx]}")
                roi_states[idx] = 'empty'

    # Display the counters on the image
    for idx, count in enumerate(roi_counters):
        # Get ROI position for placing the counter
        roi = rois[idx]
        roi_type = roi['type']
        roi_coords = roi['coords']
        if roi_type == 'rect':
            x, y, w, h = roi_coords
            position = (x - 50, y + h // 2 + 20)
        elif roi_type == 'circle':
            center_x = roi_coords['center_x']
            center_y = roi_coords['center_y']
            radius = roi_coords['radius']
            position = (center_x - radius - 50, center_y + 20)
        cv2.putText(image_with_rois, f"Count: {count}", position,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show the image with ROIs, detected hands, and counts
    cv2.imshow("Video with ROIs and Detections", image_with_rois)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Release resources
cv2.destroyAllWindows()
bag.close()
