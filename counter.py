import cv2
import numpy as np
import rosbag
from cv_bridge import CvBridge
import json

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
hand_area_threshold = (500, 5000)  # Min and max area for hand detection
object_area_threshold = (10, 5000)  # Min and max area for object detection

# Initialize background subtractor
bg_subtractor_hand = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=25, detectShadows=False)
bg_subtractor_object = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

# Helper functions
def is_contour_in_roi(contour, roi):
    x, y, w, h = cv2.boundingRect(contour)
    contour_center = (x + w // 2, y + h // 2)
    roi_type = roi['type']
    roi_coords = roi['coords']
    if roi_type == 'rect':
        x_roi, y_roi, w_roi, h_roi = roi_coords
        return (x_roi <= contour_center[0] <= x_roi + w_roi) and (y_roi <= contour_center[1] <= y_roi + h_roi)
    elif roi_type == 'circle':
        cx, cy = roi_coords['center_x'], roi_coords['center_y']
        r = roi_coords['radius']
        return (contour_center[0] - cx) ** 2 + (contour_center[1] - cy) ** 2 <= r ** 2
    else:
        return False

# Read images from the rosbag file and process them in real-time
bag_file = '9_26_0930am.bag'
bag = rosbag.Bag(bag_file)

# Process each image frame-by-frame
for topic, msg, t in bag.read_messages(topics=['/arena_camera_node_0/image_raw']):
    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    # Apply background subtraction to detect hands/motion
    gray_frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    fg_mask_hand = bg_subtractor_hand.apply(gray_frame)
    # Threshold and clean up the mask
    _, fg_mask_hand = cv2.threshold(fg_mask_hand, 127, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    fg_mask_hand = cv2.morphologyEx(fg_mask_hand, cv2.MORPH_OPEN, kernel)

    # Find contours (possible hands)
    contours_hand, _ = cv2.findContours(fg_mask_hand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Apply background subtraction to detect objects
    fg_mask_object = bg_subtractor_object.apply(gray_frame)
    # Threshold and clean up the mask
    _, fg_mask_object = cv2.threshold(fg_mask_object, 127, 255, cv2.THRESH_BINARY)
    fg_mask_object = cv2.morphologyEx(fg_mask_object, cv2.MORPH_OPEN, kernel)

    # Find contours (possible objects)
    contours_object, _ = cv2.findContours(fg_mask_object, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw ROIs on the image
    image_with_rois = draw_rois(cv_image.copy(), rois)

    # Process each ROI
    for idx, roi in enumerate(rois):
        # Decrease buffer if needed
        if roi_buffers[idx] > 0:
            roi_buffers[idx] -= 1

        # Initialize flags
        hand_in_roi = False
        object_in_roi = False

        # Check for hands in ROI
        for contour in contours_hand:
            area = cv2.contourArea(contour)
            if hand_area_threshold[0] < area < hand_area_threshold[1]:
                if is_contour_in_roi(contour, roi):
                    hand_in_roi = True
                    # Draw the contour
                    cv2.drawContours(image_with_rois, [contour], -1, (255, 0, 0), 2)
                    break  # We can stop after finding one hand

        # Check for objects in ROI
        for contour in contours_object:
            area = cv2.contourArea(contour)
            if object_area_threshold[0] < area < object_area_threshold[1]:
                if is_contour_in_roi(contour, roi):
                    object_in_roi = True
                    # Draw the contour
                    cv2.drawContours(image_with_rois, [contour], -1, (0, 255, 255), 2)
                    break  # We can stop after finding one object

        # Update state and counting logic
        if roi_states[idx] == 'empty':
            if hand_in_roi:
                roi_states[idx] = 'hand_in'
            elif object_in_roi:
                roi_states[idx] = 'object_in'
                # Count immediately when object enters
                if roi_buffers[idx] == 0:
                    roi_counters[idx] += 1
                    roi_buffers[idx] = buffer_frames
                    print(f"Object entered ROI {idx}, count: {roi_counters[idx]}")
        elif roi_states[idx] == 'hand_in':
            if not hand_in_roi:
                # Hand left the ROI
                if roi_buffers[idx] == 0:
                    roi_counters[idx] += 1
                    roi_buffers[idx] = buffer_frames
                    print(f"Hand action in ROI {idx}, count: {roi_counters[idx]}")
                roi_states[idx] = 'empty'
        elif roi_states[idx] == 'object_in':
            if not object_in_roi:
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

    # Show the image with ROIs, detected hands, objects, and counts
    cv2.imshow("Video with ROIs and Detections", image_with_rois)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Release resources
cv2.destroyAllWindows()
bag.close()
