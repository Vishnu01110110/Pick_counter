import cv2
import numpy as np
import rosbag
from cv_bridge import CvBridge
import json
from ultralytics import YOLO

# Initialize the CvBridge
bridge = CvBridge()

# Load YOLOv10 model (using the Nano model for performance)
model = YOLO('yolov10n.pt')

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

# Initialize counters and trackers
roi_counters = [0 for _ in rois]  # A counter for each ROI
person_trackers = {}
person_id_counter = 0

# Parameters
buffer_frames = 5  # Number of frames to buffer before counting again

# Helper functions
def detect_persons(image):
    results = model.predict(image)
    persons = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls == 0 and conf > 0.5:  # Class 0 is 'person'
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1
                persons.append((x1, y1, w, h, conf))
    return persons

def initialize_person_trackers(persons):
    global person_id_counter
    # Sort persons by area (largest to smallest) to select the most prominent ones
    persons = sorted(persons, key=lambda p: p[2] * p[3], reverse=True)
    # Take the first 3 persons
    selected_persons = persons[:3]
    for (x, y, w, h, conf) in selected_persons:
        person_trackers[person_id_counter] = {
            'id': person_id_counter,
            'bbox': (x, y, w, h),
            'buffer': 0,
            'counted': False
        }
        person_id_counter += 1

def update_person_trackers(persons):
    new_trackers = {}
    for pid, pdata in person_trackers.items():
        # Try to match existing tracker with detected persons
        best_match = None
        best_distance = float('inf')
        for (x, y, w, h, conf) in persons:
            person_center = (x + w // 2, y + h // 2)
            tracker_center = (pdata['bbox'][0] + pdata['bbox'][2] // 2,
                              pdata['bbox'][1] + pdata['bbox'][3] // 2)
            distance = np.linalg.norm(np.array(person_center) - np.array(tracker_center))
            if distance < best_distance:
                best_distance = distance
                best_match = (x, y, w, h)

        if best_match and best_distance < 50:
            # Update tracker
            pdata['bbox'] = best_match[:4]
            new_trackers[pid] = pdata
        else:
            # Tracker lost; keep the last known position
            new_trackers[pid] = pdata

    return new_trackers

def update_counts_and_states():
    for idx, roi in enumerate(rois):
        roi_type = roi['type']
        roi_coords = roi['coords']

        # Get ROI bounding box
        if roi_type == 'rect':
            roi_bbox = (roi_coords[0], roi_coords[1], roi_coords[0] + roi_coords[2], roi_coords[1] + roi_coords[3])
        elif roi_type == 'circle':
            # For circle, approximate with bounding box
            cx, cy = roi_coords['center_x'], roi_coords['center_y']
            r = roi_coords['radius']
            roi_bbox = (cx - r, cy - r, cx + r, cy + r)
        else:
            continue  # Unsupported ROI type

        # Update person trackers
        for pid, pdata in person_trackers.items():
            x, y, w, h = pdata['bbox']
            person_bbox = (x, y, x + w, y + h)

            # Check if person bbox intersects with ROI bbox
            intersection_area = compute_intersection_area(person_bbox, roi_bbox)
            if intersection_area > 0:
                if not pdata['counted']:
                    roi_counters[idx] += 1
                    pdata['counted'] = True
                    pdata['buffer'] = buffer_frames
                    print(f"Person {pid} counted in ROI {idx}")
            else:
                if pdata['buffer'] > 0:
                    pdata['buffer'] -= 1
                else:
                    pdata['counted'] = False  # Reset counted status when outside ROI

def compute_intersection_area(boxA, boxB):
    # boxA and boxB are (x1, y1, x2, y2)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    return interWidth * interHeight

# Read images from the rosbag file and process them in real-time
bag_file = '9_26_0930am.bag'
bag = rosbag.Bag(bag_file)

# Initialize person trackers in the first few frames
initial_frames = 5
frame_count = 0

# Process each image frame-by-frame
for topic, msg, t in bag.read_messages(topics=['/arena_camera_node_0/image_raw']):
    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    frame_count += 1

    # Detect persons using YOLOv10
    persons = detect_persons(cv_image)

    if frame_count <= initial_frames:
        # Initialize person trackers in the initial frames
        initialize_person_trackers(persons)
    else:
        # Update person trackers
        person_trackers = update_person_trackers(persons)

    # Update counts and states for persons
    update_counts_and_states()

    # Draw ROIs on the image
    image_with_rois = draw_rois(cv_image.copy(), rois)

    # Draw detected persons with IDs
    for pid, pdata in person_trackers.items():
        x, y, w, h = pdata['bbox']
        cv2.rectangle(image_with_rois, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(image_with_rois, f"Person {pid}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

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

    # Show the image with ROIs, detected persons, and counts
    cv2.imshow("Video with ROIs and Detections", image_with_rois)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Release resources
cv2.destroyAllWindows()
bag.close()
