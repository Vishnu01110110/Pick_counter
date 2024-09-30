import cv2
import rosbag
from cv_bridge import CvBridge

# Initialize the CvBridge
bridge = CvBridge()

# Function to select ROIs
def select_rois(image):
    rois = []
    while True:
        roi = cv2.selectROI("Select ROI (Press Enter to confirm, Esc to exit)", image, fromCenter=False, showCrosshair=True)
        if roi == (0, 0, 0, 0):
            break
        key = cv2.waitKey(0)
        if key == 27:  # Esc key to exit
            break
        elif key == ord('o'):  # 'o' key to select a circular ROI
            center = (roi[0] + roi[2] // 2, roi[1] + roi[3] // 2)
            radius = min(roi[2], roi[3]) // 2
            rois.append(('circle', (center, radius)))
        else:
            rois.append(('rect', roi))
    cv2.destroyAllWindows()
    return rois

# Read images from the rosbag file
bag_file = '9_26_0930am.bag'
bag = rosbag.Bag(bag_file)

# Get the first image from the specified topic
cv_image = None
for topic, msg, t in bag.read_messages(topics=['/arena_camera_node_0/image_raw']):
    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    break  # We only need the first image for ROI selection

# Check if an image was found
if cv_image is not None:
    # Display the image and select ROIs
    rois = select_rois(cv_image)

    # Print the selected ROIs
    print("Selected ROIs:")
    for roi_type, roi in rois:
        if roi_type == 'rect':
            x, y, w, h = roi
            print(f"Rectangle ROI: x={x}, y={y}, w={w}, h={h}")
        elif roi_type == 'circle':
            center, radius = roi
            print(f"Circle ROI: center={center}, radius={radius}")

    # Draw the selected ROIs on the image
    for roi_type, roi in rois:
        if roi_type == 'rect':
            x, y, w, h = roi
            cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        elif roi_type == 'circle':
            center, radius = roi
            cv2.circle(cv_image, center, radius, (255, 0, 0), 2)

    # Display the final image with ROIs
    cv2.imshow("Final Image with ROIs", cv_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the ROIs to a file or use them as needed
    # For example, save to a JSON file
    import json
    with open('rois.json', 'w') as f:
        json.dump(rois, f)
else:
    print("No image found in the specified topic.")

# Close the rosbag
bag.close()