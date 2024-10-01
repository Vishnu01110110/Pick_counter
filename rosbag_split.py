#!/usr/bin/env python

import rosbag
import sys
import os
import rospy

def split_bag(input_bag, output_bag_path, split_duration):
    # Open the large bag file
    bag = rosbag.Bag(input_bag)
    
    # Get the start time of the bag
    start_time = bag.get_start_time()
    
    # Calculate the end time based on the duration you want
    end_time = start_time + split_duration
    
    # Create a new bag file for the split
    with rosbag.Bag(output_bag_path, 'w') as outbag:
        print(f"Creating {output_bag_path} from time {start_time} to {end_time}")
        
        # Write messages to the new bag file for the specified duration
        for topic, msg, t in bag.read_messages(start_time=rospy.Time(start_time), end_time=rospy.Time(end_time)):
            outbag.write(topic, msg, t)
    
    bag.close()
    print(f"Created {output_bag_path} with {split_duration} seconds of data.")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: rosbag_split.py <input_bag.bag> <output_bag.bag> <split_duration_in_seconds>")
        sys.exit(1)
    
    input_bag = sys.argv[1]
    output_bag = sys.argv[2]
    split_duration = float(sys.argv[3])  # In seconds
    
    split_bag(input_bag, output_bag, split_duration)
