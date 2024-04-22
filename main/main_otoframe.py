# Title: Imports
# --------------
import os
import subprocess
from os.path import join, basename, splitext
import shutil
from keras.models import load_model
from keras.preprocessing import image as keras_image
import numpy as np
import time
import sys
import cv2


# Title: Start Timer, Handle Input Video Error, Create Empty Output Folder
# ------------------------------------------------------------------------

# Record execution time
start_time = time.time()

# Handle if user does not provide video path
if len(sys.argv) < 2:
    print("Please input video path: python OtoFrame.py <video_path>")
    sys.exit(1)

# Input video path from parameter provided by user
input_video_path = sys.argv[1]

# Check if the provided path exists
if not os.path.exists(input_video_path):
    print("Error: Video path does not exist")
    sys.exit(1)

# Set output folder for extracted images to be the same as the input video path
# Extract file name from video path and get first argument [0] without extension file
output_folder = os.path.join(os.path.dirname(input_video_path), splitext(basename(input_video_path))[0])

# Handle if output folder already exists
if os.path.exists(output_folder):
    while True:

        # Store user input in lower case
        response = str(
            input(f"The folder '{output_folder}' already exists. Do you want to overwrite it? (yes/no): ").lower())

        if response == 'yes':
            # Remove the existing folder
            shutil.rmtree(output_folder)
            break

        elif response == 'no':
            # Exit the process
            print("Process terminated")
            sys.exit(1)

        else:
            print("Invalid input. Please enter 'yes' or 'no'")

# Create output folder
os.mkdir(output_folder)


# Title: Video Re-Encoding by Changing GOP Size
# ---------------------------------------------

# GOP size
gop_size = 35

# Output new gop_video path
output_video_path = os.path.join(output_folder, basename(input_video_path))

# FFmpeg command to change GOP size / re-encoding the video
subprocess.run(['ffmpeg', '-i', input_video_path, '-g', str(gop_size), output_video_path], check=True)


# Title: FFmpeg Keyframes Extraction
# ----------------------------------

# FFmpeg for extracting frames
ffmpeg_command = f"ffmpeg -i \"{output_video_path}\" -vf \"select='eq(pict_type,I)'\" -q:v 1 -vsync vfr \"{output_folder}/out-%02d.jpeg\""

# Run ffmpeg
os.system(ffmpeg_command)

# Remove the output video file
os.remove(output_video_path)


# Title: Pre-Processing to Remove Duplicates and Black Frames
# -----------------------------------------------------------

# Threshold for duplicates
threshold = 20000000

# Load all images from folder in sorted sequential order
images = sorted(os.listdir(output_folder))

previous_frame = None

for image_name in images:
    # Get image path
    image_location = os.path.join(output_folder, image_name)

    # If file exists
    if os.path.isfile(image_location):
        # Get current frame
        current_frame = cv2.imread(image_location, cv2.IMREAD_GRAYSCALE)

        if previous_frame is not None:
            # Check if current frame is similar to the previous one
            diff = cv2.absdiff(current_frame, previous_frame).sum()
        else:
            # Avoid deleting first frame
            diff = threshold + 1

        # Count number of black pixels
        black_pixels = np.sum(current_frame == 0)
        # Extract height and width of the image
        height, width = current_frame.shape[:2]

        # Calculate percentage of black pixels
        total_pixels = height * width
        black_pixels_percentage = black_pixels / total_pixels

        if black_pixels_percentage > 0.20 or diff <= threshold:
            # Delete black frame or duplicate frame
            os.remove(image_location)
        else:
            previous_frame = current_frame


# Title: Frames Classification and Handling of Output
# ---------------------------------------------------

# Load the trained model
model = load_model('otoframe_mobilenetv2_model.h5')

# Get a list of image files in the folder
image_files = [image for image in os.listdir(output_folder) if image.endswith('.jpeg')]

# Check if folder is empty
if not image_files:
    print("No images found, folder is empty")

else:
    # List to store images names along with their probability of belonging to "unsuitable"
    image_probabilities = []

    # Counter to store number of images belonging to "Suitable" images
    counter = 0

    # Loop through images array
    for image_file in image_files:
        # Image path
        image_path = os.path.join(output_folder, image_file)

        # Load image, convert to numpy array
        loaded_image = keras_image.load_img(image_path, target_size=(224, 224))
        image_array = keras_image.img_to_array(loaded_image)

        # Image array to input array(for model input)
        input_array = np.array([image_array])
        # Normalize input for better performance
        input_array = input_array / 255.0

        # Predict the class of the image from range 0 to 1, 1 being 100%
        predictions = model.predict(input_array)

        # Print image and its probability
        print(f"Image: {image_file}, Probability: {predictions[0]}")

        # Increment counter if probability is less than 0.5, "suitable" image found
        if predictions[0] < 0.9:
            counter += 1

        # Add image file name along with its probability to the list
        image_probabilities.append((image_file, predictions[0]))


    # Set a limit for the number of frames to retain after FFmpeg and pre-processing (before deleting frames after CNN classification)
    min_frames_to_keep = 7

    # If "suitable" frames are less than "min_frames_to_keep", sort the list and keep the first "min_frames_to_keep" frames with the highest probability of being "suitable"
    if counter < min_frames_to_keep:
        image_probabilities.sort(key=lambda x: x[1])

        # Delete all images after the min_frames_to_keep index
        for image_file, probability in image_probabilities[min_frames_to_keep:]:
            os.remove(os.path.join(output_folder, image_file))
            print(f"{image_file} deleted with probability: {probability}")

    else:
        # Delete all images with probability less than 0.9, the "unsuitable" images
        for image_file, probability in image_probabilities:
            if probability > 0.9:
                os.remove(os.path.join(output_folder, image_file))
                print(f"{image_file} deleted with probability: {probability}")

    # Delete all images with probability 1
    for image_file, probability in image_probabilities:
        if probability == 1:
            try:
                os.remove(os.path.join(output_folder, image_file))
                print(f"{image_file} deleted with probability: {probability}")
            except FileNotFoundError:
                print(f"Error: {image_file} already deleted.")


# Title: Summary of Execution Time
# --------------------------------

end_time = time.time()
time_taken = end_time - start_time
print("Total Execution Time:", time_taken, "seconds")
