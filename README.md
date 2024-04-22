# OtoFrame: A Software Solution for Keyframe Extraction in Otology Endoscopic Videos

OtoFrame is an advanced software tool that leverages artificial intelligence (AI) to extract keyframes showing the tympanic membrane from otology endoscopic videos. Its purpose is to deliver clinically relevant eardrum images to the user for assessment.

## Demo



**The Process of OtoFrame Involves:**
1. Input: Video.
2. Re-encoding: Increase the number of I frames (Keyframes).
3. Keyframe Extraction: Extracts keyframes using FFmpeg.
4. Pre-processing: Removes duplicate and black frames.
5. Classification: Applies a binary CNN model for classifying images as "relevant" or "irrelevant".
6. Filtering: Removes "irrelevant" frames.
7. Output: A folder containing frames showing the tympanic membrane of the patient.


## Installation Requirements


1. **Python Recommended Interpreter: Python 3.11.8**
2. **Libraries:**

   ```bash
   Must:
   pip install keras==2.15.0 (via pip3).
   pip install tensorflow==2.15.0 (via pip3).
   pip install numpy.
   pip install opencv-python.


   Additional Libraries Required for Model Training in Addition to the Must:
   pip install matplotlib.
   pip install scikit-learn.
   pip install seaborn.
   ```
3. **Install FFmpeg:**

   For detailed instructions on installing FFmpeg please visit the official FFmpeg download page: 
   https://ffmpeg.org/download.html

## Folder Structure 

**main:**
           
* "main_otoframe.py": Handles keyframe extraction.

* "otoframe_mobilenetv2_model.h5": Transfer learning MobileNetV2 model with learned weights for classifying the keyframes.

**transfer_learning_mobilenetv2_script:**
* "transfer_learning_mobilenetv2.py": Contains the code/script for training the Transfer Learning MobileNetV2 model.


## Running Instructions
**"main_otoframe.py":**

1. Open your terminal/command prompt.
2. Navigate to the directory containing "main_otoframe.py".
3. Execute the script using Python and as a second argument provide the path to the video you want to extract frames from (example: python main_otoframe.py "C:\Users\Joban\Desktop\video.mov").
4. If the video path is valid, the software will run and create a folder containing "relevant" frames showing the tympanic membrane.


**"transfer_learning_mobilenetv2.py":**

1. Prepare your dataset in a main folder with two sub-folders, each containing images belonging to a different class.
2. Open the CNN model you want to train using any text editor.
3. Update line 16 with the path to your dataset folder (example: "/users/joban/desktop/dataset").
4. Save the file and run it with Python.
5. An H5 file will be generated containing the learned weights for classifying the keyframes.


## Contact

For any issues or inquiries please contact:
*email: 210142187@aston.ac.uk.