# Theft-Detection-Project-using-OpenCV
This project implements a bike theft detection system using deep learning techniques. It utilizes computer vision and deep learning models to identify potential bike theft events in a given dataset.

Installation To run the bike theft detection system, please follow these steps:

Install the required dependencies: shell Copy code !pip install opencv-python-headless !pip install torch torchvision !pip install mtcnn Clone the YOLOv5 repository: shell Copy code !git clone https://github.com/ultralytics/yolov5 Install the YOLOv5 requirements: shell Copy code !pip install -r yolov5/requirements.txt Usage Mount Google Drive to access the dataset: python Copy code from google.colab import drive drive.mount('/gdrive') Preprocess the dataset: python Copy code from pathlib import Path

train_folder = (r'/gdrive/My Drive/bike_dataset/train/') test_folder = (r'/gdrive/My Drive/bike_dataset/test/') valid_folder = (r'/gdrive/My Drive/bike_dataset/valid/')

preprocessed_train_folder = (r'/gdrive/My Drive/bike_dataset/preprocessed_train/') preprocessed_test_folder = (r'/gdrive/My Drive/bike_dataset/preprocessed_test/') preprocessed_valid_folder = (r'/gdrive/My Drive/bike_dataset/preprocessed_valid/')

Preprocess the training set
preprocess_images(train_folder, preprocessed_train_folder)

Preprocess the test set
preprocess_images(test_folder, preprocessed_test_folder)

Preprocess the validation set
preprocess_images(valid_folder, preprocessed_valid_folder) Run the bike theft detection system: python Copy code preprocessed_image_paths = glob.glob(preprocessed_test_folder + '*.jpg')

for img_path in preprocessed_image_paths: img = cv2.imread(img_path)

faces = get_faces(img)
all_detections = get_bike_and_human_detections_yolov5(img)

draw_bounding_boxes(img, all_detections)

if is_theft_event(faces, all_detections):
    output_folder = (r'/gdrive/My Drive/bike_dataset/theft/')
else:
    output_folder = (r'/gdrive/My Drive/bike_dataset/non-theft/')

os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, os.path.basename(img_path))
save_screenshot(img, output_path)
Summary This bike theft detection system uses deep learning models, including YOLOv5, to identify potential bike theft events in a given dataset. It preprocesses the dataset, detects faces and bikes in the images, and applies a similarity score to determine if a bike theft event has occurred. The system saves the output images in separate folders based on the classification.

Please note that this is a simplified implementation, and further improvements can be made to enhance the system's accuracy and performance.

Feel free to modify and expand the sections to provide more details about your project.
