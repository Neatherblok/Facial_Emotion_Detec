import cv2
import torchvision.models as models
import torch
import numpy as np
from torch import nn
from torchvision import transforms
from PIL import Image
import time

# Initialize the camera
stream = cv2.VideoCapture(0)

# Specify the path to the saved model
PATH = 'Models/InceptionV3/trained_inception_v3.pt'

# Load the model
model = models.inception_v3(pretrained=True)

# Freeze the parameters
for parameter in model.parameters():
    parameter.requires_grad = False

# Replace the last fully connected layer with a new one that outputs 7 classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 7)  # Output layer with 7 classes
model.aux_logits = False
model.AuxLogits = None

# Load the saved weights
model_state_dict = torch.load(PATH)

# Load the state_dict into the model
model.load_state_dict(model_state_dict)

# Define the transforms
transforms = transforms.Compose([
            transforms.Resize(299),  # Resize to size 299x299,
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# Define the function to make the prediction

class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
def score_frame(frame, model, transforms):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    frame = transforms(Image.fromarray(frame))
    frame = torch.unsqueeze(frame, 0)
    labels = model(frame)
    labels = torch.argmax(labels)
    labels = labels.detach().cpu().numpy()
    labels = class_names[labels]  # Get the class name
    labels = "Prediction: " + str(labels)
    return labels

# Define the main function

def main():
    assert stream.isOpened()  # Make sure that their is a stream.
    # Below code creates a new video writer object to write our
    # output stream.
    x_shape = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    y_shape = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    four_cc = cv2.VideoWriter_fourcc(*"MJPG")  # Using MJPEG codex
    out = cv2.VideoWriter('output.mp4', four_cc, 20, \
                          (x_shape, y_shape))
    while True:
        ret, frame = stream.read()  # Read the first frame.
        if not ret:
            break
        start_time = time.time()  # We would like to measure the FPS.
        # Always display the last predicted result
        results = score_frame(frame, model, transforms)  # Score the Frame
        end_time = time.time()
        fps = 1 / np.round(end_time - start_time, 3)  # Measure the FPS.
        print(f"Frames Per Second : {fps}")
        # Define the text settings
        text_color = (0, 0, 0)  # Black color
        text_thickness = 2
        text_box_color = (255, 255, 255)  # White color
        text_box_thickness = 2

        # Add the text to the frame
        text_size, _ = cv2.getTextSize(results, cv2.FONT_HERSHEY_SIMPLEX, 1, text_thickness)
        x_text_pos = 10
        y_text_pos = y_shape - 10 - text_size[1]
        cv2.putText(frame, results, (x_text_pos, y_text_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, text_thickness)
        cv2.imshow("frame", frame)
        cv2.waitKey(1)
        out.write(frame)  # Write the frame onto the output.
    out.release()
    cv2.destroyAllWindows()

# Run the main function
main()