import os
import cv2

# Input and output directories
input_folder = "Dataset/valid/HighRes"
output_folder = "Dataset/valid/LowRes"

# Define the downsampling factor (e.g., 2 for reducing by half)
downsampling_factor = 4

# Create the output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List all image files in the input folder
image_files = [
    f for f in os.listdir(input_folder) if f.endswith((".jpg", ".png", ".jpeg", ".bmp"))
]

# Downsample and save images
for image_file in image_files:
    # Read the image
    image = cv2.imread(os.path.join(input_folder, image_file))

    # Get the new dimensions after downsampling
    new_width = image.shape[1] // downsampling_factor
    new_height = image.shape[0] // downsampling_factor

    # Resize the image
    downsized_image = cv2.resize(image, (new_width, new_height))

    # Save the downsized image to the output folder
    output_file = os.path.join(output_folder, image_file)
    cv2.imwrite(output_file, downsized_image)

print("Downsampling completed.")
