import os
import random
from PIL import Image


def random_crop_and_save(
    input_folder, output_folder, prefix="aug_", crop_size=(400, 400)
):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all files in the input folder
    image_files = [
        f
        for f in os.listdir(input_folder)
        if os.path.isfile(os.path.join(input_folder, f))
    ]

    i = 0
    for image_file in image_files:
        # Open the image
        image_path = os.path.join(input_folder, image_file)
        img = Image.open(image_path)

        # Skip images smaller than 256x256
        width, height = img.size
        if width < 400 or height < 400:
            print(f"Skipping {image_file} - Image size is below 400x400")
            continue

        # Adjust crop size if the image is smaller
        crop_width, crop_height = crop_size[0], crop_size[1]

        # Get a random crop
        left = random.randint(0, width - crop_width)
        top = random.randint(0, height - crop_height)
        right = left + crop_width
        bottom = top + crop_height

        # Crop the image
        cropped_img = img.crop((left, top, right, bottom))

        # Save the cropped image with the specified naming format
        output_path = os.path.join(output_folder, prefix + image_file)
        cropped_img.save(output_path)
        i = i + 1
    print(f"{i} Images are created")


if __name__ == "__main__":
    # Specify your input and output folders
    input_folder_path = "HighRes"
    output_folder_path = "HighRes/Aug"

    # Call the function to perform random cropping and saving
    random_crop_and_save(input_folder_path, output_folder_path)
