import os
from PIL import Image


def is_grayscale(image_path):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    for i in range(w):
        for j in range(h):
            r, g, b = img.getpixel((i, j))
            if r != g != b:
                return False
    return True


def delete_grayscale_images(folder_path):
    # List all files in the folder
    image_files = [
        f
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]

    # Iterate through each image file
    for image_file in image_files:
        # Check if the image is grayscale
        image_path = os.path.join(folder_path, image_file)
        if is_grayscale(image_path):
            # Delete the grayscale image
            os.remove(image_path)
            print(f"Deleted grayscale image: {image_file}")


if __name__ == "__main__":
    # Specify the path to your folder containing images
    folder_path = "HighRes-Copy"

    # Call the function to delete grayscale images
    delete_grayscale_images(folder_path)
