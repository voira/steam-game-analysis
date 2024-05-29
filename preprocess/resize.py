from PIL import Image
import os

# Define input and output directories
<<<<<<< HEAD
input_dir = 'C:/Users/nicol/OneDrive - KU Leuven/Desktop/python/Advanced Analytics in Business/2/images'
output_dir = 'C:/Users/nicol/OneDrive - KU Leuven/Desktop/python/Advanced Analytics in Business/2/images2'
=======
input_dir = 'C:/Users/Beste/Desktop/AAB/images-big'
output_dir = 'C:/Users/Beste/Desktop/AAB/images'
>>>>>>> 949e9077c93cb15850514220030b89cc7b6e57d0

# Define the target size for resizing
target_size = (256, 256)  # Specify the new dimensions

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
#'C:/Users/Beste/Desktop/AAB/images/images\\1029210_ss_ddf975e414f08a0bb419818d2cd94be4ebff6db9.1920x1080.jpg'
#'C:/Users/Beste/Desktop/AAB/images/images\\1029210_ss_7096afc94036153001b266a75253fb6abdd03c54.1920x1080.jpg'

# Iterate over the images in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Add other formats if needed
        # Open the image file
        image_path = os.path.join(input_dir, filename)
        with Image.open(image_path) as img:
            # Resize the image
            resized_img = img.resize(target_size)
            # Save the resized image to the output directory
            output_path = os.path.join(output_dir, filename)
            resized_img.save(output_path)