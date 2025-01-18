import cv2
import os
import csv

def count_pixels_in_region(face_parsing_mask, region_class_number):
    pixel_counter = 0
    for i in range(0,222):
        for k in range(0,222):
            pixel_class_number = face_parsing_mask[i, k]
            if pixel_class_number == region_class_number:
                pixel_counter += 1
    return pixel_counter



aligned_images_directory_path = "C:\\Users\\admin\\Desktop\\aligned-VGGFace200k-folders-50-images"
face_parsing_masks_directory_path = "C:\\Users\\admin\\source\\repos\\OFIQ-Project-FGFP\\install_x86_64\\Release\\bin\\face_parsing_images"
blackout_images_path_base = "C:\\Users\\admin\\Desktop\\blackout-aligned-VGGFace200k-50-images"

region_class_dictionary = {
    "Nasal" : 20,
    "LeftOrbital" : 21,
    "RightOrbital" : 22,
    "Mental" : 23,
    "LeftBuccal" : 24,
    "RightBuccal" : 25,
    "LeftZygoInfraParo" : 26,
    "RightZygoInfraParo" : 27
}

region_pixel_count_dictionary = {} # will store the number of pixels in the blacked out region in each blacked out image
for root, dirs, aligned_images_filenames in os.walk(aligned_images_directory_path):
        
        
    person_directory_name = root.split("\\")[-1]
    print(f"Computing region pixel counts for person: {person_directory_name}")
    region_pixel_count_dictionary[person_directory_name] = {}

    for image_filename in aligned_images_filenames:
        face_parsing_filename = image_filename.replace("aligned", "face_parsing")
        face_parsing_filename = face_parsing_filename.replace(".jpg", ".png")
        face_parsing_mask_path = os.path.join(face_parsing_masks_directory_path, face_parsing_filename)
        face_parsing_mask_gray = cv2.imread(face_parsing_mask_path, cv2.IMREAD_GRAYSCALE) # Assumed to have size 200x200
        # aligned image is cropped before being input to face parsing algorithm. To adjust for this, 22 pixels must be added in each direction in the places specified (when face parsing size is 200x200)
        face_parsing_mask_padding = cv2.copyMakeBorder(face_parsing_mask_gray, 0, 22, 11, 11, cv2.BORDER_CONSTANT, (0)) 

        file_name, file_extension = os.path.splitext(image_filename)
        for region_name, region_number in region_class_dictionary.items():
            blackout_img_name = f"{file_name}_Blackout_{region_name}{file_extension}"
            directory = f"{blackout_images_path_base}\\{person_directory_name}\\{region_name}"
            blackout_img_save_path = os.path.join(directory, blackout_img_name)
        
            region_pixel_count = count_pixels_in_region(face_parsing_mask_padding, region_number)
            region_pixel_count_dictionary[person_directory_name][blackout_img_save_path] = region_pixel_count
            print(f"{blackout_img_save_path} - {region_pixel_count}")

region_pixel_count_output_file = "./output_files/region_pixel_count.csv"
with open(region_pixel_count_output_file, 'w', newline='') as output_csv:
    writer = csv.writer(output_csv, delimiter=';')
    for person, blackout_images_dictionary in region_pixel_count_dictionary.items():
        for blackout_image_path, pixel_count in blackout_images_dictionary.items():         
            row = [person, blackout_image_path, pixel_count]
            writer.writerow(row)
            


