import cv2
import os
import csv

# Author: Gustav Nilsson Pedersen - s174562@student.dtu.dk


def create_image_with_blacked_out_region(original_img, face_parsing_mask, region_class_dictionary, blackout_images_path_base, org_image_filename, person_directory_name):
    
    region_images_dictionary = {}
    pixel_counter_dictionary = {}
    region_classnumber_to_name_dictionary = {}
    for region_name, number in region_class_dictionary.items():
        region_images_dictionary[region_name] = original_img.copy()
        pixel_counter_dictionary[region_name] = 0
        region_classnumber_to_name_dictionary[number] = region_name

    for i in range(0,222):
        for k in range(0,222):
            pixel_class_number = face_parsing_mask[i, k]

            if pixel_class_number in region_classnumber_to_name_dictionary:
                region_name = region_classnumber_to_name_dictionary[pixel_class_number]
                region_images_dictionary[region_name][i, k] = 0
                pixel_counter_dictionary[region_name] += 1

    region_pixel_count_with_image_path_dictionary = {}
    file_name, file_extension = os.path.splitext(org_image_filename)
    for region_name, region_number in region_class_dictionary.items():
        blackout_img_name = f"{file_name}_Blackout_{region_name}{file_extension}"
        directory = f"{blackout_images_path_base}\\{person_directory_name}\\{region_name}"
        if not os.path.exists(directory):
            os.makedirs(directory)
        blackout_img_save_path = os.path.join(directory, blackout_img_name)
        cv2.imwrite(blackout_img_save_path, region_images_dictionary[region_name])
        region_pixel_count_with_image_path_dictionary[blackout_img_save_path] = pixel_counter_dictionary[region_name]

    return region_pixel_count_with_image_path_dictionary



# aligned_images_directory_path = "C:\\Users\\admin\\Desktop\\aligned-test"
# aligned_images_directory_path = "C:\\Users\\admin\\Desktop\\aligned-VGGFace200k-folders-50-images"
aligned_images_directory_path = "aligned-images"

# face_parsing_masks_directory_path = "C:\\Users\\admin\\Desktop\\face-parsing-test"
# face_parsing_masks_directory_path = "C:\\Users\\admin\\source\\repos\\OFIQ-Project-FGFP\\install_x86_64\\Release\\bin\\face_parsing_images"
face_parsing_masks_directory_path = "face-parsing-masks"

# blackout_images_path_base = "C:\\Users\\admin\\Desktop\\blackout-aligned-VGGFace200k-50-images"
# blackout_images_path_base = "C:\\Users\\admin\\Desktop\\blackout-aligned-VGGFace200k-50-images-Existing-regions"
# blackout_images_path_base = "C:\\Users\\admin\\Desktop\\test-blackout-existing-regions"
blackout_images_path_base = "blackout_images"


region_class_dictionary = {
    "LeftEyeBrow" : 2,
    "RightyeBrow" : 3,
    "LeftEye" : 4,
    "RightEye" : 5,
    "Nose" : 10,
    "Mouth" : 11,
    "UpperLip" : 12,
    "LowerLip" : 13,
    "Nasal" : 20,
    "RightOrbital" : 21,
    "LeftOrbital" : 22,
    "Mental" : 23,
    "RightBuccal" : 24,
    "LeftBuccal" : 25,
    "RightZygoInfraParo" : 26,
    "LeftZygoInfraParo" : 27
}



region_pixel_count_dictionary = {} # will store the number of pixels in the blacked out region in each blacked out image
for root, dirs, aligned_images_filenames in os.walk(aligned_images_directory_path):
    
    for dir in dirs: # assumes there are only directories in the root directory and that those directories are person-directories
        new_directory_path = os.path.join(blackout_images_path_base, dir)
        region_pixel_count_dictionary[dir] = {}
        if not os.path.exists(new_directory_path):
            os.makedirs(new_directory_path)


    person_directory_name = root.split("\\")[-1]
    print(f"Blacking out regions for person: {person_directory_name}")

    for image_filename in aligned_images_filenames:
        original_img_path = os.path.join(root, image_filename)
        original_img = cv2.imread(original_img_path) # should have size 222x222 when face parsing image has size 200x200

        face_parsing_filename = image_filename.replace("aligned", "nearest_face_parsing")
        face_parsing_filename = face_parsing_filename.replace(".jpg", ".png")
        face_parsing_mask_path = os.path.join(face_parsing_masks_directory_path, face_parsing_filename)
        face_parsing_mask_gray = cv2.imread(face_parsing_mask_path, cv2.IMREAD_GRAYSCALE) # Assumed to have size 200x200
        # aligned image is cropped before being input to face parsing algorithm. To adjust for this, 22 pixels must be added in each direction in the places specified (when face parsing size is 200x200)
        face_parsing_mask_padding = cv2.copyMakeBorder(face_parsing_mask_gray, 0, 22, 11, 11, cv2.BORDER_CONSTANT, (0)) 

        # original_images_directory = f"{blackout_images_path_base}\\{person_directory_name}\\originals"
        # if not os.path.exists(original_images_directory):
        #     os.makedirs(original_images_directory)
        # original_image_small_save_path = f"{original_images_directory}\\{image_filename}"
        # cv2.imwrite(original_image_small_save_path, original_img)

        region_pixel_count_with_image_path_dictionary = create_image_with_blacked_out_region(original_img, 
                                                                    face_parsing_mask_padding, 
                                                                    region_class_dictionary.copy(), 
                                                                    blackout_images_path_base, 
                                                                    image_filename, 
                                                                    person_directory_name)
        for blackout_image_path, blackout_region_image_pixel_count in region_pixel_count_with_image_path_dictionary.items():
            region_pixel_count_dictionary[person_directory_name][blackout_image_path] = blackout_region_image_pixel_count

region_pixel_count_output_file = "region_pixel_count.csv"
with open(region_pixel_count_output_file, 'w', newline='') as output_csv:
    writer = csv.writer(output_csv, delimiter=';')
    for person, blackout_images_dictionary in region_pixel_count_dictionary.items():
        for blackout_image_path, pixel_count in blackout_images_dictionary.items():         
            row = [person, blackout_image_path, pixel_count]
            writer.writerow(row)
            


