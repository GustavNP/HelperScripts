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



face_parsing_masks_directory_path = "C:\\Users\\admin\\source\\repos\\OFIQ-Project-FGFP\\install_x86_64\\Release\\bin\\face_parsing_images\\AlignmentImpactTest"


# region_class_dictionary = {
#     "Nasal" : 20,
#     "LeftOrbital" : 21,
#     "RightOrbital" : 22,
#     "Mental" : 23,
#     "LeftBuccal" : 24,
#     "RightBuccal" : 25,
#     "LeftZygoInfraParo" : 26,
#     "RightZygoInfraParo" : 27
# }



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
    "LeftOrbital" : 21,
    "RightOrbital" : 22,
    "Mental" : 23,
    "LeftBuccal" : 24,
    "RightBuccal" : 25,
    "LeftZygoInfraParo" : 26,
    "RightZygoInfraParo" : 27
}


region_pixel_count_dictionary = {} # will store the number of pixels in the blacked out region in each blacked out image
for root, dirs, face_parsing_images_filenames in os.walk(face_parsing_masks_directory_path):
        
    for image_filename in face_parsing_images_filenames:
        face_parsing_mask_path = os.path.join(root, image_filename)
        face_parsing_mask_gray = cv2.imread(face_parsing_mask_path, cv2.IMREAD_GRAYSCALE) # Assumed to have size 200x200
        # aligned image is cropped before being input to face parsing algorithm. To adjust for this, 22 pixels must be added in each direction in the places specified (when face parsing size is 200x200)
        face_parsing_mask_padding = cv2.copyMakeBorder(face_parsing_mask_gray, 0, 22, 11, 11, cv2.BORDER_CONSTANT, (0)) 

        region_pixel_count_dictionary[image_filename] = {}
        for region_name, region_number in region_class_dictionary.items():
        
            region_pixel_count = count_pixels_in_region(face_parsing_mask_padding, region_number)
            region_pixel_count_dictionary[image_filename][region_name] = region_pixel_count
            print(f"{image_filename} - {region_pixel_count}")

region_pixel_count_output_file = "./output_files/region_pixel_count_AlignedImpactTest.csv"
with open(region_pixel_count_output_file, 'w', newline='') as output_csv:
    writer = csv.writer(output_csv, delimiter=';')
    for image_filename, region_pixel_counts_for_image_dict in region_pixel_count_dictionary.items():
        for region_name, pixel_count in region_pixel_counts_for_image_dict.items():         
            row = [image_filename, region_name, pixel_count]
            writer.writerow(row)
            


