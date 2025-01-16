import cv2
import os

def create_image_with_blacked_out_region(original_img, face_parsing_mask, region_class_number, blackout_img_save_path):
    blackout_img = original_img.copy()
    for i in range(0,222):
        for k in range(0,222):
            pixel_class_number = face_parsing_mask[i, k]
            if pixel_class_number == region_class_number:
                blackout_img[i, k] = 0
    
    cv2.imwrite(blackout_img_save_path, blackout_img)



# aligned_images_directory_path = "C:\\Users\\admin\\Desktop\\aligned-test"
# aligned_images_directory_path = "C:\\Users\\admin\\Desktop\\aligned-VGGFace200k-folders-50-images"
# aligned_images_directory_path = "C:\\Users\\admin\\Desktop\\aligned-VGGFace200k-folders-50-images-first-quarter"
# aligned_images_directory_path = "C:\\Users\\admin\\Desktop\\aligned-VGGFace200k-folders-50-images-second-quarter"
# aligned_images_directory_path = "C:\\Users\\admin\\Desktop\\aligned-VGGFace200k-folders-50-images-third-quarter"
aligned_images_directory_path = "C:\\Users\\admin\\Desktop\\aligned-VGGFace200k-folders-50-images-fourth-quarter"
# face_parsing_masks_directory_path = "C:\\Users\\admin\\Desktop\\face-parsing-test"
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


for root, dirs, aligned_images_filenames in os.walk(aligned_images_directory_path):
    
    for dir in dirs: # assumes there are only directories in the root directory and that those directories are person-directories
        new_directory_path = os.path.join(blackout_images_path_base, dir)
        if not os.path.exists(new_directory_path):
            os.makedirs(new_directory_path)


    person_directory_name = root.split("\\")[-1]
    print(f"Blacking out regions for person: {person_directory_name}")

    for image_filename in aligned_images_filenames:
        original_img_path = os.path.join(root, image_filename)
        original_img = cv2.imread(original_img_path) # should have size 222x222 when face parsing image has size 200x200

        face_parsing_filename = image_filename.replace("aligned", "face_parsing")
        face_parsing_filename = face_parsing_filename.replace(".jpg", ".png")
        face_parsing_mask_path = os.path.join(face_parsing_masks_directory_path, face_parsing_filename)
        face_parsing_mask_gray = cv2.imread(face_parsing_mask_path, cv2.IMREAD_GRAYSCALE) # Assumed to have size 200x200
        # aligned image is cropped before being input to face parsing algorithm. To adjust for this, 22 pixels must be added in each direction in the places specified (when face parsing size is 200x200)
        face_parsing_mask_padding = cv2.copyMakeBorder(face_parsing_mask_gray, 0, 22, 11, 11, cv2.BORDER_CONSTANT, (0)) 

        original_images_directory = f"{blackout_images_path_base}\\{person_directory_name}\\originals"
        if not os.path.exists(original_images_directory):
            os.makedirs(original_images_directory)
        original_image_small_save_path = f"{original_images_directory}\\{image_filename}"
        cv2.imwrite(original_image_small_save_path, original_img)

        
        file_name, file_extension = os.path.splitext(image_filename)
        for region_name, region_number in region_class_dictionary.items():
            blackout_img_name = f"{file_name}_Blackout_{region_name}{file_extension}"
            directory = f"{blackout_images_path_base}\\{person_directory_name}\\{region_name}"
            if not os.path.exists(directory):
                os.makedirs(directory)
            blackout_img_save_path = os.path.join(directory, blackout_img_name)
        
            create_image_with_blacked_out_region(original_img, face_parsing_mask_padding, region_number, blackout_img_save_path)

