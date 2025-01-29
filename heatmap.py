import cv2
import os
import csv


face_parsing_masks_directory_path = "C:\\Users\\admin\\source\\repos\\OFIQ-Project-FGFP\\install_x86_64\\Release\\bin\\face_parsing_images"


# region_class_dictionary = {
#     "LeftEyeBrow" : 2,
#     "RightyeBrow" : 3,
#     "LeftEye" : 4,
#     "RightEye" : 5,
#     "Nose" : 10,
#     "Mouth" : 11,
#     "UpperLip" : 12,
#     "LowerLip" : 13,
#     "Nasal" : 20,
#     "LeftOrbital" : 21,
#     "RightOrbital" : 22,
#     "Mental" : 23,
#     "LeftBuccal" : 24,
#     "RightBuccal" : 25,
#     "LeftZygoInfraParo" : 26,
#     "RightZygoInfraParo" : 27
# }



# face_parsing_filename = "C:\\Users\\admin\\source\\repos\\OFIQ-Project-FGFP\\install_x86_64\\Release\\bin\\face_parsing_images\\nearest_face_parsing_n000002-0009_01.png"
face_parsing_filename = "C:\\Users\\admin\\source\\repos\\OFIQ-Project-FGFP\\install_x86_64\\Release\\bin\\face_parsing_images\\AlignmentImpactTest\\nearest_face_parsing_AlignmentImpactTest_n000136-0028_01.png"

face_parsing_mask_gray = cv2.imread(face_parsing_filename, cv2.IMREAD_GRAYSCALE) # Assumed to have size 200x200
# aligned image is cropped before being input to face parsing algorithm. To adjust for this, 22 pixels must be added in each direction in the places specified (when face parsing size is 200x200)
face_parsing_mask_padding = cv2.copyMakeBorder(face_parsing_mask_gray, 0, 22, 11, 11, cv2.BORDER_CONSTANT, (0)) 


# img_copy = face_parsing_mask_padding.copy()
# for i in range(0,222):
#     for k in range(0,222):
#         pixel_class_number = face_parsing_mask_padding[i, k]

#         if pixel_class_number == 255:
#             img_copy[i, k] = pixel_class_number
#             continue
            
#         if pixel_class_number > 15 and pixel_class_number < 30:
#             pixel_class_number -= 15
#             pixel_class_number *= 15

#         img_copy[i, k] = pixel_class_number




# img_copy = face_parsing_mask_padding.copy()
# for i in range(0,222):
#     for k in range(0,222):
#         pixel_class_number = face_parsing_mask_padding[i, k]

#         if pixel_class_number == 255:
#             pixel_class_number = 1 # skin
            
#         pixel_class_number *= 8

#         img_copy[i, k] = pixel_class_number





img_copy = face_parsing_mask_padding.copy()
for i in range(0,222):
    for k in range(0,222):
        pixel_class_number = face_parsing_mask_padding[i, k]

        if pixel_class_number == 255:
            pixel_class_number = 1 # skin

        if pixel_class_number > 1 and pixel_class_number < 4: # eyebrows
            pixel_class_number *= 70

        img_copy[i, k] = pixel_class_number



colormap_face_parsing_img = cv2.applyColorMap(img_copy, cv2.COLORMAP_COOL)


# cv2.imwrite("colormap_face_parsing_all_regions.png", colormap_face_parsing_img)
# cv2.imwrite("colormap_face_parsing_new_subregions.png", colormap_face_parsing_img)
cv2.imwrite("colormap_face_parsing_only_eyebrows.png", colormap_face_parsing_img)
