import os
import shutil



# input_root = "C:\\Users\\admin\\Desktop\\blackout-aligned-VGGFace200k-50-images"
# output_root = "C:\\Users\\admin\\Desktop\\blackout-aligned-VGGFace200k-folders-50-images-flat-identity-folders"

input_root = "C:\\Users\\admin\\Desktop\\blackout-aligned-VGGFace200k-50-images-Existing-regions"
output_root = "C:\\Users\\admin\\Desktop\\blackout-aligned-VGGFace200k-50-images-Existing-regions-flat-identity-folders"



if not os.path.exists(output_root):
    os.makedirs(output_root)

counter = 0
for root, dirs, files in os.walk(input_root):
    for file in files:
        counter += 1
        if counter % 100 == 0:
            print(f"Files copied: {counter}. Now copying {file}")
            
        output_directory_name = root.split("\\")[-2]
        output_directory_path = os.path.join(output_root, output_directory_name)

        if not os.path.exists(output_directory_path):
            os.makedirs(output_directory_path)

        input_file_path = os.path.join(root, file)
        output_file_path = os.path.join(output_directory_path, file)

        shutil.copy(input_file_path, output_file_path)
        print(f"Copied {input_file_path} to {output_file_path}")

