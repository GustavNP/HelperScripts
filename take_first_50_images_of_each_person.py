import os
import shutil

input_root = "C:\\Users\\admin\\Desktop\\aligned-VGGFace200k-folders"
output_root = "C:\\Users\\admin\\Desktop\\aligned-VGGFace200k-folders-50-images"

if not os.path.exists(output_root):
    os.makedirs(output_root)

for root, dirs, files in os.walk(input_root):
    counter = 1

    for dir in dirs: # assumes there are only directories in the root directory and that those directories are person-directories
        new_directory_path = os.path.join(output_root, dir)
        if not os.path.exists(new_directory_path):
            os.makedirs(new_directory_path)

    for file in files:
        if counter > 50: # only take first 50 files in each directory
            break

        old_file_path = os.path.join(root, file)
        
        person_directory = root.split("\\")[-1]
        output_file_path = os.path.join(output_root, person_directory, file)

        shutil.copy(old_file_path, output_file_path)
        print(f"Copied {old_file_path} to {output_file_path}")
        counter += 1
