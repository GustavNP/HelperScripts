import pandas as pd
import random
import numpy as np
import shutil


pixel_count_file = "output_files/region_pixel_count-VGGFace200k-ALL.csv"

region = "LeftEyeBrow"

pixel_count_df = pd.read_csv(pixel_count_file, sep=';', header=None)
pixel_count_df.columns = ["Person", "Filename", "PixelCount"]

# images with region blacked out
pixel_count_df = pixel_count_df[pixel_count_df["Filename"].apply(lambda x: region in x)]
print(len(pixel_count_df))

# images where pixel count is 0
pixel_count_df = pixel_count_df[pixel_count_df["PixelCount"].apply(lambda x: x == 0)]
print(len(pixel_count_df))

# randomly choose 100 images to investigate
rows_to_choose = []
random.seed(36)
for i in range(0,100):
    random_number = random.randrange(len(pixel_count_df))
    while(random_number in rows_to_choose): # get number not already in list
        random_number = random.randrange(len(pixel_count_df))
    rows_to_choose.append(random_number)

print(pixel_count_df.head())

pixel_count_df = pixel_count_df.set_index(keys=np.arange(0, len(pixel_count_df)))
print(pixel_count_df)


pixel_count_df = pixel_count_df.iloc[rows_to_choose]

print(pixel_count_df)





# Find the originals of the chosen blacked out images and copy them to specific directory
pixel_count_df["OriginalImageFilename"] = pixel_count_df["Filename"].apply(lambda x: x.split('_Blackout')[0].split('aligned_')[1] + '.jpg')
pixel_count_df["AlignedImageFilename"] = pixel_count_df["Filename"].apply(lambda x: x.split('_Blackout')[0] + '.jpg')
print(pixel_count_df)


original_images_root = "C:\\Users\\admin\\Downloads\\archive1\\vggface-50-images"
output_directory = "C:\\Users\\admin\\Desktop\\aligned-vs-non-aligned-LeftEyeBrow-test"

pixel_count_df.apply(lambda row: shutil.copy(f"{original_images_root}\\{row['Person']}\\{row['OriginalImageFilename']}", f"{output_directory}\\AlignmentImpactTest_{row['OriginalImageFilename']}"), axis=1)


aligned_images_root = "C:\\Users\\admin\\Desktop\\aligned-VGGFace200k-folders-50-images"
aligned_output_directory = "C:\\Users\\admin\\Desktop\\aligned-vs-non-aligned-LeftEyeBrow-test-AlignedImages"

pixel_count_df.apply(lambda row: shutil.copy(f"{aligned_images_root}\\{row['Person']}\\{row['AlignedImageFilename']}", f"{aligned_output_directory}\\AlignmentYes_{row['AlignedImageFilename']}"), axis=1)
