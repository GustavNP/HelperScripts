import pandas as pd
import random
import numpy as np
import shutil


aligned_pixel_count_file = "output_files/region_pixel_count-VGGFace200k-ALL.csv"
nonaligned_pixel_count_file = "output_files/region_pixel_count_AlignedImpactTest.csv"

region = "LeftEyeBrow"

# aligned_pixel_count_df = pd.read_csv(aligned_pixel_count_file, sep=';', header=None)
# aligned_pixel_count_df.columns = ["Person", "Filename", "PixelCount"]

nonaligned_pixel_count_df = pd.read_csv(nonaligned_pixel_count_file, sep=';', header=None)
nonaligned_pixel_count_df.columns = ["Filename", "Region", "PixelCount"]

nonaligned_pixel_count_df = nonaligned_pixel_count_df[nonaligned_pixel_count_df['Region'].apply(lambda region_name: region_name == region)]
nonaligned_pixel_count_df["OriginalImageFilename"] = nonaligned_pixel_count_df["Filename"].apply(lambda x: x.split('AlignmentImpactTest_')[1].split('.')[0] + '.jpg')

print(nonaligned_pixel_count_df.describe())

non_zero_pixel_counts_df = nonaligned_pixel_count_df[nonaligned_pixel_count_df['PixelCount'].apply(lambda pixel_count: pixel_count > 0)]
print(non_zero_pixel_counts_df)



nonaligned_pixel_count_df.to_csv(f"AlignedImpactTest-nonaligned-images-pixel-counts-for-region-{region}.csv", sep=';', header=False,  index=False)


# Find original filenames of both aligned and nonaligned, so they can be matched on that
# aligned_pixel_count_df["OriginalImageFilename"] = aligned_pixel_count_df["Filename"].apply(lambda x: x.split('_Blackout')[0].split('aligned_')[1] + '.jpg')
# nonaligned_pixel_count_df["OriginalImageFilename"] = nonaligned_pixel_count_df["Filename"].apply(lambda x: x.split('AlignmentImpactTest_')[1].split('.')[0] + '.jpg')

# find the pixel counts for the region we're actually investigating right now
# aligned_pixel_count_df = aligned_pixel_count_df[aligned_pixel_count_df['Filename'].apply(lambda filename: region in filename)]
# nonaligned_pixel_count_df = nonaligned_pixel_count_df[nonaligned_pixel_count_df['Region'].apply(lambda region_name: region_name == region)]

# Find the pixel counts for the aligned versions
# original_filenames = nonaligned_pixel_count_df['OriginalImageFilename'].values
# aligned_pixel_count_df = aligned_pixel_count_df[aligned_pixel_count_df['OriginalImageFilename'].apply(lambda filename: filename in original_filenames)]


# print(aligned_pixel_count_df)
# print(nonaligned_pixel_count_df)

# combined_df = pd.merge(aligned_pixel_count_df, nonaligned_pixel_count_df, on="OriginalImageFilename")


# print(combined_df)
      
