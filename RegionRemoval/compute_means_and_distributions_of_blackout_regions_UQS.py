import os
import pandas as pd
import csv
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

# Author: Gustav Nilsson Pedersen - s174562@student.dtu.dk
  
def convert_blackout_filename_to_original_filename(filename):
    split_filename = filename.split('_Blackout')
    original_image_filename = split_filename[0] + ".jpg"
    return original_image_filename

pd.options.display.max_colwidth = 200

score_file = "./VGGFace200k-50-image-blackout-regions-UQS-ALL.csv"
score_df = pd.read_csv(score_file, sep=';')

originals_df = score_df[score_df["Filename"].apply(lambda filename: "_Blackout_" not in filename)]
print(len(originals_df))

average_UQS_originals = originals_df["UnifiedQualityScore.scalar"].mean()
print(f"Average score for original aligned images: {average_UQS_originals}")

region_names = [
    "LeftEyeBrow",
    "RightyeBrow",
    "LeftEye",
    "RightEye",
    "Nose",
    "Mouth",
    "UpperLip",
    "LowerLip",
    "Nasal",
    "LeftOrbital",
    "RightOrbital",
    "Mental",
    "LeftBuccal",
    "RightBuccal",
    "LeftZygoInfraParo",
    "RightZygoInfraParo",
]


region_pixel_count_file = "region_pixel_count-VGGFace200k-ALL.csv" # should have row structure: <blackout-image-filename>;<region-pixel-count>
region_pixel_count_df = pd.read_csv(region_pixel_count_file, sep=';', header=None)
region_pixel_count_df.columns = ['person', 'Filename', 'pixel_count']
# region_pixel_count_df["Filename"] = region_pixel_count_df["Filename"].apply(lambda x: x.split("\\")[-1]) # only keep actual filename, not full path


region_averages_dictionary = { "Originals" : average_UQS_originals }
region_absolute_differences_averages_dictionary = { "Originals" : 0.0 }
region_average_absolute_difference_per_pixel_per_image_dictionary = { "Originals" : 0.0 }
region_average_difference_per_pixel_per_image_dictionary = { "Originals" : 0.0 }
region_number_of_images_with_0_pixels_dictionary = { "Originals" : 0 }
uqs_dataframe_dictionary = { "Originals" : originals_df["UnifiedQualityScore.scalar"] }
difference_blackout_region_scores_filenames = []
difference_per_pixel_blackout_region_scores_filenames = []
difference_dataframe_dictionary = {}
absolute_difference_dataframe_dictionary = {}
abs_dev_per_img_per_pixel_dataframe_dictionary = {}
diff_per_img_per_pixel_dataframe_dictionary = {}
for region in region_names:
    blackout_region_UQS_scores_df = score_df[score_df["Filename"].apply(lambda filename: f"{region}." in filename)] # ends with region
    # print(len(blackout_region_UQS_scores_df))
    # print(blackout_region_UQS_scores_df)
    

    # ========= Only consider regions where the region was actually identified, so there are >0 pixels ==========
    blackout_region_UQS_scores_and_pixel_count_df = pd.merge(blackout_region_UQS_scores_df, region_pixel_count_df, on='Filename', how='left')
    missing_pixel_counts = blackout_region_UQS_scores_and_pixel_count_df["pixel_count"].isnull().sum()
    # print(f"Missing pixel counts for {region}: {missing_pixel_counts}")
    no_of_images_with_0_pixels = len(blackout_region_UQS_scores_and_pixel_count_df[blackout_region_UQS_scores_and_pixel_count_df["pixel_count"] == 0].axes[0])
    # print(f"Number of images where pixel count is 0 for {region}: {no_of_images_with_0_pixels}")
    region_number_of_images_with_0_pixels_dictionary[region] = no_of_images_with_0_pixels



    # ========== Remove rows that have 0 pixels in regions ==========
    blackout_region_UQS_scores_and_pixel_count_with_more_than_0_pixels_df = blackout_region_UQS_scores_and_pixel_count_df[~(blackout_region_UQS_scores_and_pixel_count_df["pixel_count"] == 0)]
    print(len(blackout_region_UQS_scores_and_pixel_count_with_more_than_0_pixels_df))
    blackout_region_UQS_scores_df = blackout_region_UQS_scores_and_pixel_count_with_more_than_0_pixels_df[blackout_region_UQS_scores_df.columns]
    # print(blackout_region_UQS_scores_df.head())
    # print(len(blackout_region_UQS_scores_df))
    original_filenames_of_blackout_images_dict_for_lookup = dict.fromkeys(blackout_region_UQS_scores_df["Filename"].apply(convert_blackout_filename_to_original_filename).values, 0)
    # print(f"original_filenames_of_blackout_images_dict_for_lookup lenght for region {region}: {len(original_filenames_of_blackout_images_dict_for_lookup)}")
    originals_of_non_zero_images_df = originals_df[originals_df["Filename"].apply(lambda filename: filename in original_filenames_of_blackout_images_dict_for_lookup)]
    # print(len(originals_of_non_zero_images_df))



    # ========== Compute KDE plot for UQS of the non-zero blackout images ==========
    # blackout_region_UQS_scores_df.hist(column='UnifiedQualityScore.scalar', bins=50, rwidth=0.8)
    # plt.yscale("log")
    uqs_dataframe_dictionary[region] = blackout_region_UQS_scores_df["UnifiedQualityScore.scalar"]
    # blackout_region_UQS_scores_df["UnifiedQualityScore.scalar"].plot.kde()
    ax = sns.kdeplot(blackout_region_UQS_scores_df["UnifiedQualityScore.scalar"], fill=True, label=region)
    ax.legend()
    plt.savefig(f"./distribution_plots/average_UQS/average_UQS_non_zero_images_{region}.png")
    plt.clf()

    #blackout_region_UQS_scores_weird_values_df = blackout_region_UQS_scores_df[blackout_region_UQS_scores_df["UnifiedQualityScore.scalar"].apply(lambda x: x < 0)]
    #print(len(blackout_region_UQS_scores_weird_values_df))
    average_UQS = blackout_region_UQS_scores_df["UnifiedQualityScore.scalar"].mean()
    print(f"Average score for images with {region} region blacked out: {average_UQS}")
    region_averages_dictionary[region] = average_UQS


    # ========== Compute difference from original ==========
    difference_from_original_series = originals_of_non_zero_images_df["UnifiedQualityScore.scalar"] - blackout_region_UQS_scores_df["UnifiedQualityScore.scalar"].values
    # difference_from_original_df.hist(bins=50, rwidth=0.8)
    # plt.yscale("log")
    difference_dataframe_dictionary[region] = difference_from_original_series
    # difference_from_original_series.plot.kde()
    ax = sns.kdeplot(difference_from_original_series, fill=True, label=region)
    ax.legend()
    plt.savefig(f"./distribution_plots/difference_from_original/difference_from_original_{region}.png")
    plt.clf()
    #print(difference_from_original_df.head())
    #print(blackout_region_UQS_scores_df.dtypes)

    
    # ========== Save blacked out image differences from original images ==========
    difference_from_original_with_filename_df = pd.concat([blackout_region_UQS_scores_df["Filename"].reset_index(drop=True), difference_from_original_series.reset_index(drop=True)], axis=1, ignore_index=True)
    #print(differences_from_original_with_filename_df.head())
    dbrs_region_filename = f"./difference_blackout_region_scores/VGGFace200k-50-images-region_UQS_difference_from_original-{region}.csv"
    difference_blackout_region_scores_filenames.append(dbrs_region_filename)
    difference_from_original_with_filename_df.to_csv(dbrs_region_filename, sep=';', index=False)

    

    # ========== Compute absolute differences ==========
    difference_from_original_absolute_series = difference_from_original_series.abs()
    # difference_from_original_absolute_df.hist(bins=50, rwidth=0.8)
    # plt.yscale("log")
    absolute_difference_dataframe_dictionary[region] = difference_from_original_absolute_series
    # difference_from_original_absolute_series.plot.kde()
    ax = sns.kdeplot(difference_from_original_absolute_series, fill=True, label=region)
    ax.legend()
    plt.savefig(f"./distribution_plots/absolute_differences_from_original/absolute_differences_from_original_{region}.png")
    plt.clf()


    # ========== Compute average absolute difference from original ==========
    average_absolute_difference_from_original_UQS = difference_from_original_absolute_series.mean()
    region_absolute_differences_averages_dictionary[region] = average_absolute_difference_from_original_UQS
    print(f"Average (absolute) difference from original for images with {region} blacked out: {average_absolute_difference_from_original_UQS}")



    # ========== Compute average absolute difference per pixel per image ==========
    absolute_differences_df = pd.concat([blackout_region_UQS_scores_df["Filename"].reset_index(drop=True), difference_from_original_absolute_series.reset_index(drop=True)], axis=1, ignore_index=True)
    absolute_differences_df.columns = ["Filename", "UQSDifferenceFromOriginal"]
    absolute_differences_df["Filename"] = absolute_differences_df["Filename"].apply(lambda x: x.split("/")[-1]) # only keep actual filename, not full path
    # print(absolute_differences_and_pixel_count_df)
    absolute_differences_and_pixel_count_df = pd.merge(absolute_differences_df, region_pixel_count_df, on='Filename', how='left')
    absolute_difference_per_pixel_series = absolute_differences_and_pixel_count_df["UQSDifferenceFromOriginal"].div(absolute_differences_and_pixel_count_df["pixel_count"], axis=0)
    
    # difference_per_pixel_df.hist(bins=50, rwidth=0.8)
    # plt.yscale("log")
    abs_dev_per_img_per_pixel_dataframe_dictionary[region] = absolute_difference_per_pixel_series
    # difference_per_pixel_series.plot.kde()
    ax = sns.kdeplot(absolute_difference_per_pixel_series, fill=True, label=region)
    ax.legend()
    plt.xlim((0, 1))
    plt.savefig(f"./distribution_plots/absolute_difference_per_pixel_per_image/absolute_difference_per_pixel_per_image_{region}.png")
    plt.clf()

    # print(difference_per_pixel_df.head())
    average_absolute_difference_per_pixel_per_image = absolute_difference_per_pixel_series.mean()
    region_average_absolute_difference_per_pixel_per_image_dictionary[region] = average_absolute_difference_per_pixel_per_image
    print(f"Average absolute difference per pixel per image for images with {region} blacked out: {average_absolute_difference_per_pixel_per_image}")



    # ========== Compute average difference per pixel per image ==========
    differences_df = pd.concat([blackout_region_UQS_scores_df["Filename"].reset_index(drop=True), difference_from_original_series.reset_index(drop=True)], axis=1, ignore_index=True)
    differences_df.columns = ["Filename", "UQSDifferenceFromOriginal"]
    differences_df["Filename"] = differences_df["Filename"].apply(lambda x: x.split("/")[-1]) # only keep actual filename, not full path
    # print(absolute_differences_and_pixel_count_df)
    differences_and_pixel_count_df = pd.merge(differences_df, region_pixel_count_df, on='Filename', how='left')
    differences_per_pixel_series = differences_and_pixel_count_df["UQSDifferenceFromOriginal"].div(differences_and_pixel_count_df["pixel_count"], axis=0)
    differences_and_pixel_count_df["DifferencePerPixel"] = differences_per_pixel_series


    # difference_per_pixel_df.hist(bins=50, rwidth=0.8)
    # plt.yscale("log")
    diff_per_img_per_pixel_dataframe_dictionary[region] = differences_and_pixel_count_df["DifferencePerPixel"].values
    # difference_per_pixel_series.plot.kde()
    ax = sns.kdeplot(differences_per_pixel_series, fill=True, label=region)
    ax.legend()
    plt.xlim((0, 1))
    plt.savefig(f"./distribution_plots/difference_per_pixel_per_image/difference_per_pixel_per_image_{region}.png")
    plt.clf()

    # print(difference_per_pixel_df.head())
    average_difference_per_pixel_per_image = differences_per_pixel_series.mean()
    region_average_difference_per_pixel_per_image_dictionary[region] = average_difference_per_pixel_per_image
    print(f"Average difference per pixel per image for images with {region} blacked out: {average_difference_per_pixel_per_image}")

    difference_per_pixel_with_filename_df = pd.concat([blackout_region_UQS_scores_df["Filename"].reset_index(drop=True), differences_per_pixel_series.reset_index(drop=True)], axis=1, ignore_index=True)
    #print(differences_from_original_with_filename_df.head())
    dpp_region_filename = f"./difference_per_pixel_blackout_region_scores/VGGFace200k-50-images-region_UQS_difference_per_pixel_from_original-{region}.csv"
    difference_per_pixel_blackout_region_scores_filenames.append(dpp_region_filename)
    difference_per_pixel_with_filename_df.to_csv(dpp_region_filename, sep=';', index=False)









# ======== Plot kde plots in same figure ========


fgfp_regions = [
    "Nasal",
    "LeftOrbital",
    "RightOrbital",
    "Mental",
    "LeftBuccal",
    "RightBuccal",
    "LeftZygoInfraParo",
    "RightZygoInfraParo",
]

for region, data in uqs_dataframe_dictionary.items(): # Only old regions
    if region not in fgfp_regions:
        ax = sns.kdeplot(data, label=region)
plt.xlabel("UQS")
ax.legend()
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.title("Unified Quality Score - Existing Regions")
plt.subplots_adjust(right=0.67) # get labels to not be cut off
plt.savefig(f"./distribution_plots/average_UQS/average_UQS_non_zero_images_EXISTING.png")
plt.show()

ax = sns.kdeplot(uqs_dataframe_dictionary["Originals"], label="Originals")
for region, data in uqs_dataframe_dictionary.items(): # Only new FGFP regions
    if region in fgfp_regions:
        ax = sns.kdeplot(data, label=region)
plt.xlabel("UQS")
ax.legend()
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.title("Unified Quality Score - FGFP Regions")
plt.subplots_adjust(right=0.67) # get labels to not be cut off
plt.savefig(f"./distribution_plots/average_UQS/average_UQS_non_zero_images_FGFP.png")
plt.show()



for region, data in difference_dataframe_dictionary.items():
    if region not in fgfp_regions:
        ax = sns.kdeplot(data, label=region)
plt.xlabel("Difference in UQS")
ax.legend()
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.title("Difference from Original UQS - Existing Regions")
plt.subplots_adjust(right=0.67) # get labels to not be cut off
plt.savefig(f"./distribution_plots/difference_from_original/difference_from_original_EXISTING.png")
plt.show()

for region, data in difference_dataframe_dictionary.items():
    if region in fgfp_regions:
        ax = sns.kdeplot(data, label=region)
plt.xlabel("Difference in UQS")
ax.legend()
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.title("Difference from Original UQS - FGFP Regions")
plt.subplots_adjust(right=0.67) # get labels to not be cut off
plt.savefig(f"./distribution_plots/difference_from_original/difference_from_original_FGFP.png")
plt.show()



for region, data in absolute_difference_dataframe_dictionary.items():
    if region not in fgfp_regions:
        ax = sns.kdeplot(data, label=region)
plt.xlabel("Absolute Difference in UQS")
ax.legend()
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.title("Absolute Difference from Original UQS - Existing Regions")
plt.subplots_adjust(right=0.67) # get labels to not be cut off
plt.savefig(f"./distribution_plots/absolute_differences_from_original/absolute_differences_from_original_EXISTING.png")
plt.show()

for region, data in absolute_difference_dataframe_dictionary.items():
    if region in fgfp_regions:
        ax = sns.kdeplot(data, label=region)
plt.xlabel("Absolute Difference in UQS")
ax.legend()
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.title("Absolute Difference from Original UQS - FGFP Regions")
plt.subplots_adjust(right=0.67) # get labels to not be cut off
plt.savefig(f"./distribution_plots/absolute_differences_from_original/absolute_differences_from_original_FGFP.png")
plt.show()



for region, data in abs_dev_per_img_per_pixel_dataframe_dictionary.items():
    if region not in fgfp_regions:
        ax = sns.kdeplot(data, label=region)
plt.xlim((0.0, 0.5))
plt.xlabel("Absolute Difference in UQS per Pixel")
ax.legend()
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.title("Absolute Difference in UQS per Pixel - Existing Regions")
plt.subplots_adjust(right=0.67) # get labels to not be cut off
plt.savefig(f"./distribution_plots/absolute_difference_per_pixel_per_image/absolute_difference_per_pixel_per_image_EXISTING.png")
plt.show()

for region, data in abs_dev_per_img_per_pixel_dataframe_dictionary.items():
    if region in fgfp_regions:
        ax = sns.kdeplot(data, label=region)
plt.xlim((0.0, 0.5))
plt.xlabel("Absolute Difference in UQS per Pixel")
ax.legend()
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.title("Absolute Difference in UQS per Pixel - FGFP Regions")
plt.subplots_adjust(right=0.67) # get labels to not be cut off
plt.savefig(f"./distribution_plots/absolute_difference_per_pixel_per_image/absolute_difference_per_pixel_per_image_FGFP.png")
plt.show()

for region, data in diff_per_img_per_pixel_dataframe_dictionary.items():
    if region not in fgfp_regions:
        ax = sns.kdeplot(data, label=region)
plt.xlim((-0.5, 0.5))
plt.xlabel("Difference in UQS per Pixel")
ax.legend()
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.title("Difference in UQS per Pixel per Image - Existing Regions")
plt.subplots_adjust(right=0.67) # get labels to not be cut off
plt.savefig(f"./distribution_plots/difference_per_pixel_per_image/difference_per_pixel_per_image_EXISTING.png")
plt.show()

for region, data in diff_per_img_per_pixel_dataframe_dictionary.items():
    if region in fgfp_regions:
        ax = sns.kdeplot(data, label=region)
plt.xlim((-0.5, 0.5))
plt.xlabel("Difference in UQS per Pixel")
ax.legend()
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.title("Difference in UQS per Pixel per Image - FGFP Regions")
plt.subplots_adjust(right=0.67) # get labels to not be cut off
plt.savefig(f"./distribution_plots/difference_per_pixel_per_image/difference_per_pixel_per_image_FGFP.png")
plt.show()




# ========= Combine difference blackout region score files in one file =============

dbrs_combined_df = pd.concat([pd.read_csv(file, sep=';') for file in difference_blackout_region_scores_filenames], ignore_index=True)
dbrs_combined_df.to_csv(f"./difference_blackout_region_scores/VGGFace200k-50-images-region_UQS_difference_from_original-ALL.csv", sep=';', index=False)


# ========= Combine difference per pixel blackout region score files in one file =============

dpp_combined_df = pd.concat([pd.read_csv(file, sep=';') for file in difference_per_pixel_blackout_region_scores_filenames], ignore_index=True)
dpp_combined_df.to_csv(f"./difference_per_pixel_blackout_region_scores/VGGFace200k-50-images-region_UQS_difference_per_pixel_from_original-ALL.csv", sep=';', index=False)






# ========= Save files with averages =============

def save_dictionary_as_csv(dictionary, output_file):
    with open(output_file, 'w', newline='') as output_csv:
        writer = csv.writer(output_csv, delimiter=';')
        for key, value in dictionary.items():
            writer.writerow([key, value])


output_file = "./average_blackout_region_scores/VGGFace200k-50-images-average_blackout_region_scores.csv"
save_dictionary_as_csv(region_averages_dictionary, output_file)

output_file = "./average_absolute_difference_from_original_scores/VGGFace200k-50-images-average_absolute_difference_from_original_scores.csv"
save_dictionary_as_csv(region_absolute_differences_averages_dictionary, output_file)

output_file = "./average_absolute_difference_per_pixel_per_image/VGGFace200k-50-images-average_absolute_difference_per_pixel_per_image.csv"
save_dictionary_as_csv(region_average_absolute_difference_per_pixel_per_image_dictionary, output_file)

output_file = "./average_difference_per_pixel_per_image/VGGFace200k-50-images-average_difference_per_pixel_per_image.csv"
save_dictionary_as_csv(region_average_difference_per_pixel_per_image_dictionary, output_file)

output_file = "./number_of_images_with_no_pixels_per_region/VGGFace200k-50-images-Number-of-images-with-no-pixels-per-region.csv"
save_dictionary_as_csv(region_number_of_images_with_0_pixels_dictionary, output_file)
