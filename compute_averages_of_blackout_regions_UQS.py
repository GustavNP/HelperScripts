import os
import pandas as pd
import csv
import cv2

# =========== TODO: CONSIDER THIS ==========
# Okay, I'm going to keep this here for now, but the non-absolute deviation is just the average of the originals minus the average of the blacked out region,
# so very easy to calculate and not really providing any new value.
# Earlier thought: "Right now, we only compute the average absolute deviation. This only tells us the deviation, not in which direction.
# Computing the average deviation (not absolute) would give us insight into the average direction."
# ==========================================
  
pd.options.display.max_colwidth = 200

score_file = "./output_files/VGGFace200k-50-images-per-identity-blackout-UQS-scores.csv"
score_df = pd.read_csv(score_file, sep=';')

originals_df = score_df[score_df["Filename"].apply(lambda filename: "_Blackout_" not in filename)]
print(len(originals_df))

average_UQS_originals = originals_df["UnifiedQualityScore.scalar"].mean()
print(f"Average score for original aligned images: {average_UQS_originals}")

region_names = [
    "Nasal",
    "LeftOrbital",
    "RightOrbital",
    "Mental",
    "LeftBuccal",
    "RightBuccal",
    "LeftZygoInfraParo",
    "RightZygoInfraParo"
]


region_pixel_count_file = "./output_files/region_pixel_count.csv" # should have row structure: <blackout-image-full-path>;<region-pixel-count>
region_pixel_count_df = pd.read_csv(region_pixel_count_file, sep=';', header=None)
region_pixel_count_df.columns = ['person', 'Filename', 'pixel_count']
region_pixel_count_df["Filename"] = region_pixel_count_df["Filename"].apply(lambda x: x.split("\\")[-1]) # only keep actual filename, not full path


region_averages_dictionary = { "Originals" : average_UQS_originals }
region_deviations_averages_dictionary = { "Originals" : 0.0 }
region_average_deviation_per_pixel_per_image_dictionary = { "Originals" : 0.0 }
for region in region_names:
    blackout_region_UQS_scores_df = score_df[score_df["Filename"].apply(lambda filename: region in filename)]
    print(len(blackout_region_UQS_scores_df))

    #blackout_region_UQS_scores_weird_values_df = blackout_region_UQS_scores_df[blackout_region_UQS_scores_df["UnifiedQualityScore.scalar"].apply(lambda x: x < 0)]
    #print(len(blackout_region_UQS_scores_weird_values_df))
    average_UQS = blackout_region_UQS_scores_df["UnifiedQualityScore.scalar"].mean()
    print(f"Average score for images with {region} region blacked out: {average_UQS}")
    region_averages_dictionary[region] = average_UQS

    # Compute deviations from original
    deviation_from_original_df = originals_df["UnifiedQualityScore.scalar"] - blackout_region_UQS_scores_df["UnifiedQualityScore.scalar"].values
    #print(deviation_from_original_df.head())
    #print(blackout_region_UQS_scores_df.dtypes)
    deviation_from_original_absolute_df = deviation_from_original_df.abs()

    # Compute average deviation from original
    average_absolute_deviation_from_original_UQS = deviation_from_original_absolute_df.mean()
    region_deviations_averages_dictionary[region] = average_absolute_deviation_from_original_UQS
    print(f"Average (absolute) deviation from original for images with {region} blacked out: {average_absolute_deviation_from_original_UQS}")

    # Save blacked out image deviations from original images
    deviations_from_original_with_filename_df = pd.concat([blackout_region_UQS_scores_df["Filename"].reset_index(drop=True), deviation_from_original_df.reset_index(drop=True)], axis=1, ignore_index=True)
    #print(deviations_from_original_with_filename_df.head())
    deviations_from_original_with_filename_df.to_csv(f"./deviation_blackout_region_scores/VGGFace200k-50-images-region_UQS_deviation_from_original-{region}.csv")

    # Compute average deviation per pixel per image
    absolute_deviations_df = pd.concat([blackout_region_UQS_scores_df["Filename"].reset_index(drop=True), deviation_from_original_absolute_df.reset_index(drop=True)], axis=1, ignore_index=True)
    absolute_deviations_df.columns = ["Filename", "UQSDeviationFromOriginal"]
    absolute_deviations_df["Filename"] = absolute_deviations_df["Filename"].apply(lambda x: x.split("/")[-1]) # only keep actual filename, not full path
    absolute_deviations_and_pixel_count_df = pd.merge(absolute_deviations_df, region_pixel_count_df, on='Filename', how='left')
    missing_pixel_counts = absolute_deviations_and_pixel_count_df["pixel_count"].isnull().sum()
    print(f"Missing pixel counts for {region}: {missing_pixel_counts}")
    # print(absolute_deviations_and_pixel_count_df)
    deviation_per_pixel_df = absolute_deviations_and_pixel_count_df["UQSDeviationFromOriginal"].div(absolute_deviations_and_pixel_count_df["pixel_count"], axis=0)
    # print(deviation_per_pixel_df.head())
    average_deviation_per_pixel_per_image = deviation_per_pixel_df.mean()
    region_average_deviation_per_pixel_per_image_dictionary[region] = average_deviation_per_pixel_per_image
    print(f"Average deviation per pixel per image for images with {region} blacked out: {average_deviation_per_pixel_per_image}")



output_file = "./average_blackout_region_scores/VGGFace200k-50-images-average_blackout_region_scores.csv"
with open(output_file, 'w', newline='') as output_csv:
    writer = csv.writer(output_csv, delimiter=';')
    for region_name, average in region_averages_dictionary.items():
        writer.writerow([region_name, average])


output_file = "./average_deviation_from_original_scores/VGGFace200k-50-images-average_deviation_from_original_scores.csv"
with open(output_file, 'w', newline='') as output_csv:
    writer = csv.writer(output_csv, delimiter=';')
    for region_name, average in region_deviations_averages_dictionary.items():
        writer.writerow([region_name, average])

output_file = "./average_deviation_per_pixel_per_image/VGGFace200k-50-images-average_deviation_per_pixel_per_image.csv"
with open(output_file, 'w', newline='') as output_csv:
    writer = csv.writer(output_csv, delimiter=';')
    for region_name, average in region_average_deviation_per_pixel_per_image_dictionary.items():
        writer.writerow([region_name, average])


