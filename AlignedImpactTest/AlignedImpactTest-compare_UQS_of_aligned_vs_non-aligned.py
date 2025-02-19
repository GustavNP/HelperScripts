

import pandas as pd
import matplotlib.pyplot as plt


pd.options.display.max_colwidth = 200

aligned_scores_file = ".\VGGFace200k-50-image-blackout-regions-UQS-ALL.csv"
non_aligned_scores_file = ".\VGGFace2-200k-all-OFIQ-only-filenames.csv"


aligned_df = pd.read_csv(aligned_scores_file, sep=';')
non_aligned_df = pd.read_csv(non_aligned_scores_file, sep=';')
aligned_df = aligned_df[aligned_df["Filename"].apply(lambda x: "Blackout" not in x)]


aligned_df = aligned_df[["Filename", "UnifiedQualityScore", "UnifiedQualityScore.scalar"]]
non_aligned_df = non_aligned_df[["Filename", "UnifiedQualityScore", "UnifiedQualityScore.scalar"]]


combined_df = pd.merge(aligned_df, non_aligned_df, on='Filename', how='inner', suffixes=["_aligned", "_non_aligned"])

print(len(combined_df))
print(combined_df)

combined_df["UQS_difference"] = combined_df["UnifiedQualityScore.scalar_non_aligned"] - combined_df["UnifiedQualityScore.scalar_aligned"]
combined_df["UQS_absolute_difference"] = combined_df["UQS_difference"].abs()


print(combined_df.describe())


higher_than_10_df = combined_df[combined_df["UQS_absolute_difference"].apply(lambda x: x > 10)]
print("Number of images with absolute difference higher than 10:")
print(len(higher_than_10_df))
print("Mean of those higher than 10:")
print(higher_than_10_df[["UQS_difference", "UQS_absolute_difference"]].mean())

higher_than_20_df = combined_df[combined_df["UQS_absolute_difference"].apply(lambda x: x > 20)]
print("Number of images with absolute difference higher than 20:")
print(len(higher_than_20_df))


higher_than_30_df = combined_df[combined_df["UQS_absolute_difference"].apply(lambda x: x > 30)]
print("Number of images with absolute difference higher than 30:")
print(len(higher_than_30_df))


higher_than_40_df = combined_df[combined_df["UQS_absolute_difference"].apply(lambda x: x > 40)]
print("Number of images with absolute difference higher than 40:")
print(len(higher_than_40_df))



higher_than_50_df = combined_df[combined_df["UQS_absolute_difference"].apply(lambda x: x > 50)]
print("Number of images with absolute difference higher than 50:")
print(len(higher_than_50_df))

print("Images with absolute difference higher than 40:")
print(higher_than_40_df)

# combined_df["UQS_absolute_difference"].plot(kind="hist")
# # combined_df["UQS_absolute_difference"].plot(kind="kde")

# plt.yscale("log")
# plt.show()


# combined_df["UQS_difference"].plot(kind="hist")
combined_df.hist(column='UQS_difference', bins=50, grid=True, rwidth=.8, color='purple')
# combined_df["UQS_difference"].plot(kind="kde")

plt.yscale("log")
plt.show()

# combined_df.to_csv(output_file, sep=';', index=False)