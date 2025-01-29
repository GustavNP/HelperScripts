

import pandas as pd
import os
import shutil



archive1_val_directory = "C:\\Users\\admin\\Downloads\\archive1\\val"
archive1_train_directory = "C:\\Users\\admin\\Downloads\\archive1\\train"

# test_set_df = pd.read_csv("C:\\Users\\admin\\source\\repos\\RandomForestUQS\\VGGFace200k-random-state-36-test_set-only-filenames.csv", sep=';')
# filenames_in_test_set_dictionary_for_lookup = pd.Series(test_set_df["UnifiedQualityScore.scalar"].values, index=test_set_df["Filename"]).to_dict()


train_set_df = pd.read_csv("C:\\Users\\admin\\source\\repos\\RandomForestUQS\\VGGFace200k-random-state-36-train_set-only-filenames.csv", sep=';')
filenames_in_test_set_dictionary_for_lookup = pd.Series(train_set_df["UnifiedQualityScore.scalar"].values, index=train_set_df["Filename"]).to_dict()


# output_directory = "C:\\Users\\admin\\Downloads\\archive1\\rs-36-test-set"
output_directory = "C:\\Users\\admin\\Downloads\\archive1\\rs-36-train-set"

for root, dirs, files in os.walk(archive1_val_directory):
    for file in files:
        if file.endswith(".jpg"):
            if file in filenames_in_test_set_dictionary_for_lookup:
                original_file_path = os.path.join(root, file)
                new_file_path = os.path.join(output_directory, file)
                shutil.copy(original_file_path, new_file_path)


for root, dirs, files in os.walk(archive1_train_directory):
    for file in files:
        if file.endswith(".jpg"):
            if file in filenames_in_test_set_dictionary_for_lookup:
                original_file_path = os.path.join(root, file)
                new_file_path = os.path.join(output_directory, file)
                shutil.copy(original_file_path, new_file_path)


