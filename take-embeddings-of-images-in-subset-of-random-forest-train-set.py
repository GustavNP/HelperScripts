

import pandas as pd
import os
import random
import numpy as np



files_directory = "C:\\Users\\admin\\source\\repos\\FaceSimilarity\\embedding_files"

test_set_df = pd.read_csv("C:\\Users\\admin\\source\\repos\\RandomForestUQS\\predicted_UQS_files\\Predicted-UQS_Test_set-RFR_SPECIFIC_9_5-Fold-CV-all-features-vggface200k.csv", sep=';')
test_set_df["Filename"] = ['aligned_' + x.split('/')[-1] for x in test_set_df["Filename"]]

filenames_in_test_set_dictionary_for_lookup = pd.Series(test_set_df["UQS"].values, index=test_set_df["Filename"]).to_dict()

for root, dirs, files in os.walk(files_directory):
    for file in files:
        if file.endswith(".csv"):
            file_path = os.path.join(root, file)
            file_df = pd.read_csv(file_path, sep=';', header=None)
            file_df[1] = [x.split('/')[-1] for x in file_df[1]]
            file_df = file_df[~file_df[1].isin(filenames_in_test_set_dictionary_for_lookup)]
            print(file_df)

            number_of_images_to_use = round(len(file_df) / 4) # around 25% of the images
            rows_to_choose = []
            random.seed(36)
            for i in range(0,number_of_images_to_use):
                random_number = random.randrange(len(file_df))
                while(random_number in rows_to_choose): # get number not already in list
                    random_number = random.randrange(len(file_df))
                rows_to_choose.append(random_number)
            
            file_df = file_df.set_index(keys=np.arange(0, len(file_df)))
            print(file_df)
            file_df = file_df.iloc[rows_to_choose]
            print(file_df)
        
            file_name, file_extension = os.path.splitext(file)
            person = file_name.split('_')[-1]
            file_df.to_csv(f"./embedding-files-25-percent-of-train-set/embeddings-vggface200k-{person}-25-percent-of-images-in-Train-set.csv", sep=';', index=False, header=False)
            # file_df.to_csv(f"./embedding-files-test-set/embeddings-vggface200k-{person}-images-in-test-set.csv", sep=';', index=False, header=False)
