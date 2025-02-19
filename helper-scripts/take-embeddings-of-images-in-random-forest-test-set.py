

import pandas as pd
import os



files_directory = "C:\\Users\\admin\\source\\repos\\FaceSimilarity\\embedding_files"

test_set_df = pd.read_csv("C:\\Users\\admin\\source\\repos\\RandomForestUQS\\predicted_UQS_files\\Predicted-UQS_Test_set.csv", sep=';')
test_set_df["Filename"] = ['aligned_' + x.split('/')[-1] for x in test_set_df["Filename"]]

filenames_in_test_set_dictionary_for_lookup = pd.Series(test_set_df["UQS"].values, index=test_set_df["Filename"]).to_dict()

for root, dirs, files in os.walk(files_directory):
    for file in files:
        if file.endswith(".csv"):
            file_path = os.path.join(root, file)
            file_df = pd.read_csv(file_path, sep=';', header=None)
            file_df[1] = [x.split('/')[-1] for x in file_df[1]]
            file_df = file_df[file_df[1].isin(filenames_in_test_set_dictionary_for_lookup)]
            file_name, file_extension = os.path.splitext(file)
            person = file_name.split('_')[-1]
            file_df.to_csv(f"./embedding-files-test-set/embeddings-vggface200k-{person}-images-in-test-set.csv", sep=';', index=False, header=False)
