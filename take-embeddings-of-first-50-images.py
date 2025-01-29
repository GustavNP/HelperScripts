

import pandas as pd
import os



files_directory = "C:\\Users\\admin\\source\\repos\\FaceSimilarity\\embedding_files"


images_50_directory = "C:\\Users\\admin\\Desktop\\aligned-VGGFace200k-folders-50-images"
dictionary_for_looking_up_filenames = {}
for root, dirs, files in os.walk(images_50_directory):
    for file in files:
        dictionary_for_looking_up_filenames[file] = 0 # only use dictionary for lookup of key





for root, dirs, files in os.walk(files_directory):
    for file in files:
        if file.endswith(".csv"):
            file_path = os.path.join(root, file)
            file_df = pd.read_csv(file_path, sep=';', header=None)
            file_df[1] = [x.split('/')[-1] for x in file_df[1]]
            file_df = file_df[file_df[1].isin(dictionary_for_looking_up_filenames)]
            file_name, file_extension = os.path.splitext(file)
            person = file_name.split('_')[-1]
            file_df.to_csv(f"./embedding-files-50-images/embeddings-vggface200k-{person}-50-images.csv", sep=';', index=False, header=False)
