
import pandas as pd
import os

files_directory = "C:\\Users\\admin\\source\\repos\\OFIQ-Project-FGFP\\install_x86_64\\Release\\bin\\score_files"
output_file = "./output_files/VGGFace200k-50-images-per-identity-blackout-UQS-scores.csv"


input_files = []
for root, dirs, files in os.walk(files_directory):
    for file in files:
        # if file.endswith(".csv") and len(file.split('-')) == 2: # IMPORTANT: assumes the actual score files to use has the structure <identity>-scores.csv, where all other files have more '-' in them.
        if file.endswith(".csv") and 'blackout' in file: # IMPORTANT: assumes the only actual score files to use has contains 'blackout' in their filename.
            file_path = os.path.join(root, file)
            input_files.append(file_path)


combined_df = pd.concat([pd.read_csv(file) for file in input_files], ignore_index=True)
combined_df.to_csv(output_file, index=False)

print(f"Combined score file saved as {output_file}")
