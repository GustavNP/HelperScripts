
import pandas as pd
import os

input_file = "./output_files/VGGFace2-200k-all.csv"
output_file = "./output_files/VGGFace-200k-431-502.csv"

input_file_df = pd.read_csv(input_file, sep=';')
person_dataframes = []
for i in range(431,503):
    person_name = f"n000{i}"
    person_dataframes.append(input_file_df[input_file_df['Filename'].apply(lambda x: person_name in x)])


combined_df = pd.concat(person_dataframes, ignore_index=True)
combined_df.to_csv(output_file, sep=';', index=False)

print(f"Combined score file saved as {output_file}")
