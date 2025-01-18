import os
import csv

input_directory = "C:/Users/admin/Downloads/archive1/val"
output_file = "./output_files/identities_in_directory.csv"

with open(output_file, 'w', newline='') as output_csv:
    writer = csv.writer(output_csv, delimiter=';')

    for root, dirs, files in os.walk(input_directory):
        for dir in dirs:
            writer.writerow([dir])
