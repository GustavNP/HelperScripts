import pandas as pd


# input_df = pd.read_csv('C:\\Users\\admin\\source\\repos\\FaceSimilarity\\similarity_score_files\\similarity_scores_Facenet512_yunet_VGGFace-200k-train-431-502.csv', sep=';', header=None)
# input_df = pd.read_csv('C:\\Users\\admin\\source\\repos\\FaceSimilarity\\larger_embedding_files\\embeddings_Facenet512_yunet_VGGFace-200k-train-431-502.csv', sep=';', header=None)

# input_df = pd.read_csv('C:\\Users\\admin\\source\\repos\\FaceSimilarity\\quality_score_files\\Predicted-UQS_Test_set-KNeighbors_VGGFace-200k.csv', sep=';', header=None)
# input_df = pd.read_csv('C:\\Users\\admin\\source\\repos\\FaceSimilarity\\quality_score_files\\Predicted-UQS_Test_set-RFR_SPECIFIC_9_5-Fold-CV-all-features-vggface200k.csv', sep=';', header=None)
# input_df = pd.read_csv('C:\\Users\\admin\\source\\repos\\FaceSimilarity\\quality_score_files\\Predicted-UQS_Test_set-RFR-Ablation_Top_10_features_VGGFace-200k.csv', sep=';', header=None)
# input_df = pd.read_csv('C:\\Users\\admin\\source\\repos\\FaceSimilarity\\quality_score_files\\Predicted-UQS_Test_set-RFR-Ablation_Top_15_features_VGGFace-200k.csv', sep=';', header=None)
# input_df = pd.read_csv('C:\\Users\\admin\\source\\repos\\FaceSimilarity\\quality_score_files\\Predicted-UQS_Test_set-RFR-Ablation_Top_20_features_VGGFace-200k.csv', sep=';', header=None)
# input_df = pd.read_csv('C:\\Users\\admin\\source\\repos\\FaceSimilarity\\quality_score_files\\VGGFace2-200k-all-OFIQ.csv', sep=';', header=None)
# input_df = pd.read_csv('C:\\Users\\admin\\source\\repos\\FaceSimilarity\\quality_score_files\\Predicted-UQS_Test_set-RFR-SPECIFIC-14-All-Features-VGGFace200k.csv', sep=';', header=None)
# input_df = pd.read_csv('C:\\Users\\admin\\source\\repos\\FaceSimilarity\\quality_score_files\\Predicted-UQS_Test_set-RFR-SPECIFIC-15-All-Features-VGGFace200k.csv', sep=';', header=None)
# input_df = pd.read_csv('C:\\Users\\admin\\source\\repos\\FaceSimilarity\\quality_score_files\\Predicted-UQS_Test_set_RFR-SPECIFIC-14-NoPreprocessing-VGGFace200k.csv', sep=';', header=None)
# input_df = pd.read_csv('C:\\Users\\admin\\source\\repos\\FaceSimilarity\\quality_score_files\\Predicted-UQS_SPECIFIC-14-OcclusionFeaturesRemoved.csv', sep=';', header=None)
# input_df = pd.read_csv('output_files\\VGGFace200k-50-images-existing-regions-blackout.csv', sep=';')
# input_df = pd.read_csv('output_files\\VGGFace200k-50-images-per-identity-blackout-UQS-scores.csv', sep=';')
# input_df = pd.read_csv('output_files\\VGGFace200k-50-images-new-regions-blackout.csv', sep=';')
# input_df = pd.read_csv('output_files\\region_pixel_count-VGGFace200k-ALL.csv', sep=';', header=None)
# input_df = pd.read_csv('C:\\Users\\admin\\source\\repos\\RandomForestUQS\\VGGFace200k-random-state-36-test_set.csv', sep=';')
input_df = pd.read_csv('C:\\Users\\admin\\source\\repos\\FaceSimilarity\\similarity_score_files\\similarity_scores_Facenet512_yunet_VGGFace-200k-rs-36-test-set.csv', sep=';', header=None)



print("loaded dataframe")

# input_df[0] = ["aligned_" + x.split('/')[-1] for x in input_df[0]]
input_df[0] = [x.split('/')[-1] for x in input_df[0]]
input_df[1] = [x.split('/')[-1] for x in input_df[1]]

# input_df["Filename"] = [x.split('/')[-1] for x in input_df["Filename"]]
# input_df["Filename"] = ["aligned_" + x.split('/')[-1] for x in input_df["Filename"]]


print(input_df.head())

# input_df.to_csv('C:\\Users\\admin\\source\\repos\\FaceSimilarity\\larger_embedding_files\\embeddings_Facenet512_yunet_VGGFace-200k-train-431-502-only-filenames.csv', sep=';', header=None)
# input_df.to_csv('C:\\Users\\admin\\source\\repos\\FaceSimilarity\\quality_score_files\\Predicted-UQS_Test_set-RFR_SPECIFIC_9_5-Fold-CV-all-features-vggface200k-only-filenames.csv', sep=';', header=False, index=False)
# input_df.to_csv('C:\\Users\\admin\\source\\repos\\FaceSimilarity\\quality_score_files\\Predicted-UQS_Test_set-RFR-Ablation_Top_10_features_VGGFace-200k-only-filenames.csv', sep=';', header=False, index=False)
# input_df.to_csv('C:\\Users\\admin\\source\\repos\\FaceSimilarity\\quality_score_files\\Predicted-UQS_Test_set-RFR-Ablation_Top_15_features_VGGFace-200k-only-filenames.csv', sep=';', header=False, index=False)
# input_df.to_csv('C:\\Users\\admin\\source\\repos\\FaceSimilarity\\quality_score_files\\Predicted-UQS_Test_set-RFR-Ablation_Top_20_features_VGGFace-200k-only-filenames.csv', sep=';', header=False, index=False)
# input_df.to_csv('C:\\Users\\admin\\source\\repos\\FaceSimilarity\\quality_score_files\\VGGFace2-200k-all-OFIQ-only-filenames.csv', sep=';', header=False, index=False)
# input_df.to_csv('C:\\Users\\admin\\source\\repos\\FaceSimilarity\\quality_score_files\\Predicted-UQS_Test_set-RFR-SPECIFIC-14-All-Features-VGGFace200k-only-filenames.csv', sep=';', header=False, index=False)
# input_df.to_csv('C:\\Users\\admin\\source\\repos\\FaceSimilarity\\quality_score_files\\Predicted-UQS_Test_set-RFR-SPECIFIC-15-All-Features-VGGFace200k-only-filenames.csv', sep=';', header=False, index=False)
# input_df.to_csv('C:\\Users\\admin\\source\\repos\\FaceSimilarity\\quality_score_files\\Predicted-UQS_SPECIFIC-14-OcclusionFeaturesRemoved-only-filenames.csv', sep=';', header=False, index=False)
# input_df.to_csv('output_files\\VGGFace200k-50-images-existing-regions-blackout-only-filenames.csv', sep=';', index=False)
# input_df.to_csv('output_files\\VGGFace200k-50-images-per-identity-blackout-UQS-scores-only-filenames.csv', sep=';', index=False)
# input_df.to_csv('output_files\\VGGFace200k-50-images-new-regions-blackout-only-filenames.csv', sep=';', index=False)
# input_df.to_csv('output_files\\region_pixel_count-VGGFace200k-ALL-2.csv', sep=';', index=False, header=False)
# input_df.to_csv('C:\\Users\\admin\\source\\repos\\RandomForestUQS\\VGGFace200k-random-state-36-test_set-only-filenames.csv', sep=';', index=False)
input_df.to_csv('C:\\Users\\admin\\source\\repos\\FaceSimilarity\\similarity_score_files\\similarity_scores_Facenet512_yunet_VGGFace-200k-rs-36-test-set-only-filenames.csv', sep=';', index=False, header=False)

