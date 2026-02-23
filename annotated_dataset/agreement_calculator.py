import pandas as pd
import numpy as np
import argparse

file_1_name = "output_batch_1.xlsx"
file_2_name = "output_batch_4.xlsx"

if __name__ == "__main__":

    df_1 = pd.read_excel(file_1_name)
    df_2 = pd.read_excel(file_2_name)

    # Find common sentence pairs between the two files.
    common_sentences = df_1.merge(df_2, on=["Sentence 1", "Sentence 2"], how="inner")

    #print dataframe

    # Our agreement score is based on the semantic score assigned by each annotater

    #first assigned integer labels to each score

    common_sentences.loc[common_sentences["Word Semantic Score_x"] == 0.0, "Word Semantic Score_x"] = 0
    common_sentences.loc[common_sentences["Word Semantic Score_y"] == 0.0, "Word Semantic Score_y"] = 0

    common_sentences.loc[common_sentences["Word Semantic Score_x"] == 1.0, "Word Semantic Score_x"] = 2
    common_sentences.loc[common_sentences["Word Semantic Score_y"] == 1.0, "Word Semantic Score_y"] = 2

    common_sentences.loc[common_sentences["Word Semantic Score_x"] == 0.5, "Word Semantic Score_x"] = 1
    common_sentences.loc[common_sentences["Word Semantic Score_y"] == 0.5, "Word Semantic Score_y"] = 1




    print(common_sentences.to_string())

    # Construct a confusion matrix
    confusion_matrix = pd.crosstab(common_sentences["Word Semantic Score_x"], common_sentences["Word Semantic Score_y"])
    print(confusion_matrix)

    # calculate trace across diagonal
    p_obs = np.trace(confusion_matrix.values)
    print('Agreement # Observed: ', p_obs)

    # calculate the sum of every row and column
    contingency_mat = np.zeros((2, confusion_matrix.shape[0]))
    for i in range(confusion_matrix.shape[0]):
        contingency_mat[0][i] = confusion_matrix.iloc[i].sum() # row sum
        contingency_mat[1][i] = confusion_matrix.iloc[:,i].sum() # col sum

    print(contingency_mat)

    row_sums = confusion_matrix.sum(axis=1).values
    col_sums = confusion_matrix.sum(axis=0).values

    p_ef = np.sum((row_sums * col_sums) / common_sentences.shape[0])

    expected_agreement = np.sum(p_ef)

    print('Agreement Expected: ', expected_agreement)

    # Now we can get kappa

    kappa = (p_obs - expected_agreement) / (common_sentences.shape[0] - expected_agreement)

    print('Kappa: ', kappa)







