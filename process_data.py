# parsing logic heavily inspired from: https://medium.com/atlas-research/ner-for-clinical-text-7c73caddd180

import glob
import os
import pandas as pd
import numpy as np

# data_dir: Path to directory containing .con and .txt files
def generate_annotated_data(data_dir):
    concept_corpus = glob.glob(data_dir+'concept/*.con')  # Make list of concept files
    text_corpus = glob.glob(data_dir+'txt/*.txt')  # Make list of documents

    concept_filenames = []
    text_filenames = []

    # Use regex to create doc id 

    for con in concept_corpus:
        concept_filenames.append(os.path.basename(con).split('.')[0])
    for doc in text_corpus:
        text_filenames.append(os.path.basename(doc).split('.')[0])

    concept_filenames = tuple(sorted(concept_filenames)) 
    text_filenames = tuple(sorted(text_filenames))

    intersection = list(set(concept_filenames) & set(text_filenames))
    if len(intersection) == len(concept_filenames):
        print("Count of concept files with corresponding doc:", len(intersection))

    # # Build annotation and entry corpora

    # concepts
    concept_corpus = []
    # texts
    text_corpus = []

    for f_id in concept_filenames:
        path = data_dir + "concept/" + f_id +".con"
        with open(path) as f:
            content = f.read().splitlines()
            concept_corpus.append(content)

        path = data_dir + "txt/" + f_id +".txt"
        with open(path) as f:
            content = f.read().splitlines()
            text_corpus.append(content)

    entries_cols = ["id", "row", "offset", "word", "line_num"]
    entries_df = pd.DataFrame(columns=entries_cols)

    annotations_cols = ["id", "NER_tag", "row", "offset", "length"]
    annotations_df = pd.DataFrame(columns=annotations_cols)

    annotations_df = pd.DataFrame(columns=annotations_cols)  # Reset df
    tmp_list = []  # Set up variable to hold row info

    for i, document in enumerate(concept_corpus):
        
        for row in document:
            row = row.split("||")
            text_info = row[0]
            type_info = row[1]
            
            text = text_info.split('"')[1]
            
            offset_start = text_info.split(' ')[-2]
            offset_end = text_info.split(' ')[-1]
            
            line = offset_start.split(':')[0] # Given one sentence to line, 
                                            # line number will be the same for offset_start and offset_end
            
            word_offset_start = int(offset_start.split(':')[1])
            word_offset_end = int(offset_end.split(':')[1])
            length = word_offset_end-word_offset_start +1
            
            a_type = type_info.split('"')[-2]
            
            # Split text into tokens with IOB tags
            first = True  # Set up flag to id start of text
            BIO_tag = "B-"
            if length > 1:  # Isolate text with multiple tokens 
                for offset in range(word_offset_start, word_offset_end+1):
                    if first:
                        tag_label = BIO_tag + a_type # Set tag for first word to start with B-
                        first = False  # Change flag
                    else:
                        tag_label = tag_label.replace("B-", "I-")
                    tmp_list.append([concept_filenames[i], tag_label, line, offset, 1])                
            else:
                tmp_list.append([concept_filenames[i], BIO_tag + a_type, line, word_offset_start, length])
            
    annotations_df = pd.DataFrame(tmp_list, columns=annotations_cols)
    annotations_df = annotations_df.drop(columns=["length"])

    # print(annotations_df.head(50))

    entries_df = pd.DataFrame(columns=entries_cols)  # Reset df
    tmp_list = []

    row_offset = 0
    for doc_i, document in enumerate(text_corpus):
        
        tmp_list.append([0, 0, 0, "-DOCSTART-", 0])
        tmp_list.append([0, 0, 0, "-EMPTYLINE-", 0])
        rows_in_doc = 0
        for row_i, row in enumerate(document):
            row_split = row.split(" ")
            for word_i, word in enumerate(row_split):
                word = word.replace("\t", "")
                word_id = concept_filenames[doc_i]
                word_row = row_i+1  # 1-based indexing 
                word_offset = word_i # 0-based indexing
                
                if len(word) > 0 and "|" not in word:
                    tmp_list.append([word_id, word_row, word_offset, word, word_row + row_offset])
            rows_in_doc += 1
        row_offset += rows_in_doc
        tmp_list.append([0, 0, 0, "-EMPTYLINE-", 0])

    entries_df = pd.DataFrame(tmp_list, columns=entries_cols)

    # Ensure correct dtypes

    annotations_df[['row', 'offset']] = annotations_df[['row', 'offset']].apply(pd.to_numeric)
    annotations_df['NER_tag'] = annotations_df["NER_tag"].astype(str)
    entries_df[['row', 'offset', 'line_num']] = entries_df[['row', 'offset', 'line_num']].apply(pd.to_numeric)
    entries_df["word"] = entries_df["word"].astype(str)

    result_df = pd.merge(entries_df, annotations_df, how="left", on=['id', 'row', 'offset'])

    # Check for NaNs (should be only in NER_tag, where NaNs will be replaced with "O" (outside))
    print("Columns with missing data:\n", result_df.isna().any())

    result_df = result_df.fillna("O")
    result_df = result_df.drop(columns=["id", "row", "offset"])

    print(result_df.head(50))

    ner_counter = [1 for i in result_df["NER_tag"] if "B-" in i]
    print(len(ner_counter), "named entities and", result_df.shape[0], "tokens")
    return result_df

training_set = generate_annotated_data('./data/training_data/')
test_set = generate_annotated_data('./data/test_data/')
np.savetxt("./processed/train.txt", training_set.values, fmt="%s")
np.savetxt("./processed/test.txt", test_set.values, fmt="%s")

#np.savetxt(output_dir+"train.txt", result_df.values, fmt="%s")