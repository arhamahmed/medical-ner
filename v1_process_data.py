# parsing logic from: https://medium.com/atlas-research/ner-for-clinical-text-7c73caddd180

import glob  # Finds all the pathnames matching a specified pattern, 
             # typically specified with regex (re) 
import re
import pandas as pd

# pd.set_option('max_colwidth', None)  # Remove any limitation on length 
#                                      # of text displayed in a cell
# pd.set_option('max_rows', 300)  # Display up to 300 rows in a dataset

data_dir = './data/training_data/set_one/'  # Path to directory containing .con and .txt files
output_dir = data_dir+'output/'

a_corpus = glob.glob(data_dir+'concept/*.con')  # Make list of concept files
e_corpus = glob.glob(data_dir+'txt/*.txt')  # Make list of documents

# TODO: for set_one this is record, for set_two it is something else. prob diff for test
base_str = 'record-' 

a_ids = []
e_ids = []

# Use regex to create doc id 

for con in a_corpus:
    f_id = re.findall(r'\d+', con)[0]
    a_ids.append(f_id)
for doc in e_corpus:
    f_id = re.findall(r'\d+', doc)[0]
    e_ids.append(f_id)
    
a_ids = tuple(sorted(a_ids)) 
e_ids = tuple(sorted(e_ids))

intersection = list(set(a_ids) & set(e_ids))
if len(intersection) == len(a_ids):
    print("Count of concept files with corresponding doc:", len(intersection))

# # Build annotation and entry corpora

# concepts
a_corpus = []
# texts
e_corpus = []

for f_id in a_ids:
    path = data_dir + "concept/" + base_str + f_id +".con"
    with open(path) as f:
        content = f.read().splitlines()
        a_corpus.append(content)

    path = data_dir + "txt/" + base_str + f_id +".txt"
    with open(path) as f:
        content = f.read().splitlines()
        e_corpus.append(content)

entries_cols = ["id", "row", "offset", "word"]
entries_df = pd.DataFrame(columns=entries_cols)

annotations_cols = ["id", "NER_tag", "row", "offset", "length"]
annotations_df = pd.DataFrame(columns=annotations_cols)

annotations_df = pd.DataFrame(columns=annotations_cols)  # Reset df
tmp_list = []  # Set up variable to hold row info

for i, document in enumerate(a_corpus):
    
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
                tmp_list.append([a_ids[i], tag_label, line, offset, 1])                
        else:
            tmp_list.append([a_ids[i], BIO_tag + a_type, line, word_offset_start, length])
        
annotations_df = pd.DataFrame(tmp_list, columns=annotations_cols)
annotations_df = annotations_df.drop(columns=["length"])

print(annotations_df.head(50))

entries_df = pd.DataFrame(columns=entries_cols)  # Reset df
tmp_list = []

for doc_i, document in enumerate(e_corpus):
    
    tmp_list.append([0, 0, 0, "-DOCSTART-"])
    tmp_list.append([0, 0, 0, "-EMPTYLINE-"])
    
    for row_i, row in enumerate(document):
        row_split = row.split(" ")
        for word_i, word in enumerate(row_split):
            word = word.replace("\t", "")
            word_id = a_ids[doc_i]
            word_row = row_i+1  # 1-based indexing 
            word_offset = word_i # 0-based indexing
            
            if len(word) > 0 and "|" not in word:
                tmp_list.append([word_id, word_row, word_offset, word])
        
    tmp_list.append([0, 0, 0, "-EMPTYLINE-"])

entries_df = pd.DataFrame(tmp_list, columns=entries_cols)

# Ensure correct dtypes

annotations_df[['id', 'row', 'offset']] = annotations_df[['id', 'row', 'offset']].apply(pd.to_numeric)
annotations_df['NER_tag'] = annotations_df["NER_tag"].astype(str)
entries_df[['id', 'row', 'offset']] = entries_df[['id', 'row', 'offset']].apply(pd.to_numeric)
entries_df["word"] = entries_df["word"].astype(str)

result_df = pd.merge(entries_df, annotations_df, how="left", on=['id', 'row', 'offset'])

# Check for NaNs (should be only in NER_tag, where NaNs will be replaced with "O" (outside))
print("Columns with missing data:\n", result_df.isna().any())

result_df = result_df.fillna("O")
result_df = result_df.drop(columns=["id", "row", "offset"])

print(result_df.head(50))

ner_counter = [1 for i in result_df["NER_tag"] if "B-" in i]
print(len(ner_counter), "named entities and", result_df.shape[0], "tokens")