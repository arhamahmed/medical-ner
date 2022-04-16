# inspired by https://github.com/noc-lab/clinical_concept_extraction under:

# MIT License

# Copyright (c) 2019 Henghui Zhu

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# -----------------------------------------------------------------------------

import os
import re
import pickle

def parse_dir(base_path):
    base_txt_path = base_path + 'txt/'
    base_con_path = base_path + 'concept/'

    all_txt_files = os.listdir(base_txt_path)
    all_txt_files = [item for item in all_txt_files if item[-3:] == 'txt']
    all_txt_files.sort()

    all_tokens = []
    all_concepts = []

    for txt_filename in all_txt_files:
        # read text file
        text = open(base_txt_path + txt_filename, 'r', encoding='utf-8-sig').read()
        token_list = [re.split('\ +', sentence) for sentence in text.split('\n')]
        token_list = [sentence for sentence in token_list if len(sentence) > 0]

        # read concept file
        concepts = open(base_con_path + txt_filename[:-3] + 'con', 'r', encoding='utf-8-sig').read()
        concepts = concepts.split('\n')
        concepts = [concept_item for concept_item in concepts if len(concept_item) > 1]

        # build annotation
        concepts_list = [[''] * len(sentence) for sentence in token_list]

        for concept_item in concepts:
            concept_name = re.findall(r'c="(.*?)" \d', concept_item)[0]
            concept_tag = re.findall(r't="(.*?)"$', concept_item)[0]

            concept_span_string = re.findall(r'(\d+:\d+\ \d+:\d+)', concept_item)[0]

            span_1, span_2 = concept_span_string.split(' ')
            line1, start = span_1.split(':')
            line2, end = span_2.split(':')

            assert line1 == line2

            line1, start, end = int(line1), int(start), int(end)

            concept_name = re.sub(r'\ +', ' ', concept_name)
            original_text = ' '.join(token_list[line1 - 1][start:end + 1])

            if concept_name != original_text.lower():
                print(concept_name, original_text)
                raise RuntimeError

            first = True
            for start_id in range(start, end + 1):
                if first:
                    concepts_list[line1 - 1][start_id] = 'B-' + concept_tag
                    first = False
                else:
                    concepts_list[line1 - 1][start_id] = concept_tag

        all_tokens += (token_list)
        all_concepts += (concepts_list)

    return all_tokens, all_concepts


def main():
    save_dir = './data/preprocessed/pkl/'

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    training_set_one_path = './data/training_data/set_one/'
    training_set_two_path = './data/training_data/set_two/'
    test_path = './data/test_data/'

    all_tokens, all_concepts = parse_dir(training_set_one_path)
    pickle.dump([all_tokens, all_concepts], open(save_dir + 'beth.pkl', 'wb'))

    all_tokens, all_concepts = parse_dir(training_set_two_path)
    pickle.dump([all_tokens, all_concepts], open(save_dir + 'partners.pkl', 'wb'))

    all_tokens, all_concepts = parse_dir(test_path)
    pickle.dump([all_tokens, all_concepts], open(save_dir + 'text.pkl', 'wb'))

if __name__ == '__main__':
    main()