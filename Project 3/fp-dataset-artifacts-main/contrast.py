# contrast.py

import os
import json
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='contrast.py')
    parser.add_argument('--file', type=str, default='./LIT_SNLI/original+basic/test.tsv', help='file to use to create contrast file.')
    parser.add_argument('--output', type=str, default='contrast_set.json', help='output file to write contrast set.')
    parser.add_argument('--output_dir', type=str, default='./contrast_dir/', help='output directory to write contrast set.')
    args = parser.parse_args()
    return args

def label_index(label: str):
    # 0 = entailment, 1 = neutral, 2 = contradiction
    if label == 'entailment': return 0
    if label == 'neutral': return 1
    if label == 'contradiction': return 2

def main():
    args = parse_args()
    print(args)
    
    # read TSV file into DataFrame
    df = pd.read_table(args.file)
    print(df)
    
    # write to json file
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, args.output), encoding='utf-8', mode='w') as f:
        for index, row in df.iterrows():
            # ['premise', 'hypothesis', 'label']
            # print (index, ' id=', row['captionID'], ' s1=', row['sentence1'], ' s2=', row['sentence2'], ' gold=', row['gold_label'])
            if row['captionID'] != 'original':
                f.write(json.dumps({'premise':row['sentence1'], 'hypothesis':row['sentence2'], 'label':label_index(row['gold_label'])})) 
    

if __name__ == "__main__":
    main()