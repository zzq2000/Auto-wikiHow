import os
import json
import argparse
from tqdm import tqdm
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split




def data_split(file_path, random_state, base_path, train_size, val_size, test_size):

    table = pq.read_table(file_path)
    df = table.to_pandas()

    df['lang'] = df['METADATA'].apply(lambda x: json.loads(x)['language'])

    grouped = df.groupby('lang')

    

    for lang, group in tqdm(grouped):
        train, temp = train_test_split(group, train_size=train_size, random_state=random_state)  # 80% train, 20% temp
        val, test = train_test_split(temp, train_size=val_size, random_state=random_state)  # 10% val, 10% test

        # save to dir
        os.makedirs(os.path.join(base_path, lang), exist_ok=True)
        train.to_json(os.path.join(base_path, lang, f'{lang}_train.jsonl'), orient='records', lines=True)
        val.to_json(os.path.join(base_path, lang, f'{lang}_val.jsonl'), orient='records', lines=True)
        test.to_json(os.path.join(base_path, lang, f'{lang}_test.jsonl'), orient='records', lines=True)  
        
        # average response length
        lengths = group["RESPONSE"].str.len()
        average_length = lengths.mean()
        print(f"average response length {lang}: ", average_length)

if __name__ == "__main__":

    # parse args
    parser = argparse.ArgumentParser(description="Split Dataset")
    parser.add_argument('raw_file_path', type=str, help='Path to the input dataset', default='multilingual-wikihow-qa-16k/data/train-00000-of-00001-0bdf6bc5b4b507e0.parquet')
    parser.add_argument('random_state', type=int, help='Random seed for data splitting', default = 42)
    parser.add_argument('data_base_path', type=str, help='Path to the input dataset', default='./dataset')
    parser.add_argument('train_size', type=float, default=0.8)
    parser.add_argument('val_size', type=float, default = 0.1)
    parser.add_argument('test_size', type=float, default=0.1)
    args = parser.parse_args()
    
    data_split(args.raw_file_path, args.random_state, args.data_base_path, args.train_size, args.val_size, args.test_size)