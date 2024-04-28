import os
import json

def load_dataset(file_path):
    datas = []
    with open(file_path) as fin:
        for line in fin.readlines():
            data = json.loads(line)
            input = data['INSTRUCTION']
            output = data['RESPONSE']
            datas.append({
                "input": input,
                "output": output
            })
    return datas

if __name__ == "__main__":
    base_path = "./dataset"
    
    for lang in ["en", "ru", "pt", "nl", "it", "fr", "es", "de"]:
        for mode in ["train", "val", "test"]:
            file_path = os.path.join(base_path, lang, f'{lang}_{mode}.jsonl')
            dataset = load_dataset(file_path)

