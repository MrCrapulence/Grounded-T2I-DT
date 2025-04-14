import pandas as pd
from collections import OrderedDict
import os
import json

tsv_path='/root/autodl-tmp/dataset/ITM/coco/train_hard_neg_t20_i4OpenClip_ViT-H-14.tsv'
data = pd.read_csv(tsv_path, delimiter='\t')
imgname2txtid = OrderedDict()
for index, row in data.iterrows():
    image_name =  file_name = os.path.basename(row['filepath'])
    if image_name not in imgname2txtid:
        imgname2txtid[image_name] = [index]
    else:
        imgname2txtid[image_name].append(index)

output_dir='/root/autodl-tmp/dataset/ITM/coco/train_hard_neg_t20_i4OpenClip_ViT-H-14'
os.makedirs(output_dir, exist_ok=True)

output_file = os.path.join(output_dir, 'imgname2txtid.json')

# 先删
if(os.path.exists(output_file)):
    os.remove(output_file)

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(imgname2txtid, f, ensure_ascii=False, indent=2, sort_keys=False)

# 3. 保存每行数据为单独的JSON文件
for index, row in data.iterrows():
    json_file_path = os.path.join(output_dir, f'{index}.json')
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(row.to_dict(), f, ensure_ascii=False, indent=2)    