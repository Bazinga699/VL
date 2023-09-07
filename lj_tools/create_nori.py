import os
import nori2 as nori
import json
from tqdm import tqdm

path = "/data/datasets/flickr30k/flickr30k_images" #文件夹目录
path2id = {}

assert os.path.isdir(path)
 
nw = nori.open("/data/nori/flickr30k.nori", "w")    # open an empty nori dataset for writing

for root, dirs, files in os.walk(path):
     #for dir in tqdm(dirs):
          #files= os.listdir(os.path.join(root, dir))
          for file in tqdm(files):
            f = open(os.path.join(root, file), "rb")
    
            filedata = f.read()

            # data_id is the new nori DataID of this file
            # use this data_id for your own purpose
            data_id = nw.put(filedata, filename=file)
            file_name = os.path.join(path.split('/')[-1], file)
            path2id[file_name] = data_id
            f.close()


# file.close() when use fileobj in tarfile.open(), fileobj will not be closed
nw.close()
with open('/data/datasets/flickr30k/annotations/path2id.json', 'w') as f:
     json.dump(path2id, f)