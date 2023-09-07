import sys
 
sys.path.append('./')
 
import nori2 as nori
from nori2 import smart_open
import boto3
import tarfile
import io

nori_name = 'CC12M'
 
nori_path = f's3://lijun//nori_datasets/{nori_name}.nori'
 
nw = nori.open(nori_path, 'w')  # 同local的使用方法
 

file = nori.utils.smart_open("s3://research-model-hh-b/Dataset/CC12M/00000.tar","r")
data = io.BytesIO(file.read())
tar = tarfile.open(fileobj=data, mode="r|*", encoding="utf-8")
for tarinfo in tar:
    if tarinfo.isfile():              # folders or symlinks should be skipped
        f = tar.extractfile(tarinfo)  # extractfile returns a file object
        filedata = f.read()
 
        # data_id is the new nori DataID of this file
        # use this data_id for your own purpose
        data_id = nw.put(filedata, filename=tarinfo.name)

# csv_file = csv.reader(smart_open("s3://xxx.csv", "r"))  # smart_open作用等同于open
#
# for cnt, line in enumerate(csv_file):
#     obj = s3_client.get_object(Bucket="zsdata",
#                                Key=img_path)  # img_path为bucket的名字以外的数据路径，例如oss上我要访问的是s3://zsdata/imgs/demo.jpg，则此处img_path为imgs/demo.jpg
#     feat_buffer = ''
#     nw.put(img, filename=img_path)
#
# nw.close()
 
print(f'Finish saving features to nori: {nori_path}')