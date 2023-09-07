import nori2 as nori

# convert a tarfile into a nori dataset
nw = nori.open("/home/lijun07/code/LAVIS/lavis/datasets/demo.nori", "w")    # open an empty nori dataset for writing
# if nori is on oss
#
# nw = nori.open("s3://mybucket/demo.nori","w")
# 

#tar = tarfile.open("./demo.tar")      # path of a demo tarfile
# if tar is on OSS
#
# file = nori.utils.smart_open("s3://mybucket/demo.tar","r")
# tar = tarfile.open(fileobj=file)
#
with open('/home/lijun07/code/LAVIS/docs/_static/football.png', 'rb') as f:
    file_data = f.read()
    nw.put(file_data, filename='/home/lijun07/code/LAVIS/docs/_static/football.png')


#tar.close()
# file.close() when use fileobj in tarfile.open(), fileobj will not be closed 
nw.close()