import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image as im


 
def perfect_reconstruction(x):
    cov = np.cov(x,bias=True)
    val,vec = np.linalg.eig(cov)
    idx = val.argsort()[::-1]
    val = val[idx]
    vec = vec[:,idx]
    y = np.dot(vec,x)
    perf = np.dot(np.linalg.inv(vec),y)
    return perf


 
def partial_reconstruction(x,p):
    cov = np.cov(x,bias=True)
    val,vec = np.linalg.eig(cov)
    idx = val.argsort()[::-1]
    val = val[idx]
    vec = vec[:,idx]
    vec = np.linalg.inv(vec)
    n = round((p*np.shape(vec)[0])/100)
    vec1 = vec[0:n]
    y = np.dot(vec1,x)
    part = np.round(np.dot(np.transpose(vec1),y),decimals=3)
    sq = np.square(x-part)
    rowerror = np.sum(sq,axis=1)/sq.shape[1]
    error = np.sum(rowerror,axis=0)
    return part,error


 
arr = []
row = []
r = int(input("number of rows: "))
c = int(input("number of columns: "))
for i in range(r):
    row = list(map(int,input("\nEnter the values for row", (i+1), ": ").strip().split()))[:c]
    arr.append(row)
arr = np.array(arr)
print("input array:")
print(arr,"\n\n")
p = int(input("Percentage of partial recontsruction: "))
op = partial_reconstruction(arr,p)
print("\n",p,"% partial reconstruction:\n",op[0])
print("error=",op[1])


img_path = r"/home/anveshak/Downloads/r6.jpg"

img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
grayscale = im.fromarray(img)
if grayscale.mode != 'RGB':
        grayscale = grayscale.convert('RGB')


perf_rec = im.fromarray(perfect_reconstruction(img))
if perf_rec.mode != 'RGB':
    perf_rec = perf_rec.convert('RGB')


p = int(input("\n\nPercentage of partial reconstruction for given image: "))
part_rec = im.fromarray(partial_reconstruction(img,p)[0])
if part_rec.mode != 'RGB':
    part_rec = part_rec.convert('RGB')


grayscale.save("grayscale.png")
perf_rec.save("perfect reconstruction.png")
part_rec.save("{}% partial reconstruction.png".format(p))


 
import glob
from os import remove


font = cv2.FONT_HERSHEY_SIMPLEX
org = (100,100)
fontscale = 1
color = (0,0,255)
thickness = 1
img_array = []
i=0


img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
for i in range(0,101,10):
    img1 = partial_reconstruction(img,i)[0]
    data = im.fromarray(img1)
    if data.mode != 'RGB':
        data = data.convert('RGB')
    data.save(r"temp\{}.jpg".format(i))


i=0
for filename in glob.glob(r'temp\*.jpg'):
    img = cv2.imread(filename)
    img = cv2.putText(img,str(i),org,font,fontscale,color,thickness,cv2.LINE_AA)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
    i+=10
    remove(filename)
 
 
out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc(*'DIVX'), 1, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
