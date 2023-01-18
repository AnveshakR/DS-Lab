import pandas
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib as mpl
from shapely.geometry import Point
from shapely.geometry import LineString


df = pandas.read_csv('D4.csv')


df = df.dropna(subset=['Sr.'])
df = df.dropna(axis=1)


c1 = []
c2 = []
for i in range(len(df['Class'])):
    if df['Class'][i] == 'C1':
        c1.append((float(df['X1'][i]),float(df['X2'][i])))
    elif df['Class'][i] == 'C2':
        c2.append((float(df['X1'][i]),float(df['X2'][i])))
c1 = np.array(c1)
c2 = np.array(c2)


colors = {
    'C1':'red',
    'C2':'blue'
}
plt.figure(figsize=(7,7))
plt.scatter(df['X1'],df['X2'],c=df['Class'].apply(lambda x:colors[x]))
plt.xlabel('X2')
plt.ylabel('X1')
plt.show()


arr1 = df.iloc[:,1:3].to_numpy()
arr1 = np.transpose(arr1)


#x1^2, x2^2, x1*x2


x1_2 = []
x2_2 = []
x1x2 = []
for i in range(len(arr1[0])):
    x1_2.append(arr1[0][i]*arr1[0][i])
    x2_2.append(arr1[1][i]*arr1[1][i])
    x1x2.append(arr1[0][i]*arr1[1][i])


ax = plt.axes(projection = '3d')
ax.scatter3D(x1_2,x2_2,x1x2,c=df['Class'].apply(lambda x:colors[x]))


cova = np.cov(arr1,bias=True)
val,vec = np.linalg.eig(cova)
r1 = arr1.shape[0]
col1 = arr1.shape[1]
ex = np.reshape((np.sum(arr1,axis=1)/col1),(r1,1))
origin = (float(ex[0][0]),float(ex[1][0]))


vec1 = vec[:,0]
vec2 = vec[:,1]


slope1 = float(vec1[0]/vec1[1])
slope2 = float(vec2[0]/vec2[1])


yint1 = origin[1] - slope1*origin[0]
yint2 = origin[1] - slope2*origin[0]


fig,ax = plt.subplots()


ax.axline((0,yint1),slope=slope1,color='black')
ax.axline((0,yint2),slope=slope2,color='black')


colors = {
    'C1':'red',
    'C2':'blue'
}
plt.scatter(df['X1'],df['X2'],c=df['Class'].apply(lambda x:colors[x]))
plt.xlabel('X2')
plt.ylabel('X1')
plt.show()


total1 = 0
total2 = 0


fig,ax = plt.subplots()
ax.axline((0,yint1),slope=slope1,color='black')
for i in range(len(arr1[0])):
    c = df['Class'][i]
    if(c=='C1'):
        c='r'
    else:
        c='b'
    point = Point(arr1[:,i])
    line = LineString([(0, yint1), origin])


    x = np.array(point.coords[0])


    u = np.array(line.coords[0])
    v = np.array(line.coords[len(line.coords)-1])


    n = v - u
    n /= np.linalg.norm(n, 2)


    P = u + n*np.dot(x - u, n)
    total1 += P[1]-(slope1*P[0])-yint1
    plt.scatter(P[0],P[1],c=c)
plt.show()


fig,ax = plt.subplots()
ax.axline((0,yint2),slope=slope2,color='black')
for i in range(len(arr1[0])):
    c = df['Class'][i]
    if(c=='C1'):
        c='r'
    else:
        c='b'
    point = Point(arr1[:,i])
    line = LineString([(0, yint2), origin])


    x = np.array(point.coords[0])


    u = np.array(line.coords[0])
    v = np.array(line.coords[len(line.coords)-1])


    n = v - u
    n /= np.linalg.norm(n, 2)


    P = u + n*np.dot(x - u, n)
    total2 += P[1]-(slope2*P[0])-yint2
    plt.scatter(P[0],P[1],c=c)
plt.show()
print(total1,total2)
