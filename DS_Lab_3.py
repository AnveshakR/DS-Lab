import pandas
import numpy as np
from matplotlib import pyplot as plt

df = pandas.read_csv("<CSV FILE>")
df = df.dropna(subset=['Sr. No.'])
df = df.dropna(axis=1)

colors = {
    'C1':'red',
    'C2':'blue'
}

plt.figure(figsize=(7,7))
plt.scatter(df['X1'],df['X2'],c=df['Class'].apply(lambda x:colors[x]))
plt.show()

arr = df.iloc[:,1:3].to_numpy()
arr = np.transpose(arr)
cova = np.cov(arr,bias=True)
val,vec = np.linalg.eig(cova)
r1 = arr.shape[0]
c1 = arr.shape[1]
ex = np.reshape((np.sum(arr,axis=1)/c1),(r1,1))

origin = ex
vec1 = vec[:,0]
vec2 = vec[:,1]

plt.quiver(*origin, *vec1, color=['r'], scale=2)
plt.quiver(*origin, *(-vec1), color=['r'], scale=2)
plt.quiver(*origin, *vec2, color=['b'], scale=3)
plt.quiver(*origin, *(-vec2), color=['b'], scale=3)

colors = {
    'C1':'red',
    'C2':'blue'
}

plt.scatter(df['X1'],df['X2'],c=df['Class'].apply(lambda x:colors[x]))
plt.show()