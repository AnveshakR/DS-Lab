# Perform perfect reconstruction, partial reconstruction, rotation, scaling and
# shifting on a user defined image after changing the basis.

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import cmath
img = cv2.imread(r"/home/anveshak/Downloads/star.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (100, 100))

#coordinates of black points
coords = np.column_stack(np.where(img == 0))
def plotter(coords,coords_type):
    x = []
    y = []
    if(coords_type == "real"):
        for i in coords:
            x.append(i[0])
            y.append(i[1])
    elif(coords_type == "complex"):
        for i in coords:
            x.append(i.real)
            y.append(i.imag)
    plt.scatter(x,y)
    plt.show()
    return

#converting into complex coordinates
complex_coords = []
for i in coords:
    complex_coords.append(complex(i[0],i[1]))

N = len(complex_coords)

print("number of coords = ", N)

plotter(complex_coords,"complex")


def perfect_reconstruction(cords):
    complex_exp = []
    A_u = 0
    #changing basis
    for u in range(0,N):
        print(u)
        A_u = 0
        for k in range(0,N):
            A_u = A_u + np.vdot(cords[k], cmath.exp((complex(0, -(2*math.pi*k*u)/N))))
        print()
        complex_exp.append(A_u)

    a_k = 0
    perfect_rec = []

    #perfect reconstruction
    for k in range(0,N):
        a_k = 0
        for u in range(0,N):
            a_k = a_k + (1/N)*(np.vdot(complex_exp[u], (cmath.exp((complex(0, (2*math.pi*k*u)/N))))))
        perfect_rec.append(a_k)

    plotter(perfect_rec,"complex")
    return


#partial reconstruction
def partial_reconstruction(p,cords):

    m = int((p*N)/100)
    complex_exp = []
    #changing basis
    for u in range(0,N):
        A_u = 0
        for k in range(0,N):
            A_u = A_u + np.vdot(cords[k], cmath.exp((complex(0, -(2*math.pi*k*u)/N))))
        complex_exp.append(A_u)

    partial_rec = []
    a_k = 0
    
    for k in range(m):
        a_k = 0
        for u in range(m):
            a_k += (1/m)*(np.vdot(complex_exp[u], cmath.exp(complex(0, (2*math.pi*k*u)/m))))
        partial_rec.append(a_k)

    print(p, "% partial reconstruction:")
    plotter(partial_rec, "complex")
    return


def rotation(deg,cords):
    x = []
    y = []
    for i in cords:
        x.append(i.real)
        y.append(i.imag)
    
    angle = math.radians(deg)
    x_rot = []
    y_rot = []
    ox,oy = (0,0)

    for i in range(len(x)):
        p = ox + math.cos(angle) * (x[i] - ox) - math.sin(angle) * (y[i] - oy)
        q = oy + math.sin(angle) * (x[i] - ox) + math.cos(angle) * (y[i] - oy)
        x_rot.append(p)
        y_rot.append(q)
        p,q=0,0
    
    rotated = []
    for i in range(len(x_rot)):
        rotated.append(complex(x_rot[i],y_rot[i]))
    
    print(deg, "° rotation:")
    perfect_reconstruction(rotated)
    return


def shifting(ch, a, cords):
    shifted = []
    for i in cords:
        if ch == 1:
            shifted.append(complex(i[0]+a, i[1]))
        elif ch == 2:
            shifted.append(complex(i[0], i[1] + a))
    if ch==1:
        ch='x'
    elif ch==2:
        ch='y'
    
    print("Shifting by ",a," units along ",ch," axis:")
    perfect_reconstruction(shifted)
    return


def scaling(n, cords):
    scaled = []
    for i in cords:
        scaled.append(complex(i[0]*n, i[1]*n))

    print("Scaling by a factor of ",n)
    perfect_reconstruction(scaled)
    return


perfect_reconstruction(complex_coords)


m = int(input("enter percentage of partial reconstruction: "))
partial_reconstruction(m, complex_coords)


r = int(input("enter rotation value: "))
rotation(r, complex_coords)


ch = int(input("1. shift along x axis \n2. shift along y axis\n"))
sh = int(input("enter value to shift by: "))
shifting(ch,sh,coords)


sc = int(input("enter value to scale by: "))
scaling(sc,coords)

print("On changing the point of beginning, we get the following results")

coords_flip = np.flip(np.column_stack(np.where(img == 0)), axis=1)
#converting into complex coordinates
complex_coords_flip = []
for i in coords_flip:
    complex_coords_flip.append(complex(i[0],i[1]))

N = len(complex_coords_flip)

print("number of coords = ", N)
plotter(complex_coords,"complex")