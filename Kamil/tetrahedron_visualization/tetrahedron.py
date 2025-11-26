import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colormaps
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy.ma as ma

# -------------------------------------------------
#  Tetrahedron vertex coordinates
# -------------------------------------------------
A = np.array([0.0, 0.0, 0.0])
B = np.array([1.0, 0.0, 0.0])
C = np.array([0.5, np.sqrt(3)/2, 0.0])
D = np.array([0.5, np.sqrt(3)/6, np.sqrt(6)/3])


# -------------------------------------------------
#  Function that maps barycentric weights → value
# -------------------------------------------------
def func_value(a,b,c,d):
    if a+b ==0:
        return np.nan
    val = (a+d) / (a+b)
    return val


# -------------------------------------------------
#  Convert barycentric to 3D point
# -------------------------------------------------
def f_point(a, b, c, d):
    w = np.array([a, b, c, d])
    if w.sum() == 0: 
        return np.zeros(3)

    w = w / w.sum()
    return w[0]*A + w[1]*B + w[2]*C + w[3]*D


# -------------------------------------------------
#  Barycentric grid
# -------------------------------------------------
pts = []
vals = []
N = 20   # resolution

for i in range(N+1):
    for j in range(N+1-i):
        for k in range(N+1-i-j):
            l = N - i - j - k
            w = np.array([i,j,k,l]) / N
            pts.append(f_point(*w))
            vals.append(func_value(*w))

pts = np.array(pts)
vals = np.array(vals)
# vals = ma.masked_invalid(vals)

# -------------------------------------------------
#  Plot
# -------------------------------------------------
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')

# edges
edges = [(A,B),(A,C),(A,D),(B,C),(B,D),(C,D)]
for P,Q in edges:
    ax.plot([P[0],Q[0]],[P[1],Q[1]],[P[2],Q[2]], color='black')

cmap = colormaps["viridis"].copy()
cmap.set_bad(color="red")
print("invalid count:", np.isnan(vals).sum())
# gradient scatter
sc = ax.scatter(
    pts[:,0], pts[:,1], pts[:,2],
    s=25,
    c=vals,
    cmap=cmap
)

ax.text(*A, "A", fontsize=16, weight='bold')
ax.text(*B, "B", fontsize=16, weight='bold')
ax.text(*C, "C", fontsize=16, weight='bold')
ax.text(*D, "D", fontsize=16, weight='bold')

fig.colorbar(sc, ax=ax, shrink=0.6)

ax.set_box_aspect([1,1,1])
plt.show()

