import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#Helpful functions
def ab_indicator(a=0, b=1, const=1):
    return lambda x: const * (a<=x) * (x<=b)

def f_const(c):
    return lambda *args: c

###########################################################
################SOLVING HEAT EQUATION######################

#Rod/Interval (1-dimension)
def solve_heat_1dim(u_0, u_0t=None, u_1t=None, u_x_0t=None, u_x_1t=None, D=0.01, T_max=10, x_min=0, x_max=1, dt=0.01, dx=0.1):
    x = np.arange(x_min, x_max+dx, step=dx)
    t = np.arange(0, T_max + dt, step=dt)
    U = np.zeros((t.size, x.size))
    U[0, :] = u_0(x)
    if(u_0t is not None):
        U[:, 0] = u_0t
    if(u_1t is not None):
        U[:, -1] = u_1t

    for i in range(0, t.size-1):
        for j in range(1, x.size-1):
            U[i+1, j] = U[i, j] + D * (dt/dx**2) * ( U[i, j-1] + U[i, j+1] - 2* U[i, j] )

        if(u_x_0t is not None):
            U[i+1, 0] = U[i+1, 1] + dx*u_x_0t
        if(u_x_1t is not None):
            U[i+1, -1] = U[i+1, -2] + dx*u_x_1t
    return t, x, U

#Plate/Square (2-dimensions)
def solve_heat_2dim(u_0, u_1=None, u_2=None,  u_3=None, u_4=None, u_x_1=None, u_x_2=None, u_x_3=None, u_x_4=None, D=0.01, T_max=10, x1_min=0, x1_max=1, x2_min=0, x2_max=1, dt=0.01, dx=0.1):
    x1 = np.arange(x1_min, x1_max+dx, step=dx)
    x2 = np.arange(x2_min, x2_max+dx, step=dx)
    t = np.arange(0, T_max + dt, step=dt)
    U = np.zeros((t.size, x1.size, x2.size))
    X1, X2 = np.meshgrid(x1, x2)
    U[0, :, :] = u_0(X1, X2)

    if(u_1 is not None):
        U[:, -1, :] = u_1(X1[-1,:], X2[-1, :])
    if(u_2 is not None):
        U[:, :, -1] = u_2(X1[:, -1], X2[:, -1])
    if(u_3 is not None):
        U[:, 0, :] = u_3(X1[0, :], X2[0, :])
    if(u_4 is not None):
        U[:, :, 0] = u_4(X1[:, 0], X2[:, 0])
    
    for i in range(0, t.size-1):
        #for j in range(1, x1.size-1):
            #for k in range(1, x2.size-1):
                #U[i+1, j, k] = U[i, j, k] + D * (dt/dx**2) * ( U[i, j-1, k] + U[i, j+1, k] + U[i, j, k-1] + U[i, j, k+1] - 4* U[i, j, k] )
        U[i+1, 1:-1, 1:-1] = U[i, 1:-1, 1:-1] + D * (dt/dx**2) * ( U[i, :-2, 1:-1] + U[i, 2:, 1:-1] + U[i, 1:-1, :-2] + U[i, 1:-1, 2:] - 4* U[i, 1:-1, 1:-1] )

        if(u_x_1 is not None):
            U[i+1, -1, 1:-1] = U[i+1, -2, 1:-1] +  dx*u_x_1(t[i+1], X1[-1,:], X2[-1, :])
        if(u_x_2 is not None):
            U[i+1, 1:-1, -1] = U[i+1, 1:-1, -2] +  dx*u_x_2(t[i+1], X1[:, -1], X2[:, -1])
        if(u_x_3 is not None):
            U[i+1, 0, 1:-1] = U[i+1, 1, 1:-1] +  dx*u_x_3(t[i+1], X1[0, :], X2[0, :])
        if(u_x_4 is not None):
            U[i+1, 1:-1, 0] = U[i+1, 1:-1, 1] +  dx*u_x_4(t[i+1], X1[:, 0], X2[:, 0])
            
    return X1, X2, U

##############################################################
########################PLOTS#################################

#Rod/Interval [0,1], T=1 (1-dimension)
'''
u_0 = lambda x: np.sin(3*np.pi*x)

t, x, U = solve_heat_1dim(u_0=u_0, u_0t=-1, u_x_1t=1, dt=0.001, dx=0.01, D=0.05, T_max=1) #D=0.055
X, T = np.meshgrid(x, t)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, T, U, cmap='magma', edgecolor='none')
ax.set_title('Solution u(x,t), d=0.055')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u(x,t)')
ax.view_init(40, -130)
plt.show()
'''

#Using plotly
'''
fig = go.Figure(go.Surface(z=U))
fig.show()
'''

#Plate/Square [0,1]^2, T=1, (2-dimensions)
'''
X1, X2, U = solve_heat_2dim(u_0=u_0, u_1=f_const(-1), u_3=f_const(1), u_x_2=f_const(1), u_x_4=f_const(1), dt=0.001, dx=0.01, T_max=1)
#3d plots
w = go.Surface(z=U[0,:,:])
fig = go.Figure(w)
fig.show()
w = go.Surface(z=U[200,:,:])
fig = go.Figure(w)
fig.show()
w = go.Surface(z=U[500,:,:])
fig = go.Figure(w)
fig.show()
w = go.Surface(z=U[1000,:,:])
fig = go.Figure(w)
fig.show()
'''

#########################################################
#####################ANIMATIONS###########################

def heat_animation(u_0, u_1=None, u_2=None,  u_3=None, u_4=None, u_x_1=None, u_x_2=None, u_x_3=None, u_x_4=None, D=0.01, T_max=10, x1_min=0, x1_max=1, x2_min=0, x2_max=1, dt=0.01, dx=0.1, dx_skip=2, dt_skip=10):
    X1, X2, U = solve_heat_2dim(u_0, u_1, u_2, u_3, u_4, u_x_1, u_x_2, u_x_3, u_x_4, D, T_max, x1_min, x1_max, x2_min, x2_max, dt, dx)
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.view_init(40, 35)#ax.view_init(40, -130)#
    ind = np.arange(int(1/dx), step=dx_skip)
    def heat_animate(t):
        ax.collections.clear()
        ax.plot_surface(X1[ind, :][:, ind], X2[ind, :][:, ind], U[:, ind, :][t, :, ind], cmap='magma', edgecolor='none')
        ax.set_title('Solution u(x,y,t) for t='+str(round(t*dt, 2)))
        ax.set_xlabel('y')
        ax.set_ylabel('x')
        ax.set_zlabel('u(x,y,t) ')
        return ax

    anim = FuncAnimation(fig, heat_animate, frames=np.arange(int(1/dt), step=dt_skip), interval=10)
    anim.save('MD_proj2_sol_2dim_d=0026.gif', dpi=80, writer='Pillow', fps=60)
    plt.show()

#u_0 = lambda x1, x2: np.sin(3*np.pi*x1) * np.sin(3*np.pi*x2)
#heat_animation(u_0=u_0, u_1=f_const(-1), u_3=f_const(1), u_x_2=f_const(1), u_x_4=f_const(1), dt=0.001, dx=0.01, T_max=1, D=0.026, dt_skip=10)
#u_0 = lambda x1, x2: x1*(1-x1)*x2*(1-x2)
#heat_animation(u_0=u_0, u_1=f_const(0.1), u_3=f_const(0.1), u_x_2=f_const(-1), u_x_4=f_const(-1), dt=0.001, dx=0.01, T_max=1)