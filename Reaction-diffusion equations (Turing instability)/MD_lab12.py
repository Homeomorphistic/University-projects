import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

###########################################################################################
##################################EQUILIBIRIUM#############################################
def FitzHughNagumo_equilibrium(a=0.1, beta=0.5):
    u = np.roots([-1, 1+a, -(a+beta), 0])
    v = lambda x: beta*x
    return np.array([u, v(u)])

def FitzHughNagumo_part_der_matrix(uv, a=0.1, beta=0.5):
    return np.array([[-3*uv[0]**2 + 2*(1+a)*uv[0] - a, -1], [beta, -1]])

def FitzHughNagumo_point_stability(uv, a=0.1, beta=0.5):
    A = FitzHughNagumo_part_der_matrix(uv, a=a, beta=beta)
    return (np.linalg.det(A) > 0 ) and (np.trace(A) < 0)

def FitzHughNagumo_stability(tau=0.01, c=0.2, gamma=0.5):
    eq_points = FitzHughNagumo_equilibrium(c=c, gamma=gamma)
    stab = np.apply_along_axis(FitzHughNagumo_point_stability, axis=0, arr=eq_points, tau=tau, gamma=gamma)
    return np.append(eq_points, stab.reshape(1, 3), axis=0)

eq_points = FitzHughNagumo_equilibrium()
print(FitzHughNagumo_point_stability(eq_points[:,1]))


##########################################################################################################
######################################NUMERICAL BOUNDARIES################################################
def dirichlet_boundary(X1, X2, Y, boundaries = [None, None, None, None]):
    if (boundaries[0] is not None):
        Y[:, -1, :] = boundaries[0](X1[-1, :], X2[-1, :])
    if (boundaries[1] is not None):
        Y[:, :, -1] = boundaries[1](X1[:, -1], X2[:, -1])
    if (boundaries[2] is not None):
        Y[:, 0, :] = boundaries[2](X1[0, :], X2[0, :])
    if (boundaries[3] is not None):
        Y[:, :, 0] = boundaries[3](X1[:, 0], X2[:, 0])
    return Y

def neumann_boundary(i, t, X1, X2, Y, boundaries = [None, None, None, None], dx=0.1):
    if (boundaries[0] is not None):
        Y[i + 1, -1, 1:-1] = Y[i + 1, -2, 1:-1] + dx * boundaries[0](t[i + 1], X1[-1, :], X2[-1, :])
    if (boundaries[1] is not None):
        Y[i + 1, :, -1] = Y[i + 1, :, -2] + dx * boundaries[1](t[i + 1], X1[:, -1], X2[:, -1])
    if (boundaries[2] is not None):
        Y[i + 1, 0, 1:-1] = Y[i + 1, 1, 1:-1] + dx * boundaries[2](t[i + 1], X1[0, :], X2[0, :])
    if (boundaries[3] is not None):
        Y[i + 1, :, 0] = Y[i + 1, :, 1] + dx * boundaries[3](t[i + 1], X1[:, 0], X2[:, 0])
    return Y

##########################################################################################################
######################################NUMERICAL SOLUTIONS#################################################
def FitzHughNagumo_2dim_solution(u_0, v_0, tau=0.01, c=0.2, d=10**(-5), gamma=0.5, T_max=1, x1 = [0,1], x2 = [0,1], dt=0.01, dx=0.1,
                                 u_dir=[None, None, None, None], u_neu=[None, None, None, None], v_dir=[None, None, None, None], v_neu=[None, None, None, None]):
    f = lambda u,v: u*(1-u)*(u-0.1) -v #(u - u**3 / 3 - v) / tau #
    g = lambda u,v: 0.01*(0.5*u-v) #-u -c -gamma*v #
    D = d/tau
    x1 = np.arange(x1[0], x1[1] + dx, step=dx)
    x2 = np.arange(x2[0], x2[1] + dx, step=dx)
    t = np.arange(0, T_max + dt, step=dt)
    U = np.zeros((t.size, x1.size, x2.size))
    V = np.zeros((t.size, x1.size, x2.size))
    X1, X2 = np.meshgrid(x1, x2)

    #Starting conditions
    U[0, :, :] = u_0(X1, X2)
    V[0, :, :] = v_0(X1, X2)

    #Dirichlet boundaries
    U = dirichlet_boundary(X1=X1, X2=X2, Y=U, boundaries=u_dir)
    V = dirichlet_boundary(X1=X1, X2=X2, Y=V, boundaries=v_dir)

    for i in range(0, t.size - 1):
        #Heat part
        U[i + 1, 1:-1, 1:-1] = U[i, 1:-1, 1:-1] + D * (dt / dx ** 2) * (U[i, :-2, 1:-1] + U[i, 2:, 1:-1] + U[i, 1:-1, :-2] + U[i, 1:-1, 2:] - 4 * U[i, 1:-1, 1:-1]) + dt * f(U[i, 1:-1, 1:-1], V[i, 1:-1, 1:-1])

        #Neumann boundaries
        U = neumann_boundary(i=i, t=t, X1=X1, X2=X2, Y=U, boundaries=u_neu)
        #V = neumann_boundary(i=i, t=t, X1=X1, X2=X2, Y=V, boundaries=v_neu)

        #Non-linear part
        V[i + 1, :, :] = V[i, :, :] + dt * g(U[i, :, :], V[i, :, :])

    return t, X1, X2, U, V

def u_0(X1, X2):
    n = X1.shape[0]
    k = X1.shape[1]
    U = np.zeros((n,k))
    U[:int(n/2), :int(k/2)] = 1#(1/np.sqrt(2)) # III quarter
    U[:int(n / 2), int(k / 2):] = 0#-(1/np.sqrt(2)) # II quarter
    U[int(k / 2):, :int(k / 2)] = 0#-(1/np.sqrt(2)) # IV quarter
    U[int(k / 2):, int(k / 2):] = 0#-(1/np.sqrt(2)) # I quarter
    return U

def v_0(X1, X2):
    n = X1.shape[0]
    k = X1.shape[1]
    V = np.zeros((n,k))
    V[:int(n/2), :int(k/2)] = 0#-(1/np.sqrt(2)) # III quarter
    V[:int(n / 2), int(k / 2):] = 0#-(1/np.sqrt(2)) # II quarter
    V[int(k / 2):, :int(k / 2)] = 0.1#0 # IV quarter
    V[int(k / 2):, int(k / 2):] = 0.1#0 # I quarter
    return V

barrier = lambda *args: 0

##########################################################################################################
########################################ANIMATIONS########################################################
def animation(t, X1, X2, U, dt=0.01, dx=0.1, dx_skip=2, dt_skip=10, name='U'):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.view_init(50, 35)
    ind = np.arange(X1.shape[1], step=dx_skip)
    def animate(t):
        ax.collections.clear()
        ax.plot_surface(X1[ind, :][:, ind], X2[ind, :][:, ind], U[:, ind, :][t, :, ind], cmap='magma', edgecolor='none')
        ax.set_title('Solution '+name+'(x,y,t) for t='+str(round(t*dt, 2)))
        ax.set_xlabel('y')
        ax.set_ylabel('x')
        ax.set_zlabel(name+'(x,y,t) ')
        return ax

    anim = FuncAnimation(fig, animate, frames=np.arange(t.size, step=dt_skip), interval=10)
    anim.save('MD_proj3_sol_FHN_'+name+'.gif', dpi=80, fps=60)
    #plt.show()

#t, X1, X2, U, V = FitzHughNagumo_2dim_solution(u_0=u_0, v_0=v_0, dx=0.01, dt=0.01, T_max=10, tau=1,u_neu=[barrier, barrier, barrier, barrier])
#animation(t, X1, X2, U, dx=0.01, dt=0.01, dt_skip=500, dx_skip=2, name='u')
#animation(t, X1, X2, V, dx=0.01, dt=0.01, dt_skip=500, dx_skip=2, name='v')

##########################################################################################################
########################################BIN########################################################
'''
def FitzHughNagumo_equilibrium(c=0.2, gamma=0.5):
    u = np.roots([gamma/3, 0, -(1+gamma), -c])
    v = lambda x: x - x**3/3
    return np.array([u, v(u)])

def FitzHughNagumo_part_dev_matrix(uv, tau=0.01, gamma=0.5):
    return np.array([[(1-uv[0]**2)/tau, -1/tau], [-1, -gamma]])

def FitzHughNagumo_point_stability(uv, tau=0.01, gamma=0.5):
    A = FitzHughNagumo_part_dev_matrix(uv, tau=tau, gamma=gamma)
    return (np.linalg.det(A) > 0 ) and (np.trace(A) < 0)

def FitzHughNagumo_stability(tau=0.01, c=0.2, gamma=0.5):
    eq_points = FitzHughNagumo_equilibrium(c=c, gamma=gamma)
    stab = np.apply_along_axis(FitzHughNagumo_point_stability, axis=0, arr=eq_points, tau=tau, gamma=gamma)
    return np.append(eq_points, stab.reshape(1, 3), axis=0)
'''