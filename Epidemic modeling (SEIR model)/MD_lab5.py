import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def runge_kutta_method(y_0 = np.array([0.5,0.5]), x_min=0, x_max=50, h=0.1, f = lambda x: x, sol = np.exp):
    x = np.arange(x_min, x_max+h, step=h)
    y = np.zeros((x.size, len(y_0)))
    y[0,:] = y_0

    for k in range(x.size-1):
        k_1 = y[k,:] + h*f(y[k,:])
        k_2 = y[k, :] + h * f((y[k, :] + k_1)/2)
        k_3 = y[k, :] + h * f((y[k, :] + k_2)/2)
        k_4 = y[k, :] + h * f(k_3)
        y[k + 1, :] = (k_1 + 2*k_2 + 2*k_3 + k_4)/6

    return x,y

def quiver_plot(f, x1_range, x2_range):
    X1, X2 = np.meshgrid(np.linspace(x1_range[0], x1_range[1], num=30), np.linspace(x2_range[0], x2_range[1], num=30))
    dY = f([X1, X2])
    length = np.sqrt(dY[0]**2 + dY[1]**2)
    dY[0] /= length
    dY[1] /= length
    plt.quiver(X1, X2, dY[0], dY[1])

def phase_space_plots(Y_0, f, h=0.01, x_max=365, names=['S', 'E', 'I', 'R']): #function for models containing more than 2 variables, Y_0 -- rows are starting conditions
    RK = [None] * Y_0.shape[0]
    for i in range(Y_0.shape[0]):
        t, RK[i] = runge_kutta_method(f=f, h=h, y_0=Y_0[i, :], x_max=x_max)

    labels = [''] * Y_0.shape[0] #labels for legend
    for i in range(Y_0.shape[0]):
        for j in range(Y_0.shape[1]):
            labels[i] += names[j]+'='+str(Y_0[i,j])+', '

    if(Y_0.shape[1] == 2): #if model has only 2 variables, quiver plot is reasonable
        quiver_plot(f=f, x1_range=(RK[0][:, 0].min(), RK[0][:, 0].max()), x2_range=(RK[0][:, 1].min(), RK[0][:, 1].max()))
        for i in range(Y_0.shape[0]):
            plt.plot(RK[i][:, 0], RK[i][:, 1])
            plt.title(names[0]+' vs '+names[1], loc='left')
            plt.xlabel(names[0]+'(t)')
            plt.ylabel(names[1]+'(t)')
            plt.legend(labels=labels)
    else:
        n = math.comb(Y_0.shape[1], 2) # number of plots
        fig, ax = plt.subplots(math.ceil(n/2), 2) 
        d = l = 0
        for i in range(Y_0.shape[1]):
            for j in range(i+1, Y_0.shape[1]):
                for k in range(Y_0.shape[0]):
                    ax[d, l].plot(RK[k][:, i], RK[k][:, j])
                    ax[d, l].set_title(names[i]+' vs '+names[j], loc='left')
                    ax[d, l].set_xlabel(names[i]+'(t)')
                    ax[d, l].set_ylabel(names[j]+'(t)')
                d += l
                l = (l+1)%2

        fig.suptitle('Phase plots, t in [0,'+str(x_max)+']')
        fig.legend(labels=labels)
    plt.show()

def solution_plots(Y_0, f, h=0.01, x_max=365, names=['S', 'E', 'I', 'R']):
    RK = [None] * Y_0.shape[0]
    for i in range(Y_0.shape[0]):
        t, RK[i] = runge_kutta_method(f=f, h=h, y_0=Y_0[i, :], x_max=x_max)

    if(Y_0.shape[1]==2):
      fig, ax = plt.subplots(1, 2)
      for i in range(2):
        for j in range(Y_0.shape[0]):
          ax[i].plot(t, RK[j][:, i])
        ax[i].set_title(names[i]+'(t)')
        ax[i].set_xlabel('t')
        ax[i].set_ylabel(names[i]+'(t)')
    else:
      fig, ax = plt.subplots(math.ceil(Y_0.shape[1]/2), 2) #number of plots
      d = l = 0
      for i in range(Y_0.shape[1]):
          for j in range(Y_0.shape[0]):
              ax[d, l].plot(t, RK[j][:, i])
          ax[d, l].set_title(names[i]+'(t)')
          ax[d, l].set_xlabel('t')
          ax[d, l].set_ylabel(names[i]+'(t)')
          d += l
          l = (l+1)%2

    labels = [''] * Y_0.shape[0] #labels for legend
    for i in range(Y_0.shape[0]):
        for j in range(Y_0.shape[1]):
            labels[i] += names[j]+'='+str(Y_0[i,j])+', '

    fig.suptitle('Solutions, t in [0,'+str(x_max)+']')
    fig.legend(labels=labels)
    plt.show()

def parameters_solutions_plot(y_0, model, PARAM, h=0.01, x_max=365, p_names=['gamma', 'sigma', 'R_0'], v_names=['S', 'E', 'I', 'R']):
    RK = [None] * PARAM.shape[0]
    for i in range(PARAM.shape[0]):
        t, RK[i] = runge_kutta_method(f=model(*PARAM[i, :]), h=h, y_0=y_0, x_max=x_max)

    if (y_0.size == 2):
        fig, ax = plt.subplots(1, 2)
        for i in range(2):
            for j in range(PARAM.shape[0]):
                ax[i].plot(t, RK[j][:, i])
            ax[i].set_title(v_names[i] + '(t)')
            ax[i].set_xlabel('t')
            ax[i].set_ylabel(v_names[i] + '(t)')
    else:
        fig, ax = plt.subplots(math.ceil(y_0.size / 2), 2)  # number of plots
        d = l = 0
        for i in range(y_0.size):
            for j in range(PARAM.shape[0]):
                ax[d, l].plot(t, RK[j][:, i])
            ax[d, l].set_title(v_names[i] + '(t)')
            ax[d, l].set_xlabel('t')
            ax[d, l].set_ylabel(v_names[i] + '(t)')
            d += l
            l = (l + 1) % 2

    labels = [''] * PARAM.shape[0]  # labels for legend
    for i in range(PARAM.shape[0]):
        for j in range(PARAM.shape[1]):
            labels[i] += p_names[j] + '=' + str(round(PARAM[i, j],2)) + ', '

    fig.suptitle('Solutions, t in [0,' + str(x_max) + ']')
    fig.legend(labels=labels)
    plt.show()

#MODELS
def SEIR_model(gamma=1/18, sigma=1/5.2, R_0=2.6): #parameters for covid
    beta = R_0*gamma
    return lambda y: np.array([ -beta*y[0]*y[2]/ (y[0]+y[1]+y[2]+y[3]),
                              beta *y[0]*y[2]/(y[0]+y[1]+y[2]+y[3]) - sigma*y[1],
                              sigma*y[1] - gamma*y[2],
                                gamma*y[2]])

def SIR_model(gamma=1/18, sigma=1/5.2):
    return lambda y: np.array([ -sigma * y[0] *y[1] / (y[0] + y[1] + y[2]),
                                sigma * y[0] * y[1] / (y[0] + y[1] + y[2]) - gamma * y[1],
                                gamma * y[1]])

def LV_model(a=1, b=1, c=1, d=1):
    return lambda y: np.array([ (a-b*y[1])*y[0], (-c+d*y[0])*y[1] ])                                

def whale_model(k=6, a=1/27, d=1):
  return lambda y: np.array([ y[0]*( (k-y[0]) - y[1]/(1+y[0]) ), 
                             d * y[1] * ( y[0]/(1+y[0]) - a*y[1] )]) 

Y_0 = np.array([[11*10**4, 40*20, 40, 0],
                [11*10**4, 80*20, 80, 0],
                [11*10**4, 160*20, 160, 0],
                [11*10**4, 300*20, 300, 0]])

Y_0_whale = np.array([[1,8],
                      [8,1],
                      [6,6]
                      ])

PARAM = np.array([ [1/18, 1/5.2, 2.6],
                 [2/18, 1/5.2, 2.6],
                [1/18, 2/5.2, 2.6],
                   [1/18, 1/5.2, 2.0]])

#phase_space_plots(Y_0=Y_0, f=SEIR_model())
#solution_plots(Y_0=Y_0, f=SEIR_model(tuple(PARAM[0,:])))
parameters_solutions_plot(y_0=Y_0[1, :], PARAM=PARAM, model=SEIR_model)

#phase_space_plots(Y_0=Y_0[:, [0,2,3]], f=SIR_model(), names=['S', 'I', 'R'])
#solution_plots(Y_0=Y_0[:, [0,2,3]], f=SIR_model(), names=['S', 'I', 'R'])
#parameters_solutions_plot(y_0=Y_0[1, [0,2,3]], PARAM=PARAM[:, [0,1]], model=SIR_model, p_names=['gamma', 'sigma'], v_names=['S', 'I', 'R'])

#phase_space_plot(Y_0=np.array([[10,10],[5,5]]), f=LV_model(), names=['x', 'y'])
#parameters_solutions_plot(y_0=np.array([5, 5]), PARAM=np.array([[1,1,1,1],[1,2,3,4]]), model=LV_model, p_names=['a', 'b', 'c', 'd'], v_names=['x', 'y'], x_max=50)

#phase_space_plots(Y_0=Y_0_whale, f=whale_model(), x_max=100, names=['p','h'])
#solution_plots(Y_0=Y_0_whale, f=whale_model(), x_max=100, names=['p','h'])
