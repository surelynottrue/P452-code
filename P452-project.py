import numpy as np
from LibPython.Library import DiffEq
from matplotlib import pyplot as plt

def junction(t, y, paramlist, noisy=False, noiselevel=0.01):
    I, Ic0, Q = tuple(paramlist)
    rand = np.random.normal(scale=noiselevel)
    if Q >= 1:
        dg, g = tuple(y)
        ddg = I/Ic0 - (1/Q)*dg - np.sin(g) + noisy*rand*I
        return np.array([ddg, dg])
    else:
        g = y
        dg = Q*(I/Ic0 - np.sin(g) + noisy*rand*I)
        return np.array(dg)

start, stop = 0.75, 1.5
step = 1e-4
cycle = np.concatenate((np.arange(start, stop, step), np.flip(np.arange(start, stop, step))))

for Q in [1, 1e-2]:
    noiselist = [0.0, 0.1, 0.5]
    if Q == 1: noiselist = [0.00, 0.01, 0.05]
    for noise in noiselist:
        Ic0 = 1.0
        Q = 1e-2
        tend = numpoints = 1e3
        if Q<1: tend = 1/Q**2
        tlist = np.linspace(0, tend, int(numpoints))
        vlist = []; ilist = []
        start = [1.0, 1.0]

        for alp in cycle:
            I = Ic0*alp
            paramlist = [I, Ic0, Q]
            d = DiffEq(lambda t, y, p : junction(t, y, p, noisy=bool(noise), noiselevel=noise), tlist, list(start), paramlist)
            y = d.runge_kutta()

            start = y[-1]
            if (Q<1):
                v = np.average(junction(tlist, y[:, 0], paramlist))
            else:
                v = np.average(y[:, 0])
            vlist.append(v)
            ilist.append(I)
            try:
                if (Q<1) and ilist[-2]==ilist[-1]:
                    vlist.pop()
                    ilist.pop()
                    break
            except(IndexError):
                pass
        plt.plot(np.array(ilist)/(Ic0), np.array(vlist)/(Ic0), label=f"Noise = {noise}")
    plt.xlabel("I/Ic0")
    plt.ylabel("V/R*Ic0")
    plt.legend()
    plt.savefig(f"Q_{Q}")
    plt.clear()