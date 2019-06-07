import control
import control.matlab
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    Ts = 1e-3 # sampling time
    sigma_d = 0.1 # disturbance standard deviation

    # disturbance power spectrum
    wu = 1  # bandwidth of the force disturbance
    tau_u = 1 / wu
    Hu = control.TransferFunction([1], [1 / wu, 1])
    Hu = Hu * Hu
    Hud = control.matlab.c2d(Hu, Ts)
    N_sim_imp = tau_u / Ts * 20
    t_imp = np.arange(5000) * Ts
    t, y = control.impulse_response(Hud, t_imp)
    y = y[0]
    std_tmp = np.sqrt(np.sum(y ** 2))  # np.sqrt(trapz(y**2,t))
    Hu = Hu / (std_tmp) * sigma_d


    N_skip = int(20 * tau_u // Ts) # skip initial samples to get a regime sample of d
    t_sim = 300 # simulate 10 seconds of disturbance
    N_sim = int(t_sim // Ts)
    N_sim = N_sim + N_skip
    e = np.random.randn(N_sim)
    te = np.arange(N_sim) * Ts
    _, d, _ = control.forced_response(Hu, te, e)
    d = d[N_skip:]
    td = np.arange(len(d)) * Ts

    plt.plot(td, d, label='d')
    plt.plot(td, 3 * sigma_d * np.ones(np.shape(d)), 'k--', label='$3\sigma$')
    plt.plot(td, -3 * sigma_d * np.ones(np.shape(d)), 'k--', label='-$3\sigma$')
    plt.legend()
    plt.grid(True)
