import os
import sys
import math
import time

import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt

from liblibra_core import *
import util.libutil as comn
from libra_py import units as units
from libra_py import data_conv, data_stat, data_outs, data_read
import libra_py.workflows.nbra.decoherence_times as decoherence_times
import libra_py.workflows.nbra.step4 as step4
import libra_py.workflows.nbra.ann as ann

import ham


rnd = Random()

def nice_plot_data(_plt, x, y, x_label, y_label, fig_name, fig_label):

    colors = ["red", "blue", "green", "black", "orange"]

    figure = _plt.figure(num=None, figsize=(3.21, 2.41), dpi=300, edgecolor='black', frameon=True)
    _plt.title(F"{fig_label}",fontsize=9.5)
    _plt.legend(fontsize=6.75, ncol=1, loc='upper left')
    _plt.xlabel(x_label,fontsize=10)
    _plt.ylabel(y_label,fontsize=10)

    nstates = y.shape[1]
    for i in range(nstates):
        _plt.plot(x, y[:, i], color=colors[i % 5])

    _plt.tight_layout()
    _plt.savefig(fig_name, dpi=300)
    _plt.show()


def compute_spectra(nsteps, _params, prefix, wspan=2500):

    #=============== Data gen ================
    t, energy, nac = ham.compute_Hvib_timeseries(nsteps, _params)

    nsteps = t.shape[0]

    nstates = _params["nstates"]

    nice_plot_data(plt, t * units.au2fs, energy, "Time, fs", "Energy gap, a.u.", F"{prefix}/energy-t.png", "Energy")
    nice_plot_data(plt, t * units.au2fs, nac, "Time, fs", "NAC a.u.", F"{prefix}/nac-t.png", "NAC")


    #================ Gaps ====================
    for i in range(1, nstates):
        params = { "dt": 1.0, "wspan":wspan, "dw":1.0, "do_output":False, "do_center":True,
                   "acf_type":1, "data_type":0,
                   "leading_w_fraction":0.0001,  "deriv_scaling":10.0,
                   "tau":[10.0], "training_steps":list(range(0,nsteps-1)),
                   "output_files_prefix":F"{prefix}/e_{i-1}_{i}"
                 }

        ann.step1_workflow( energy[:, i] - energy[:, i-1], params, plt)


    #================= NACs =====================
    for i in range( int(nstates*(nstates-1)/2) ):

        params = { "dt": 1.0, "wspan":wspan, "dw":1.0, "do_output":False, "do_center":True,
                   "acf_type":1, "data_type":0,
                   "leading_w_fraction":0.0001,  "deriv_scaling":50.0,
                   "tau":[10.0], "training_steps":list(range(0,nsteps-1)),
                   "output_files_prefix":F"{prefix}/nac_{i}"
                 }

        ann.step1_workflow( nac[:, i], params, plt)


def compute_model(q, params, full_id):
    #model = params["model"]

    res = ham.compute_model_nbra_direct(q, params, full_id)
    return res


def set_model_params(model, w_nac,p_nac,w_en, p_en):

    res = {}
    # nacx0.2
    freq_en = w_en; ampls_en = p_en
    freq_nac = w_nac; ampls_nac = p_nac
    if model == 102:
        gap01 = {"func_type":0, "mean": 1.5, "scaling1":1.0, "scaling2":1.0,
                 "freqs": freq_en,   "ampls": ampls_en
                }
        nac01 = {"func_type":0, "mean": 0.05, "scaling1":0.2, "scaling2":0.2,
                 "freqs": freq_nac,    "ampls": ampls_nac
                }
        res = {"nstates":2, "gap_params":[gap01], "nac_params":[nac01] }
    # nacx0.1
    elif model == 101:
        gap01 = {"func_type":0, "mean": 1.5, "scaling1":1.0, "scaling2":1.0,
                 "freqs": freq_en,   "ampls": ampls_en
                }
        nac01 = {"func_type":0, "mean": 0.05, "scaling1":0.1, "scaling2":0.1,
                 "freqs": freq_nac,    "ampls": ampls_nac
                }
        res = {"nstates":2, "gap_params":[gap01], "nac_params":[nac01] }

    # nacx0.05
    elif model == 1005:
        gap01 = {"func_type":0, "mean": 1.5, "scaling1":1.0, "scaling2":1.0,
                 "freqs": freq_en,   "ampls": ampls_en
                }
        nac01 = {"func_type":0, "mean": 0.05, "scaling1":0.05, "scaling2":0.05,
                 "freqs": freq_nac,    "ampls": ampls_nac
                }
        res = {"nstates":2, "gap_params":[gap01], "nac_params":[nac01] }


    return res


def main(what, model_name, model, nsteps, period, init_state, wspan, w_nac,p_nac,w_en, p_en):

    if what in [0, 2] :
        os.system(F"mkdir res_{model_name}_{model}_{nsteps}_{period}")

    nthreads = 8
    methods = {0:"FSSH", 1:"IDA", 2:"mSDM", 3:"DISH", 21:"mSDM2", 31:"DISH2" }
    init_states = init_state
    tsh_methods = [0]
    batches = list(range(10))

    #================== SET UP THE DYNAMICS AND DISTRIBUTED COMPUTING SCHEME  ===============
    rnd = Random()

    rates = None
    gaps = None

    nstates = set_model_params(model, w_nac,p_nac,w_en, p_en)["nstates"]

    params_common = { "nsteps":nsteps, "dt":1.0*units.fs2au,
                      "ntraj":250, "x0":[-4.0], "p0":[4.0], "masses":[2000.0], "k":[0.01],
                      "nstates":nstates, "istate":[1, 0],
                      "which_adi_states":range(2), "which_dia_states":range(2),
                      "rep_ham":1, "tsh_method":0, "force_method":0, "nac_update_method":0,
                      "hop_acceptance_algo":31, "momenta_rescaling_algo":0,
                      "time_overlap_method":1, "mem_output_level":-1,  "txt_output_level":3,
                      "properties_to_save": ['timestep', 'time', 'SH_pop', 'SH_pop_raw', 'hvib_adi'],
                      "state_tracking_algo":0, "convergence":0,  "max_number_attempts":100,
                      "min_probability_reordering":0.01, "decoherence_algo":0, "Temperature": 300.0
                    }

    #=========================== DIRECT ==============================

    model_params_direct = set_model_params(model, w_nac,p_nac,w_en, p_en)
    model_params_direct.update({ "filename":None, "istep":0, "period":period, "dt":1.0*units.fs2au, "model":model } )
    print(model_params_direct)

    dyn_params = dict(params_common)
    dyn_params.update({ "dir_prefix":F"res_{model_name}_{model}_{nsteps}_{period}" })


    if what in [0, 2]:
        compute_spectra(period, model_params_direct, F"res_{model_name}_{model}_{nsteps}_{period}", wspan)

        step4.namd_workflow(dyn_params, compute_model, model_params_direct, rnd, nthreads,
                            methods, init_states, tsh_methods, batches, "fork", True)

    if what in [1, 2]:

           step4.nice_plots(dyn_params, init_states, tsh_methods, methods, batches, fig_label="Direct NA-MD")


freq_file="freqs_all.txt"
f1=open(freq_file, "r")
fr1=f1.readlines()
f1.close()

ampl_file = "ampl_all.txt"
f2=open(ampl_file,"r")
fr2=f2.readlines()
f2.close()

num_model = 46

what = 2
wspan = 3000

for indx in range(num_model):
    model_name = "model"+str(indx+1)
    w_nac=[float(x) for x in fr1[indx*2].strip().split(",")]
    p_nac=[float(x) for x in fr2[indx*2].strip().split(",")]
    w_en=[float(x) for x in fr1[indx*2+1].strip().split(",")]
    p_en=[float(x) for x in fr2[indx*2+1].strip().split(",")]

    main(what, model_name, 102, 100000, 1000, [1], wspan, w_nac,p_nac,w_en, p_en)
    main(what, model_name, 102, 100000, 5000, [1], wspan, w_nac,p_nac,w_en, p_en)
    main(what, model_name, 102, 100000, 10000, [1],wspan, w_nac,p_nac,w_en, p_en)
    main(what, model_name, 102, 100000, 20000, [1], wspan, w_nac,p_nac,w_en, p_en)
    main(what, model_name, 102, 100000, 50000, [1], wspan, w_nac,p_nac,w_en, p_en)
    main(what, model_name, 102, 100000, 100000, [1], wspan, w_nac,p_nac,w_en, p_en)

    main(what, model_name, 101, 100000, 1000, [1], wspan, w_nac,p_nac,w_en, p_en)
    main(what, model_name, 101, 100000, 5000, [1], wspan, w_nac,p_nac,w_en, p_en)
    main(what, model_name, 101, 100000, 10000, [1],wspan, w_nac,p_nac,w_en, p_en)
    main(what, model_name, 101, 100000, 20000, [1], wspan, w_nac,p_nac,w_en, p_en)
    main(what, model_name, 101, 100000, 50000, [1], wspan, w_nac,p_nac,w_en, p_en)
    main(what, model_name, 101, 100000, 100000, [1], wspan, w_nac,p_nac,w_en, p_en)

    main(what, model_name, 1005, 100000, 1000, [1], wspan, w_nac,p_nac,w_en, p_en)
    main(what, model_name, 1005, 100000, 5000, [1], wspan, w_nac,p_nac,w_en, p_en)
    main(what, model_name, 1005, 100000, 10000,[1],wspan, w_nac,p_nac,w_en, p_en)
    main(what, model_name, 1005, 100000, 20000,[1], wspan, w_nac,p_nac,w_en, p_en)
    main(what, model_name, 1005, 100000, 50000,[1], wspan, w_nac,p_nac,w_en, p_en)
    main(what, model_name, 1005, 100000, 100000, [1], wspan, w_nac,p_nac,w_en, p_en)
