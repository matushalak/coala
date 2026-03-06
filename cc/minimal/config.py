import torch.nn as nn
from cc.utils import ThresholdReLU

# Broadly tuned: Familiar -> FB responses, Novel -> FF & FB responses
# X not seen in experimental data
broad = {
    "n_features": 2,
    "n_pv": 2,
    "n_context": 2,
    "activation": ThresholdReLU(threshold=0.2),
    "lr_ff": 0.0005,
    "lr_fb": 0.0003,
    "lr_lat": 0.0001,
    "lr_pv": 0.0002,
    "pyc_decay": 0.1,
    "pv_decay": 0.25,
    "alpha": 1.0,
    "weight_decay": 0.00025,
    "seed": 42,
}
# NOTE: simple model cannot capture broadly tuned cell adapting to multiple familiar

narrow_familiar = broad.copy()
narrow_familiar.update({
    "w_ff_init": {'mu': [0.9, 0.01], 'sigma': 1e-4},
    "w_fb_init": {'mu': [0.01, 0.01], 'sigma': 1e-4},
    "w_lat_init": {'mu': [0.3, 0.3], 'sigma': 1e-2},
    "W_pv_init": {'mu': ([0.1, 0.1], [0.1,0.1]), 'sigma': [1e-2, 1e-2]}})


narrow_novel = broad.copy()
narrow_novel.update({
    "w_ff_init": {'mu': [0.01, 0.9], 'sigma': 1e-4},
    "w_fb_init": {'mu': [0.01, 0.01], 'sigma': 1e-4},
    "w_lat_init": {'mu': [0.3, 0.3], 'sigma': 1e-2},
    "W_pv_init": {'mu': ([0.1, 0.1], [0.1,0.1]), 'sigma': [1e-2, 1e-2]}})
# NOTE cant 

# 1) nonresponder (subthreshold), only FF PV strengthening
nonresponder = broad.copy()
nonresponder.update({
    "w_ff_init": {'mu': [0.01, 0.01], 'sigma': 1e-4},
    "w_fb_init": {'mu': [0.01, 0.01], 'sigma': 1e-4},
    "w_lat_init": {'mu': [0.3, 0.3], 'sigma': 1e-2},
    "W_pv_init": {'mu': ([0.1, 0.1], [0.1,0.1]), 'sigma': [1e-2, 1e-2]},
    })
# [NOTE] Might need spiking models to capture sub-threshold behavior
# [NOTE] Because just FF inhibition, no way to prevent FB responses

# idea: to prevent runaway strengthening of FF PV synapses, condition strengthening on
# co-activation of lateral synapses onto PV and feedforward synapses onto PV


# 2) unresponsive -> FF responsive
# need different mechanism (strengthened cRF feedback ...)
# NOTE: cant reproduce
un_FF = broad.copy()
un_FF.update({
    "w_ff_init": {'mu': [1e-1, 1e-1], 'sigma': 1e-4},
    "w_fb_init": {'mu': [1e-1, 1e-1], 'sigma': 1e-4},
    "w_lat_init": {'mu': [1e-1, 1e-1], 'sigma': 1e-2},
    "W_pv_init": {'mu': ([0.2, 0.2], [0.2,0.2]), 'sigma': [1e-2, 1e-2]},
    })
# NOTE: impossible to have FF only because FB coming all the time independent of input,
# with no way to prevent strengthening

# 3) unresponsive -> FB responsive
# [NOTE] strengthened by other neurons being active, hard to capture in minimal 1-neuron model :| 
# especially because context independent of input
# unresponsive probably because sub-threshold
un_FB = broad.copy()
un_FB.update({
    "w_ff_init": {'mu': [1e-8, 1e-8], 'sigma': 1e-4},
    "w_fb_init": {'mu': [0.4, 0.4], 'sigma': 1e-4},
    "w_lat_init": {'mu': [0.8, 0.8], 'sigma': 1e-2},
    "W_pv_init": {'mu': ([0.1, 0.1], [0.1,0.1]), 'sigma': [1e-2, 1e-2]},
    })
    # what we're seeing are just FB responses; 
    # NOTE: also present in FF because not lateral inhibition :|

FF_FF = broad.copy()
FF_FF.update({
    "w_ff_init": {'mu': [1, 1], 'sigma': 1e-4},
    "w_fb_init": {'mu': [1e-7, 1e-7], 'sigma': 1e-4},
    "w_lat_init": {'mu': [1e-4, 1e-4], 'sigma': 1e-2},
    "W_pv_init": {'mu': ([1e-3, 1e-3], [1e-3,1e-3]), 'sigma': [1e-2, 1e-2]},
    })

FF_un = FF_FF.copy()
FF_un.update({
    "w_ff_init": {'mu': [1, 1], 'sigma': 1e-4},
    "w_fb_init": {'mu': [1e-7, 1e-7], 'sigma': 1e-4},
    "w_lat_init": {'mu': [0.3, 0.3], 'sigma': 1e-2},
    "W_pv_init": {'mu': ([0.2, 0.2], [0.2,0.2]), 'sigma': [1e-2, 1e-2]},
    })

# Overview
# unresponsive -> unresponsive (subthreshold only PV get stronger because just FF inhibition)
# unresponsive -> FF (different mechanism, X minimal circuit)
# unresponsive -> FB (based on strengthened FB without own firing and release from inhibition)

# FF -> FF, FB still strengthened, novel FF no adaptation; FF strengthened (diff mechanism)
# FF -> FB, yes, see basic
# FF -> unresponsive; not discussed, not supported by model (no way to prevent FB strengthening)

# FB -> x, generally not discussed
FB_x = broad.copy()
FB_x.update({
    "w_ff_init": {'mu': [1e-7, 1e-7], 'sigma': 1e-4},
    "w_fb_init": {'mu': [1, 1], 'sigma': 1e-4},
    "w_lat_init": {'mu': [1e-1, 1e-1], 'sigma': 1e-2},
    "W_pv_init": {'mu': ([1e-1, 1e-1], [1e-1,1e-1]), 'sigma': [1e-2, 1e-2]},
    })

    # seeing FB-FB, no way to get rid of FB responses during FF since no lateral inhibition


minimal_configs = {
    "broad": broad,
    "narrow_familiar": narrow_familiar,
    "narrow_novel": narrow_novel,
    "nonresponder": nonresponder,
    
    "un_FF": un_FF,
    "un_FB": un_FB,
    "FF_FF": FF_FF,
    "FF_un": FF_un,
    "FB_x": FB_x
}