import torch.nn as nn
from cc.utils import ThresholdReLU

# Broadly tuned: Familiar -> FB responses, Novel -> FF & FB responses
# X not seen in experimental data
broad = {
    "n_features": 2,
    "n_pv": 2,
    "n_context": 2,
    "activation": ThresholdReLU(threshold=0),
    "lr_ff": 0.03,
    "lr_fb": 0.005,
    "lr_lat": 0.01,
    "lr_pv": 0.0025,
    "w_ff_init": {'mu': [0.5, 0.5], 'sigma': 0},
    "w_fb_init": {'mu': [0.05, 0.05], 'sigma': 0},
    "w_lat_init": {'mu': [0.3, 0.3], 'sigma': 0},
    "W_pv_init": {'mu': ([0.9, 0.1], [0.1,0.9]), 'sigma': [0, 0]},
    "pyc_decay": 0.05,
    "pv_decay": 0.5,
    "alpha": 1.0,
    "weight_decay": 0.00025,
    "seed": 42,
    "receives_context": (True, True),
    "FBrule": "dampened-anti-Hebbian"
}
# NOTE: simple model cannot capture broadly tuned cell adapting to multiple familiar

# 1) unresponsive -> unresponsive; ✅ (subthreshold only PV get stronger because just FF inhibition) 
# nonresponder (subthreshold), only FF PV strengthening
nonresponder = broad.copy()
nonresponder.update({
    "w_ff_init": {'mu': [0.01, 0.01], 'sigma': 0},
    "w_fb_init": {'mu': [0.01, 0.01], 'sigma': 0},
    "w_lat_init": {'mu': [1.5, 1.5], 'sigma': 0},
    "W_pv_init": {'mu': ([0.9, 0.1], [0.1,0.9]), 'sigma': [0, 0]},
    'receives_context': (False, False)
    })
# [NOTE] Might need spiking models to capture sub-threshold behavior
# [NOTE] Because just FF inhibition, no way to prevent FB responses

# idea: to prevent runaway strengthening of FF PV synapses, condition strengthening on
# co-activation of lateral synapses onto PV and feedforward synapses onto PV


# 2) unresponsive -> FF responsive ✅ (different mechanism)
# need different mechanism (strengthened cRF feedback ...)
# NOTE: also the FB response strengthens
un_FF = nonresponder.copy()
un_FF.update({
    "FFrule": "Hebbian",
    'FBrule': "Hebbian",
    "w_ff_init": {'mu': [0.05, 0.05], 'sigma': 0},
    'receives_context': (True, True),
    "lr_ff": 0.001,
    "lr_lat": 0.0001,
    "w_lat_init": {'mu': [0.05, 0.05], 'sigma': 0},
    "W_pv_init": {'mu': ([0.1, 0.0], [0.0,0.1]), 'sigma': [0, 0]},
    })

# 3) unresponsive -> FB responsive ✅
# [NOTE] strengthened by other neurons being active, hard to capture in minimal 1-neuron model :| 
# especially because context independent of input
# unresponsive probably because sub-threshold
un_FB = nonresponder.copy()
un_FB.update({
    'receives_context': (True, True)
    })

# FF -> unresponsive; ✅ (simple) cells that don't receive context and only adapt
FF_un = broad.copy()
FF_un.update({
    "w_ff_init": {'mu': [0.5, 0.5], 'sigma': 0},
    "w_fb_init": {'mu': [1e-7, 1e-7], 'sigma': 0},
    "w_lat_init": {'mu': [0.1, 0.1], 'sigma': 0},
    "W_pv_init": {'mu': ([0.9, 0.1], [0.1,0.9]), 'sigma': [0, 0]},
    'receives_context': (False, False)
    })

# FF -> FF, FB still strengthened, novel FF no adaptation; FF strengthened (diff mechanism)
# different mechanism 
# 🚫(Hebbian FB + less adaptation)- both FF & occluded response (don't see)
# ✅ FF facilitation instead of adaptation (interesting regime; direct competition between FF and LAT inhibition)
# depending on initial condiions & lr can end up with FF strengthening or weakening
FF_FF = FF_un.copy()
FF_FF.update({
    "FFrule": "Hebbian",
    'receives_context': (False, False),
    "lr_ff": 0.001,
    "lr_lat": 0.0001,
    "w_lat_init": {'mu': [0.05, 0.05], 'sigma': 0},
    "W_pv_init": {'mu': ([0.1, 0.0], [0.0,0.1]), 'sigma': [0, 0]},
    })

# FF -> FB ✅
# broad - Familiar adapt and replaced by FB
FF_FB_broad = broad # no reason why novel response should be adapted (boosted novel FF & FB responses)

# narrow, familiar ✅
narrow_familiar = broad.copy()
narrow_familiar.update({
    "w_ff_init": {'mu': [0.9, 0.01], 'sigma': 0},
    "w_fb_init": {'mu': [0.01, 0.01], 'sigma': 0},
    # NOTE: need to already have more PV inhibition for the novel stim. otherwise no reason not to have FF response there as well
    "w_lat_init": {'mu': [0.5, 1.5], 'sigma': 0}, 
    "W_pv_init": {'mu': ([0.9, 0.1], [0.1,0.9]), 'sigma': [0, 0]},
    'lr_lat': 0.1
    })


# narrow novel ✅ strengthen FB to familiar context
# also (less) strengthened FB to unfamiliar context + also enhanced novel response (due to no adaptation + FB boost)
narrow_novel = broad.copy()
narrow_novel.update({
    "w_ff_init": {'mu': [0.01, 0.9], 'sigma': 0},
    "w_fb_init": {'mu': [0.01, 0.01], 'sigma': 0},
    "w_lat_init": {'mu': [0.3, 0.3], 'sigma': 0},
    })

# Overview
# unresponsive -> unresponsive (subthreshold only PV get stronger because just FF inhibition)
# unresponsive -> FF (different mechanism, X minimal circuit)
# unresponsive -> FB (based on strengthened FB without own firing and release from inhibition)

# Don't discuss - rare, different mechanism; not focus
# FB -> unresponsive ??
# FB -> FF ??

# FB -> more FB, ✅
# already FB responsive, becomes even more FB responsive
FB_FB = broad.copy()
FB_FB.update({
    "w_ff_init": {'mu': [1e-7, 1e-7], 'sigma': 0},
    "w_fb_init": {'mu': [0.5, 0.5], 'sigma': 0},
    "w_lat_init": {'mu': [1.5, 1.5], 'sigma': 0},
    "W_pv_init": {'mu': ([1, 0.2], [0.2,1]), 'sigma': [0, 0]},
    })

minimal_configs = {
    # "un_un": nonresponder,
    # "un_FF": un_FF, # Hebbian FF and FB plasticity
    # "un_FB": un_FB,
    
    # "FF_un": FF_un,
    # "FF_FF": FF_FF, # Hebbian FF plasticity
    
    # "FF_FB_broad":broad,
    # "FF_FB_narrow_familiar": narrow_familiar,
    # "FF_FB_narrow_novel": narrow_novel,
    
    "FB_FB": FB_FB
}