import numpy as np
from pandas import DataFrame

def prepare_collect():
    # Collect raw outputs during the loop
    steps = []
    ys = []
    xs = []
    ps = []
    cs = []
    w_ffs = []
    w_fbs = []
    w_lats = []
    W_pvs = []
    return steps, ys, xs, ps, cs, w_ffs, w_fbs, w_lats, W_pvs

def collect_outputs(step, x, y, p, c, model, collections):
    steps, ys, xs, ps, cs, w_ffs, w_fbs, w_lats, W_pvs = collections
    steps.append(step)
    ys.append(y.item())
    # Snapshot immutable copies; avoid NumPy views to tensors that mutate in-place.
    xs.append(x.detach().cpu().numpy().copy())
    ps.append(p.detach().cpu().numpy().copy())
    cs.append(c.detach().cpu().numpy().copy())
    w_ffs.append(model.w_ff.detach().cpu().numpy().copy())
    w_fbs.append(model.w_fb.detach().cpu().numpy().copy())
    w_lats.append(model.w_lat.detach().cpu().numpy().copy())
    W_pvs.append(model.W_pv.detach().cpu().numpy().copy())
    return steps, ys, xs, ps, cs, w_ffs, w_fbs, w_lats, W_pvs

def build_res(collections, model, debug=False):
    steps, ys, xs, ps, cs, w_ffs, w_fbs, w_lats, W_pvs = collections
    # Build results DataFrame once at the end
    res = {'step': steps, 'y': ys}
    
    # Stack collected arrays
    xs = np.array(xs)
    ps = np.array(ps)
    cs = np.array(cs)
    w_ffs = np.array(w_ffs)
    w_fbs = np.array(w_fbs)
    w_lats = np.array(w_lats)
    W_pvs = np.array(W_pvs)
    
    if debug: breakpoint()
    
    # Add all columns to results dict
    for i_in in range(model.n_features):
        res[f'x_{i_in}'] = xs[:, i_in]
        res[f'w_ff_{i_in}'] = w_ffs[:, i_in]
    for i_ctxt in range(model.n_context):
        res[f'c_{i_ctxt}'] = cs[:, i_ctxt]
        res[f'w_fb_{i_ctxt}'] = w_fbs[:, i_ctxt]
    for i_pv in range(model.n_pv):
        res[f'p_{i_pv}'] = ps[:, i_pv]
        res[f'w_lat_{i_pv}'] = w_lats[:, i_pv]
        for i_pv_in in range(model.n_features):
            res[f'W_pv_{i_pv}_{i_pv_in}'] = W_pvs[:, i_pv, i_pv_in]
    
    return DataFrame(res)
