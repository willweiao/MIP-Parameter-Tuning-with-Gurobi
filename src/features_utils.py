from gurobipy import *
import numpy as np

def extract_features(model):
    vars = model.getVars()
    constrs = model.getConstrs()
    A = model.getA()

    n_vars = len(vars)
    n_bin = sum(1 for v in vars if v.VType == GRB.BINARY)
    n_int = sum(1 for v in vars if v.VType == GRB.INTEGER)
    n_cont = sum(1 for v in vars if v.VType == GRB.CONTINUOUS)
    n_constrs = len(constrs)
    eq_constrs = sum(1 for c in constrs if c.Sense == '=')

    nz_per_row = [A.getrow(i).nnz for i in range(n_constrs)] if n_constrs > 0 else [0]
    avg_nz = np.mean(nz_per_row)

    obj_coeffs = [abs(v.Obj) for v in vars if v.Obj != 0]
    obj_range = (max(obj_coeffs) / min(obj_coeffs)) if obj_coeffs and min(obj_coeffs) > 0 else 1.0
    int_ratio = (n_bin + n_int) / n_vars if n_vars > 0 else 0

    return {
        'num_vars': n_vars,
        'num_bin_vars': n_bin,
        'num_int_vars': n_int,
        'num_cont_vars': n_cont,
        'num_constrs': n_constrs,
        'num_eq_constrs': eq_constrs,
        'avg_nz_per_row': avg_nz,
        'obj_range': obj_range,
        'int_var_ratio': int_ratio
    }