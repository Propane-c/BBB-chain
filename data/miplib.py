import pulp
from pathlib import Path
import numpy as np
import data.lpprblm as lp

def load_MPS(file_path):
    """
    读取MPS文件成`LpPrblm`类型
    """
    file_path = Path(file_path)
    # variables_dict, lp= pulp.LpProblem.fromMPS("academictimetablesmall.mps", pulp.LpMinimize)
    variables_dict, lp= pulp.LpProblem.fromMPS(file_path, pulp.LpMinimize)
    c = []
    A_eq = []
    b_eq = []
    G_ub = []
    h_ub = []
    conti_vars =  []
    for i, var in enumerate(variables_dict.values()):
        c.append(lp.objective.get(var) if lp.objective.get(var) is not None else 0)
        if var.cat == pulp.LpContinuous:
            conti_vars.append(i)
    constraint:pulp.LpConstraint
    for constraint in lp.constraints.values():
        coeff = [constraint.get(var) if constraint.get(var) is not None else 0 
                 for var in variables_dict.values()]
        if constraint.sense == pulp.LpConstraintEQ: # "="
            A_eq.append(coeff)
            b_eq.append(-constraint.constant)
        elif constraint.sense == pulp.LpConstraintLE: # "<="
            G_ub.append(coeff)
            h_ub.append(-constraint.constant)
        elif constraint.sense == pulp.LpConstraintGE: # ">="
            # 将'>='约束转化为'<=', 例如: ax >= b 为 -ax <= -b
            G_ub.append([-c for c in coeff])
            h_ub.append(constraint.constant)

    bounds = [(var.lowBound, var.upBound) for var in lp.variables()]
    c = -np.array(c)
    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq)
    G_ub = np.array(G_ub)
    h_ub = np.array(h_ub)
    prblm = LpPrblm(((0, 0), 0), None, 0, c, G_ub, h_ub, A_eq, b_eq, bounds)
    prblm.conti_vars = conti_vars
    # solve_lp(prblm)
    prblm.fathomed = False
    prblm.fthmd_state = False
    save_prblm_pool([prblm], f"var{c.size}_{file_path}", Path.cwd() / "MIPLIB_POOL", prblm_type = NORMAL,saveBounds = True,
                         file_name=f"int{c.size-len(conti_vars)}_conti{len(conti_vars)}_ub{h_ub.size}_eq{b_eq.size}_{file_path.stem}.json")
    # solve_ilp_by_pulp(prblm)
    return prblm
