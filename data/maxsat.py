import numpy as np
from pathlib import Path
import data.lpprblm as lp

def load_maxsat(file_path:str=None):
    file_path = Path(file_path)
    with open(file_path, 'r') as file:
        file_content = file.readlines()
    hard_clauses = []
    soft_clauses = []
    for line in file_content:
        if line.startswith('c'):
            continue
        elif line.startswith('p'): 
            continue
        elif line.startswith('h'): 
            hard_clauses.append([int(x) for x in line.split()[1:-1]])
        elif line.startswith('1'):
            soft_clauses.append([int(x) for x in line.split()[1:-1]])

    var_num = len(set(abs(var) for clause in (hard_clauses + soft_clauses) for var in clause))
    c = np.zeros(len(soft_clauses)+var_num)
    c[:len(soft_clauses)] = 1
    G_ub = np.zeros((len(soft_clauses+ hard_clauses), len(soft_clauses)+var_num))
    h_ub = np.zeros(len(soft_clauses+ hard_clauses))
    for i, clause in enumerate(soft_clauses):
        G_ub[i,i] = 1
        for var in clause:
            G_ub[i, len(soft_clauses)+ abs(var)-1]  = -1 if var >0 else 1
        h_ub[i] = len([var for var in clause if var < 0])
        print(i, h_ub[i])
    for i , clause in enumerate(hard_clauses):
        for var in clause:
            G_ub[i+len(soft_clauses), len(soft_clauses)+ abs(var)-1] = -1 if var>0 else 1
        h_ub[i+len(soft_clauses)] = -1+ len([var for var in clause if var < 0])
    bounds = [(0,1) for _ in range(len(soft_clauses)+var_num)]
    prblm = lp.LpPrblm(((0, 0), 0), None, 0, c, G_ub, h_ub, None, None, bounds)
    lp.save_prblm_pool([prblm], f"var{len(soft_clauses)+var_num}_{file_path}", 
                         Path.cwd() / "testMAXSAT", prblm_type = lp.ZERO_ONE, 
                         file_name=f"var{len(soft_clauses)+var_num}_soft{len(soft_clauses)}_con{h_ub.size}_{file_path.stem}.json")
    # lp.solve_ilp_by_pulp(prblm)