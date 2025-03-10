import os
import random
import numpy as np
from pathlib import Path
import sys
sys.path.append("E:\Files\gitspace\\bbb-github")
import data.lpprblm as lp

def spot_to_ilp(spot_file_path, task_num = None):
    with open(spot_file_path, 'r') as file:
        lines = file.readlines()

    n_tasks = int(lines[0].strip())
    task_lines = lines[1:n_tasks + 1]
    tasks = [list(map(int, line.strip().split())) for line in task_lines]
    constraint_lines = lines[n_tasks + 2:]
    constraints = [list(map(int, line.strip().split())) for line in constraint_lines]
    c, A_eq, b_eq, G_ub, h_ub, bounds = [], [], [], [], [], []
    variable_index = 0
    task_to_variables = [] 

    for task in tasks:
        task_id = task[0]
        weight = task[1]
        domain_size = task[2]
        variables_for_task = []
        for i in range(domain_size):
            c.append(weight)
            variables_for_task.append(variable_index)
            bounds.append((0, 1))
            variable_index += 1
        task_to_variables.append(variables_for_task)
    
        row = [0] * variable_index
        for var_idx in variables_for_task:
            row[var_idx] = 1
        G_ub.append(row)
        h_ub.append(1)
    
    total_variables = len(c) 
    for row in G_ub:
        if len(row) < total_variables:
            row.extend([0] * (total_variables - len(row)))
    
    for constraint in constraints:
        constraint_type = constraint[0]
        if constraint_type == 2:
            task1, task2 = constraint[1], constraint[2]
            forbidden_values = constraint[3:]
            for i in range(0, len(forbidden_values), 2):
                val1, val2 = forbidden_values[i], forbidden_values[i + 1]
                if val1 > len(task_to_variables[task1]):
                    vars1 = task_to_variables[task1][0] 
                else:
                    vars1 = task_to_variables[task1][val1 - 1]
                if val2 > len(task_to_variables[task2]):
                    vars2 = task_to_variables[task2][0]
                else:
                    vars2 = task_to_variables[task2][val2 - 1]
                row = [0] * len(c)
                row[vars1] = 1
                row[vars2] = 1
                G_ub.append(row)
                h_ub.append(1)
        elif constraint_type == 3: 
            task1, task2, task3 = constraint[1], constraint[2], constraint[3]
            forbidden_values = constraint[4:]
            for i in range(0, len(forbidden_values), 3):
                val1, val2, val3 = forbidden_values[i:i + 3]
                if val1 > len(task_to_variables[task1]):
                    vars1 = task_to_variables[task1][0]
                else:
                    vars1 = task_to_variables[task1][val1 - 1]
                if val2 > len(task_to_variables[task2]):
                    vars2 = task_to_variables[task2][0]
                else:
                    vars2 = task_to_variables[task2][val2 - 1]
                if val3 > len(task_to_variables[task3]):
                    vars3 = task_to_variables[task3][0]
                else:
                    vars3 = task_to_variables[task3][val3 - 1]
                row = [0] * len(c)
                row[vars1] = 1
                row[vars2] = 1
                row[vars3] = 1
                G_ub.append(row)
                h_ub.append(2) 
    c = np.array(c)
    G_ub = np.array(G_ub) if G_ub else None
    h_ub = np.array(h_ub) if h_ub else None
    A_eq = None
    b_eq = None
    print(c)
    orig_prblm = lp.LpPrblm(((0, 0), 0), None, 0, -c, G_ub, h_ub, A_eq, b_eq, bounds)
    orig_prblm.init_ix = [1 if i == np.argmax(c) else 0 for i in range(len(c))]
    orig_prblm.init_iz = int(-c[np.argmax(c)])
    orig_prblm.fathomed = False
    orig_prblm.fthmd_state = False 
    # orig_prblm.iz_pulp = -19125
    lp.solve_ilp_by_pulp(orig_prblm)
    lp.solve_lp(orig_prblm)
    if task_num is not None:
        lp.save_prblm_pool([orig_prblm], Path.cwd() / "Problem Pools"/ "SPOT" / "Origin", lp.NORMAL, True, f"{task_num}.json")
    return orig_prblm

def extract_new_spot(n_reduced_variables, file_id, instance):
    # instance = 5
    # n_reduced_variables = 10
    path = Path('E:\Files\A-blockchain\\branchbound\SPOT5\data', f'{instance}.spot')
    with open(path) as f:
        lines = f.readlines()
    lines = [list(map(int, line.strip().split(' '))) for line in lines]
    n_variables = lines[0][0]
    n_constraints = lines[n_variables+1][0]
    l_variables = lines[1:n_variables+1]
    l_constraints = lines[n_variables+2:len(lines)]
    l_reduced_variables = random.sample(l_variables, n_reduced_variables)
    l_reduced_constraints = []
    l_ID = []
    for i in range(len(l_reduced_variables)):
        l_ID.append(l_reduced_variables[i][0])
    for i in range(n_constraints):
        insert = True
        if(l_constraints[i][0]==2):
            for j in 1,2:
                if l_constraints[i][j] not in l_ID:
                    insert = False
        else:
            for j in 1,2,3:
                if l_constraints[i][j] not in l_ID:
                    insert = False
        if insert == True:
            l_reduced_constraints.append(l_constraints[i])
    print(f'Number of variables: {n_variables}')
    print(f'Number of constraints: {n_constraints}')

    save_dir = Path(f'e:\Files\A-blockchain\\branchbound\SPOT5\generated_data\\news3\\{n_reduced_variables}')
    if not os.path.exists(save_dir):
        save_dir.mkdir(parents=True)
    path = Path(save_dir, f'{n_reduced_variables}_{instance}_{file_id}.spot')
    with open(path, "w+") as f:
        f.write(str(n_reduced_variables))
        f.write("\n")
        for i in range(len(l_reduced_variables)):
            l_reduced_variables[i][0] = i
            f.write(str(l_reduced_variables[i]).replace("[", "").replace("]", "").replace(",",""))
            f.write("\n")
        f.write(str(len(l_reduced_constraints)))
        f.write("\n")
        for i in range(len(l_reduced_constraints)):
            if (l_reduced_constraints[i][0] == 2):
                l_reduced_constraints[i][1] = l_ID.index(l_reduced_constraints[i][1])
                l_reduced_constraints[i][2] = l_ID.index(l_reduced_constraints[i][2])
            else:
                l_reduced_constraints[i][1] = l_ID.index(l_reduced_constraints[i][1])
                l_reduced_constraints[i][2] = l_ID.index(l_reduced_constraints[i][2])
                l_reduced_constraints[i][3] = l_ID.index(l_reduced_constraints[i][3])
            f.write(str(l_reduced_constraints[i]).replace("[", "").replace("]", "").replace(",",""))
            f.write("\n")

            
if __name__ == "__main__":
    id = 100
    # Convert the content into ILP problem form
    # task_nums = [30,40,50,60,70,80,90,100]
    # instances = [414, 509, 5]
    # for task_num in task_nums:
    #     for i in range(200):
    #         for instance in instances:
    #             extract_new_spot(task_num, i,instance)
    #             spot_file_path = f"E:\Files\A-blockchain\\branchbound\SPOT5\generated_data\\news3\\{task_num}\\{task_num}_{instance}_{i}.spot"
    #             p = spot_to_ilp(spot_file_path, task_num)
    #             print(len(p.c), len(p.h_ub))

    instance = 509
    spot_file_path = f"E:\Files\A-blockchain\\branchbound\SPOT5\generated_data\\news3\\100\\100_509_1.spot"  
    spot_to_ilp(spot_file_path, 100)