import copy
import json
import os
import random
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pulp
from scipy.optimize import linprog

# 一般整数规划
NORMAL = "normal"
# 0-1规划
ZERO_ONE = "zero_one"
TSP = "tsp"

@dataclass
class IncConstr:
    idx:int
    G_ub_var:int
    h_ub_var:int

class LpPrblm(object):
    def __init__(self, pname = None, pre_pname = None, pheight = None, 
                 c = None, G_ub = None, h_ub = None, A_eq = None,  b_eq = None, 
                 bounds = None, x_nk = None, pre_rest_x = None, 
                 key_pname=None, inc_constrs= None, timestamp = None, fix_pid = None):
        '''
        Param
        -----
        pname is a tuple ((keyprblm_id, subprblmpair_id), prblm_id)
            prblm_id: 1:+ , -1:- , 0:origin_key_prblm

            subprblm_pair_name : a index tuple with two elements
                --namely: (keyprblm_id, subprblmpair_id)

            for keyblock:
                :subprblm_pair_name[0] is the key problem id
                :subprblm_pair_name[1] is 0 means it is a key problem
                :prblm_id is 0 means it is a key problem
                :e.g ((2,0), 0)表示是第2个key prblm
             for miniblock:
                :subprblm_pair_name[0] is the key problem id
                :subprblm_pair_name[1] is the subproblem index under this key problem
                :prblm_id is 1 or -1 for + or - subproblem
                :e.g ((0,1), -1)
                :代表第0个key prblm下的第1个subprblm_pair中的-
        
        minimize::

            c @ x

        subject to::

            G_ub @ x <= h_ub
            A_eq @ x == b_eq
            lb <= x <= ub
        '''
        self.pname = pname 
        self.pre_pname = pre_pname
        self.pheight = pheight
        self.timestamp = timestamp
        self.key_pname = key_pname
        self.fix_pid = fix_pid
        # coefficients
        if c is not None:
            self.c:np.ndarray = c
        else: 
            self.c = None
        self.G_ub:np.ndarray = G_ub
        self.h_ub:np.ndarray = h_ub
        self.A_eq:np.ndarray = A_eq
        self.b_eq:np.ndarray = b_eq
        if G_ub is None or G_ub.size == 0:
            self.G_ub = None
            self.h_ub = None
        if A_eq is None or A_eq.size == 0:
            self.A_eq = None
            self.b_eq = None
        self.bounds:list[tuple] = bounds
        self.conti_vars = []
        self.inc_constrs:list[IncConstr] = inc_constrs
        if inc_constrs is None:
            self.inc_constrs = []
        self.init_ix = None # keyblock 初始化的整数解
        self.init_iz = None # keyblock 初始化的目标值
        # index of the variable branched out
        self.x_nk = x_nk
        self.pre_rest_x = pre_rest_x
        # results
        self.feasible = None # feasible or not
        self.x_lp = None # linear optimal solution
        self.z_lp = None # objective value(upper bound of P_h)
        self.x_pulp = None # linear optimal solution by `pulp`
        self.z_pulp = None # objective value by `pulp`
        self.ix_pulp = None # integer optimal solution by `pulp`
        self.iz_pulp = None # objective value by `pulp`
        # fathomed field
        self.fathomed = None
        self.allInt = None
        self.lb_prblm:LpPrblm = None
        # states
        self.fthmd_state = None
    
    def get_Gub_hub(self, key_lp: 'LpPrblm' = None):
        if self.key_pname is None:
            return self.G_ub, self.h_ub
        if len(self.inc_constrs) == 0:
            return key_lp.G_ub, key_lp.h_ub
        inc_Gub = np.zeros((len(self.inc_constrs), key_lp.c.size))
        inc_hub = np.zeros(len(self.inc_constrs))
        for i, inc_con in enumerate(self.inc_constrs):
            inc_Gub[i][inc_con.idx] = inc_con.G_ub_var
            inc_hub[i] = inc_con.h_ub_var
        if key_lp.G_ub is None:
            return inc_Gub,inc_hub
        G_ub = np.append(key_lp.G_ub, inc_Gub, axis=0)
        h_ub = np.append(key_lp.h_ub, inc_hub, axis=0)
        return G_ub, h_ub
        
        
    def __deepcopy__(self, memo): 
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if cls.__name__ == 'LpPrblm' and k != 'pre_p' and k != 'next_pairs':
                setattr(result, k, copy.deepcopy(v, memo))
            if cls.__name__ == 'LpPrblm' and k == 'next_pairs':
                setattr(result, k, [])
            if cls.__name__ == 'LpPrblm' and k == 'pre_p':
                setattr(result, k, None)
            if cls.__name__ != 'LpPrblm':
                setattr(result, k, copy.deepcopy(v, memo))
        return result
    
    def __eq__(self, other: 'LpPrblm'):
        return self.pname == other.pname

        
    def __repr__(self) -> str:
        omit_keys = {'timestamp'}
        def _dict_formatter(d, n=0, mplus=1):
            if isinstance(d, dict):
                m = max(map(len, list(d.keys()))) + mplus  # width to print keys
                s = '\n'.join([k.rjust(m) + ': ' +  # right justified, width m
                            _indenter(_dict_formatter(v, m+n+2, 0), m+2)
                            for k, v in d.items()])  # +2 for ': '
            else:
                with np.printoptions(linewidth=76-n, edgeitems=10, threshold=12,
                                    formatter={'float_kind': _float_formatter_10}):
                    s = str(d)
            return s
        def _indenter(s, n=0):
            split = s.split("\n")
            indent = " "*n
            return ("\n" + indent).join(split)
        def _float_formatter_10(x):
            if np.isposinf(x):
                return "       inf"
            elif np.isneginf(x):
                return "      -inf"
            elif np.isnan(x):
                return "       nan"
            return np.format_float_scientific(x, precision=3, pad_left=2, unique=False)
        lpdict = copy.deepcopy(self.__dict__)
        if self.lb_prblm is not None:
            lpdict.update({'lb_prblm': self.lb_prblm.pname})
        if lpdict.keys():
            if self.pname[1] == 1:
                pstr = f'({self.pname[0]},+)'
            elif self.pname[1] == -1:
                pstr = f'({self.pname[0]},-)'
            elif self.pname[1] == 0:
                pstr = f'origin_prblm {self.pname[0]}'
            divstr = '\n'+'-'*10+pstr+'-'*10+'\n'
            for omk in omit_keys:
                del lpdict[omk]
            return divstr+ _dict_formatter(lpdict)
        else:
            return self.__class__.__name__ + "()"
    
    def all_integer(self):
        if self.x_lp is None:
            return False
        for i, x in enumerate(self.x_lp):
            if i in self.conti_vars:
                continue
            if not x.is_integer():
                return False
        return True
            

    def update_fathomed_state(self):
        """只要next中有任一子问题对fathomed, 则该问题也fathomed

        Returns:
            bool: 该问题的状态是否fathomed
        """
        warnings.warn("update_fathomed_state is deprecated", DeprecationWarning)
        if len(self.next_pairs) > 0:
            for (p1,p2) in  self.next_pairs:
                if p1.fthmd_state and p2.fthmd_state:
                    self.fthmd_state = True
                    return self.fthmd_state
        self.fthmd_state = False
        return self.fthmd_state


def solve_lp(lp_prblm:"LpPrblm" = None, c = None, G_ub = None, h_ub = None, 
             A_eq = None, b_eq = None, bounds = None, solve_prob = 1):
    '''Solve the given lp problem, and save the result in the LpPrblm.'''
    s = np.random.random()
    if s < solve_prob:
        if c is None:
            r = linprog(lp_prblm.c, lp_prblm.G_ub, lp_prblm.h_ub, lp_prblm.A_eq, lp_prblm.b_eq, lp_prblm.bounds)
        else:
            r = linprog(c, G_ub, h_ub, A_eq, b_eq, bounds)
        lp_prblm.feasible = r.success
        if r.success:
            lp_prblm.x_lp = r.x
            lp_prblm.z_lp = r.fun
        solved_success = True
    else: 
        solved_success = False
    # print(lp_prblm.z_lp)
    return solved_success

def rand_prblm(var_num, conNum:int = 5):
    """
    随机生成指定变量数量的`一般`整数规划问题
    """
    c = np.array(np.random.choice(np.arange(-50,150),var_num,True))
    G_ub = np.array([np.random.choice(np.arange(-50,150),var_num,True) for _ in range(conNum)])
    h_ub = np.array(np.random.choice(np.arange(-50,150),len(G_ub),True))
    A_eq = None
    b_eq = None
    bounds = []
    for _ in range(var_num):
        bounds.append((0, None))
    p = LpPrblm(((0, 0), 0), None, 0, c, G_ub, h_ub, A_eq, b_eq, bounds)
    solve_lp(p)
    p.fathomed = False
    p.fthmd_state = False 
    return p


def rand_01(var_num, conNum:int = 5):
    """
    随机生成指定变量数量的`0-1`整数规划问题
    """
    c = np.array(np.random.choice(np.arange(-50,150),var_num,True))
    G_ub = np.array([np.random.choice(np.arange(-50,150),var_num,True) for _ in range(conNum)])
    h_ub = np.array(random.sample(range(-50,150),len(G_ub)))
    A_eq = None
    b_eq = None
    bounds = []
    for _ in range(var_num):
        bounds.append((0, 1))
    p = LpPrblm(((0, 0), 0), None, 0, c, G_ub, h_ub, A_eq, b_eq, bounds)
    solve_lp(p)
    p.fathomed = False
    p.fthmd_state = False 
    return p

def prblm_generator(var_num, type:str = NORMAL , conNum:int = 5):
    """
    产生可解的线性规划问题，同时不可一次解出整数解
    注：线性规划问题可能没有整数解
    """
    while True:
        if type == NORMAL:
            p = rand_prblm(var_num, conNum)
        elif type == ZERO_ONE:
            p = rand_01(var_num, conNum)
        if p.feasible is False:
            continue
        if p.x_lp is not None:
            if all(list(map(lambda f: f.is_integer(), p.x_lp))):
                continue
        if p.z_lp == 0:
            continue
        solve_ilp_by_pulp(p)
        if p.iz_pulp == 0:
            continue
        return p


def test1():
    c = np.array([3, 13, 12])
    G_ub = np.array([[2, 9, 9], [11, -8, 0]])
    h_ub = np.array([40, 82])
    A_eq = None
    b_eq = None
    bounds = [(0, None), (0, None), (0, None)]
    p = LpPrblm(((0, 0), 0), None,0, c, G_ub, h_ub, A_eq, b_eq, bounds)
    solve_lp(p)
    
    p.fathomed = False
    p.fthmd_state = False 
    return p

def test2():
    c = np.array([35,-47, -37, -10, -78, 46, -32, -2, 5, 28, -90, -6, -45, -67, 33, -18, -86, -46, -7, 50])
    G_ub = np.array(
        [[-7, 32, 51, 34, 86, 75, 3, 77, 64, -3, 19, 30, 38, 78, -15, 10, -17, 4, 49, 37], 
         [-27, 86, 41, 88, 26, 24, -13, 48, -46, 80, 7, 28,  8, -12, -29, 89, 57, 21, -5, 69], 
         [71, 58, -44, -4, -42, 25, -2, -26, -18, 82, -50, 13, 31, 9, 14, -27, -14, -3, -8, 27],
         [49, 41, -19, 9,  16, -35, 39,  45, -38, -16, -25, 43, -6, -42, 48, 6, 18, 32, -44, -39]])
    h_ub = np.array([-18, 11, 65, 76])
    A_eq = None
    b_eq = None
    bounds = [(0, None), (0, None), (0, None), (0, None), (0, None), 
              (0, None), (0, None), (0, None), (0, None), (0, None), 
              (0, None), (0, None), (0, None), (0, None), (0, None), 
              (0, None), (0, None), (0, None), (0, None), (0, None)]
    p = LpPrblm(((0, 0), 0), None, 0, c, G_ub, h_ub, A_eq, b_eq, bounds)
    solve_lp(p)
    p.fathomed = False
    p.fthmd_state = False 
    return p

def test3():
    c = np.array([23, 53, 28, -6, -29, 78, -18, -47, 51, 72])
    G_ub= np.array([[26, 59, -46, -9, 28, 41, -19, 29, 34, 44],
                [-20, 83, 44, -18, 46, 84, -5, -22, 51, 68],
                [2, -30, 74, -38, -28, -32, 59, 86, -18, -4],
                [68, 86, 55, -1, 33, 51, 20, -42, 46, 40]])
    h_ub= np.array([68, 85, 71, 2])
    A_eq = None
    b_eq = None
    bounds = [(0, None), (0, None), (0, None), (0, None), (0, None), 
              (0, None), (0, None), (0, None), (0, None), (0, None)]
    p = LpPrblm(((0, 0), 0), None, 0, c, G_ub, h_ub, A_eq, b_eq, bounds)
    solve_lp(p)
    p.fathomed = False
    p.fthmd_state = False 
    return p

def test4():
    c = np.array(    [   2, -22, -43,  -2,  -19])
    G_ub = np.array([[  14,  58,  28,  99,   37],
                     [  41,  92, -26,  10,  -16],
                     [ 105, 125, -17,  68,  -16],
                     [ -38, -17, 104,  27,   49],
                     [ 105,  90,  71,  70,  -43]])
    h_ub = np.array( [ 133, 110, 120, 107,  74])
    A_eq = None
    b_eq = None
    bounds = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]
    p = LpPrblm(((0, 0), 0), None, 0, -c, G_ub, h_ub, A_eq, b_eq, bounds)
    solve_lp(p)
    p.fathomed = False
    p.fthmd_state = False
    return p

def test5():
    """hard"""
    c = np.array([-3, 94, -26, 138, 18, 126, 13, 144, 99, 21, 92, 83, 122, 54, 43, 76, 
                  147, 34, 135, -9, 149, 117, 128, 112, 69, -14, 96, 148, 14, 65, -24, 
                  -36, 132, -44, -1, -15, 32, 111, 140, 141, -8, -32, -29, -16, 129, -37,
                   -40, 100, 53, 73])
    G_ub =  np.array(
        [[11, 136, -20, 110, 14, 121, 86, 101, -37, 61, -26, 16, -48, 29, 37, -23, 119, 
          56, 57, 99, 90, 42, 131, -17, -38, 108, 34, 143, 138, 0, -11, -1, 117, -22, 
          10, 3, -44, -40, -14, -12, 23, 22, 36, 149, 7, 118, 55, 53, 41, 126], 
         [-39, 138, 32, 28, 108, -8, -38, 34, 139, 35, -15, -29, 60, 50, 88, 29, -41, 
          72, 111, -21, -44, -35, 112, -49, 40, -6, -24, 51, 74, 21, 104, 133, 68, -28,
          -27, 146, 143, 123, 130, 49, 125, -13, 76, 84, 126, -10, 147, 20, 14, 37], 
         [135, 134, -28, 40, 146, 104, 16, 8, 114, -24, 15, 0, 71, 61, 22, 88, 68, 37,
           -4, -10, -46, 29, 46, 103, 70, 49, 64, 131, 130, -23, 6, 144, 56, 21, 109, 
           78, -8, 13, 143, 79, 73, -21, 138, -17, 4, -2, 85, 33, 19, 54], 
         [-5, 131, 31, -2, 52, -21, -19, 126, -45, 77, -1, -14, 27, -42, 148, 94, 92,
          136, 144, 88, 99, 146, 87, -24, 10, 81, 100, 60, 123, 21, 6, 59, 67, 91, 141,
          -28, 105, 66, 68, -49, 28, 11, 46, -9, -37, 25, -26, 16, -18, 1], 
         [76, -27, 15, 25, 38, 81, -47, 88, -50, 62, -14, 56, 133, -16, 13, -31, 3, -25,
          94, -9, 21, 129, 122, 22, 20, 104, 70, 34, 89, -10, 26, -46, 65, 118, 29, -36,
          142, 141, 14, 75, 35, 138, 50, 24, 60, -13, 115, -3, 51, 124]])
    h_ub = np.array([82, 88, 97, -21, 50])
    A_eq = None
    b_eq = None
    bounds = []
    for _ in range(len(c)): 
        bounds.append((0,None))
    p = LpPrblm(((0, 0), 0), None, 0, -c, G_ub, h_ub, A_eq, b_eq, bounds)
    print(linprog(c, G_ub, h_ub, A_eq, b_eq, bounds))
    print(linprog(-c, G_ub, h_ub, A_eq, b_eq, bounds))
    solve_lp(p)
    p.fathomed = False
    p.fthmd_state = False
    return p

def prblm_pool_generator(total_num:int, var_num:int, prblm_type:str = NORMAL):
    """
    产生问题数固定的问题池
    """
    print(f"Generating problem pool with size {total_num} and var_num {var_num}")
    prblm_pool:list[LpPrblm] = []
    for i in range(total_num):
        if i % 10 == 0:
            print(f"Generate Progress: {i}")
        p = prblm_generator(var_num, prblm_type)
        prblm_pool.append(p)
    print("")
    for i in range(len(prblm_pool)):
        prblm_pool[i].fix_pid = i
    return prblm_pool

def save_prblm_pool(prblm_pool:list[LpPrblm], save_dir, prblm_type:str = ZERO_ONE,
                    saveBounds:bool = False, file_name = None):
    """ 保存产生的问题池 """

    def swich_to_list(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        elif isinstance(x, list):
            return x
        return []
    
    if not os.path.exists(save_dir):
        save_dir.mkdir(parents=True)
    var_num = len(prblm_pool[0].c)
    file_name = f'{var_num}vars.json' if file_name is None else file_name
    with open(save_dir / file_name, 'a') as f:
        for prblm in prblm_pool:
            
            p_dict = {
                'c': swich_to_list(prblm.c),
                'G_ub':swich_to_list(prblm.G_ub),
                'h_ub':swich_to_list(prblm.h_ub),
                'conti_vars':prblm.conti_vars,
                'type': prblm_type
            }
            if prblm.A_eq is not None:
                p_dict.update({
                    'A_eq':swich_to_list(prblm.A_eq),
                    'b_eq':swich_to_list(prblm.b_eq),
                })
            if prblm.iz_pulp is not None:
                p_dict.update({
                    'ix_pulp': prblm.ix_pulp,
                    'iz_pulp':prblm.iz_pulp
                })
            if saveBounds:
                p_dict.update({'bounds':prblm.bounds})  
            if prblm.init_ix is not None:
                p_dict.update({
                    'init_ix':swich_to_list(prblm.init_ix),
                    'init_iz':prblm.init_iz
                })
            p_json = json.dumps(p_dict)
            f.write(p_json + '\n')

def load_prblm_pool_from_json(file_path:str, save_path:str = None):
    """
    读取给定的file_path的prblm_pool
    """
    print("loading problem pool....")
    prblm_pool:list[LpPrblm] = []
    with open(file_path, 'r') as f:
        p_json_list = f.read().split('\n')[:-1]
        for fix_pid, p_json in enumerate(p_json_list):
            p_data = json.loads(p_json)
            c = np.array(p_data['c'])
            G_ub = np.array(p_data['G_ub'])
            h_ub = np.array(p_data['h_ub'])
            A_eq = np.array(p_data['A_eq']) if 'A_eq' in p_data.keys() else None
            b_eq = np.array(p_data['b_eq']) if 'b_eq' in p_data.keys() else None
            init_ix = p_data['init_ix'] if 'init_ix' in p_data.keys() else None
            init_iz = p_data['init_iz'] if 'init_iz' in p_data.keys() else None

            if 'bounds' in p_data.keys():
                bounds = [tuple(bound) for bound in p_data['bounds']]
            else:
                bound = (0,1) if 'type' in p_data.keys() and p_data['type'] == ZERO_ONE else (0, None)
                bounds = [bound for _ in range(len(c))]
            
            p = LpPrblm(((0, 0), 0), None, 0, copy.deepcopy(c), copy.deepcopy(G_ub), 
                        copy.deepcopy(h_ub), copy.deepcopy(A_eq),copy.deepcopy(b_eq), bounds)
            p.conti_vars = p_data['conti_vars'] if 'conti_vars' in p_data.keys() else []
            p.init_ix = init_ix
            p.init_iz = init_iz
            
            if 'ix_pulp' in p_data.keys() :
                p.ix_pulp = p_data['ix_pulp']
            if 'iz_pulp' in p_data.keys() :
                p.iz_pulp = p_data['iz_pulp']
            else:
                solve_ilp_by_pulp(p)
            
            solve_lp(p)
            p.fathomed = False
            p.fthmd_state = False
            p.fix_pid = fix_pid 
            prblm_pool.append(p)
        return prblm_pool
    
def solve_lp_by_pulp(prblm:LpPrblm):
    """
    使用`pulp`求线性数规划问题
    :return: 最优目标值，如果不可解返回None
    """
    c = prblm.c
    G_ub = prblm.G_ub
    h_ub = prblm.h_ub
    A_eq = prblm.A_eq
    b_eq = prblm.b_eq
    bounds = prblm.bounds

    lp = pulp.LpProblem("LP", pulp.LpMinimize)
    vars = [pulp.LpVariable(f'x{i}', lowBound=b[0], upBound=b[1]) 
            for i, b in enumerate(bounds)]

    lp += pulp.lpDot(c, vars)
    if G_ub is not None:
        for i in range(len(G_ub)):
            lp += (pulp.lpDot(G_ub[i], vars) <= h_ub[i])
    if A_eq is not None:
        for i in range(len(A_eq)):
            lp += (pulp.lpDot(A_eq[i], vars) == b_eq[i])
    lp.solve(pulp.PULP_CBC_CMD(msg=False))
    if lp.status == pulp.LpStatusOptimal:
        # for v in lp.variables():
        #     print(v.name, "=", v.varValue)
        obj_value = pulp.value(lp.objective)
        # print("Optimal value:", obj_value)
        prblm.z_pulp = obj_value
        prblm.x_pulp = [v.varValue for v in lp.variables()]
        return obj_value
    else:
        return None
    
def solve_ilp_by_pulp(prblm:LpPrblm):
    """
    使用`pulp`求解整数规划问题
    :return: 最优目标值，如果不可解返回None
    """
    c = prblm.c
    G_ub = prblm.G_ub
    h_ub = prblm.h_ub
    A_eq = prblm.A_eq
    b_eq = prblm.b_eq
    bounds = prblm.bounds
    conti_vars = prblm.conti_vars

    ilp = pulp.LpProblem("ILP", pulp.LpMinimize)
    vars = []
    for i, b in enumerate(bounds):
        cat = pulp.LpInteger if i not in conti_vars else pulp.LpContinuous
        vars.append(pulp.LpVariable(f'x_{i}', lowBound=b[0], upBound=b[1], cat=cat))

    ilp += pulp.lpDot(c, vars)
    if G_ub is not None:
        for i in range(len(G_ub)):
            ilp += (pulp.lpDot(G_ub[i], vars) <= h_ub[i])
    if A_eq is not None:
        for i in range(len(A_eq)):
            ilp += (pulp.lpDot(A_eq[i], vars) == b_eq[i])
    # print(ilp)
    ilp.solve(pulp.PULP_CBC_CMD(msg=False))
    if ilp.status != pulp.LpStatusOptimal:
        return None
    # for i, v in enumerate(ilp.variables()):
    #     print(v.name, "=", v.varValue)
    obj_value = pulp.value(ilp.objective)
    # print("Optimal value Integer:", obj_value)
    prblm.iz_pulp = obj_value
    prblm.ix_pulp = [v.varValue for v in ilp.variables()]
    return obj_value