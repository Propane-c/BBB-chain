import copy
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pulp
from scipy.optimize import linprog

from data.lpprblm import ZERO_ONE, LpPrblm

RES_PATH = Path.cwd() / "RES_ALGO" / time.strftime("%Y%m%d") / time.strftime('%H%M%S')
if not os.path.exists(RES_PATH):
    RES_PATH.mkdir(parents=True, exist_ok=True)

def get_time(f):
    def inner(*arg, **kwarg):
        s_time = time.time()
        res = f(*arg, **kwarg)
        e_time = time.time()
        print('耗时：{}秒'.format(e_time - s_time))
        return res

    return inner
@dataclass
class IncConstr:
    idx:int
    G_ub_var:int
    h_ub_var:int
class BBNode(object):
    def __init__(self, node_id, z=None, x=None, inc_constrs=None, feasible = None):
        self.z = z
        self.x = x
        self.feasible = feasible
        self.inc_constrs:list[IncConstr] = inc_constrs if inc_constrs is not None else []
        self.node_id = node_id
        self.last = None
        self.next = []

    def addlast(self,bbnode:'BBNode'):
        self.last = bbnode.node_id
    
    def addnext(self,bbnode:'BBNode'):
        self.next.append(bbnode.node_id)

    def shownode(self):
        print('node_id: {}\n fun: {}\n x: {}\n'.format(
            self.node_id, self.z, self.x))
        if self.next is not []:
            print('next_nodes:',[n for n in self.next])
        else: 
            print('next:[]')
        if self.last is not None:
            print('last_node:',self.last,'\n')
        else:
            print('last_node: None\n')

    

    


class BranchandBound():
    def __init__(self, c, G_ub, h_ub, A_eq, b_eq, 
                 bounds, conti_vars = None, prblm_path=None):
        # 全局参数
        self.LOWER_BOUND = -sys.maxsize
        self.UPPER_BOUND = sys.maxsize
        self.opt_val = None
        self.opt_x = None
        self.opt_node = None
        # self.Q = Queue()#广度优先
        self.open_nodes:list[BBNode] = []
        self.id_counter = 0

        # 这些参数在每轮计算中都不会改变
        self.c = -c
        self.A_eq = A_eq
        self.b_eq = b_eq
        self.G_ub = G_ub
        self.h_ub = h_ub
        if G_ub is None or G_ub.size == 0:
            self.G_ub = None
            self.h_ub = None
        if A_eq is None or A_eq.size == 0:
            self.A_eq = None
            self.b_eq = None
        self.bounds = bounds
        self.z_lps = []
        self.lower_bounds = []
        self.prblm_path = Path(prblm_path) if prblm_path is not None else None

        if prblm_path is None:
            file_name = time.strftime("%H%M%S") + ".json"
        else:
            file_name = f"{self.prblm_path.stem}_{time.strftime('%H%M%S')}.json" 
        self.save_path = RES_PATH / file_name 

        if conti_vars is None:
            self.conti_vars = []
        else:
            self.conti_vars = conti_vars
        # 首先计算一下初始问题
        r = linprog(self.c, self.G_ub, self.h_ub, self.A_eq, self.b_eq, self.bounds)
        self.pulpRes = 0
        # self.pulpRes = solve_ilp_by_pulp(c,G_ub, h_ub,A_eq, b_eq,bounds,self.conti_vars)
        # 若最初问题线性不可解
        if not r.success:
            with open(self.save_path, 'w') as f:
                d = {"problem":self.prblm_path.stem,
                    "pulpRes":self.pulpRes,
                    "z_lps":"infeasible"}
                json.dump(d, f)
            raise ValueError(F'{self.prblm_path.stem}: Not a feasible problem!')

        # 将解和约束参数放入队列
        self.head = BBNode(self.id_counter, r.fun, r.x)
        # self.z_lps.append((self.head.node_id, r.fun))
        self.z_lps.append(r.fun)
        # self.head = LpPrblm(0, None, 0, c, G_ub, h_ub, A_eq, b_eq, bounds)
        # self.Q.put(self.head)
        
        
        self.open_nodes.insert(0, self.head)
        
        with open(self.save_path, 'w') as f:
            d = {"problem":self.prblm_path.stem,
                 "pulpRes":self.pulpRes,
                 "z_lps": self.z_lps,
                 "lower_bounds": self.lower_bounds}
            json.dump(d, f)

    def all_integer(self, node:BBNode):
        if node.x is None:
            return False
        for i, x in enumerate(node.x):
            if i in self.conti_vars:
                continue
            if not x.is_integer():
                return False
        return True
    
    def get_Gub_hub(self, node:BBNode):
            # if node.key_pname is None:
            #     return node.G_ub, node.h_ub
            if len(node.inc_constrs) == 0:
                return self.G_ub, self.h_ub
            # G_ub = key_lp.G_ub
            # h_ub = key_lp.h_ub
            inc_Gub = np.zeros((len(node.inc_constrs), self.c.size))
            inc_hub = np.zeros(len(node.inc_constrs))
            for i, inc_con in enumerate(node.inc_constrs):
                # inc_Gub = np.zeros(G_ub.shape[1])
                inc_Gub[i][inc_con.idx] = inc_con.G_ub_var
                inc_hub[i] = inc_con.h_ub_var
            if self.G_ub is None:
                return inc_Gub,inc_hub
            G_ub = np.append(self.G_ub, inc_Gub, axis=0)
            h_ub = np.append(self.h_ub, inc_hub, axis=0)
            return G_ub, h_ub

    def addnode(self,last_node:BBNode,new_node:BBNode):
        last_node.addnext(new_node)
        new_node.addlast(last_node)

    def get_node_id(self):
        self.id_counter += 1
        return self.id_counter

    def solve(self):
        while len(self.open_nodes)>0:
            
            if len(self.z_lps) > 1000:
                with open(self.save_path, 'r') as file:
                    data = json.load(file)
                    data['z_lps'].extend(self.z_lps)
                    data['lower_bounds'].extend(self.lower_bounds)
                    self.z_lps.clear()
                    self.lower_bounds.clear()
                with open(self.save_path, 'w') as file:
                    json.dump(data, file)
            # print('Queue: ',[f'node{bbnode.node_id}' for bbnode in self.Q.queue])
           
            # 取出当前问题
            # cur_node = self.Q.get(block=False)
            
            # cur_node = self.open_nodes.pop(0)
            cur_node = random.choice(self.open_nodes)
            self.open_nodes.remove(cur_node)
            
            # 当前最优值小于总下界，则排除此区域 （剪枝）
            if -cur_node.z < self.LOWER_BOUND:
                continue

            # 若结果 x 中全为整数，则尝试更新全局下界、全局最优值和最优解（定界）
            
            # if all(list(map(lambda f: f.is_integer(), cur_node.x))):
            if self.all_integer(cur_node):
                if self.LOWER_BOUND < -cur_node.z:
                    self.LOWER_BOUND = -cur_node.z
                    self.lower_bounds.append((cur_node.node_id, -cur_node.z))

                if self.opt_val is None or self.opt_val < -cur_node.z:
                    self.opt_val = -cur_node.z
                    self.opt_x = cur_node.x
                    self.opt_node = cur_node
                    # print(self.LOWER_BOUND, self.opt_val, self.UPPER_BOUND)
                    print(f"prblm_path:{self.prblm_path}\n"
                          f'update lowerbound! lowerbound{self.LOWER_BOUND}, '
                          f'opt_val: {self.opt_val}, opt_x: {self.opt_x}, '
                          f'opt_node: {cur_node.node_id}')
                continue

            # if cur_node.z.is_integer():
            #     print(f"prblm_path:{self.prblm_path}\n"
            #           f"z_lp is integer {cur_node.z} with x{cur_node.x}")

            # 进行分枝
            else:
                # 随机选取不是整数的x
                nonint_idx = [idx for idx, xx in enumerate(cur_node.x)
                      if not xx.is_integer() and idx not in self.conti_vars]
                idx = random.choice(nonint_idx)
                # for idx, x in enumerate(cur_node.x):
                #     if not x.is_integer():
                #         break

                # 构建新的约束条件
                inc_constr1 = copy.deepcopy(cur_node.inc_constrs)
                inc_constr2 = copy.deepcopy(cur_node.inc_constrs)
                inc_constr1.append(IncConstr(idx, -1, -math.ceil(cur_node.x[idx])))
                inc_constr2.append(IncConstr(idx, 1, math.floor(cur_node.x[idx])))
                node1 = BBNode(self.get_node_id(), inc_constrs = inc_constr1)
                node2 = BBNode(self.get_node_id(), inc_constrs = inc_constr2)
                Gub1,hub1 = self.get_Gub_hub(node1)
                Gub2,hub2 = self.get_Gub_hub(node2)
                # new_con1 = np.zeros(cur_node.A_ub.shape[1])
                # new_con1[idx] = -1
                # new_con2 = np.zeros(cur_node.A_ub.shape[1])
                # new_con2[idx] = 1
                # new_A_ub1 = np.append(cur_node.A_ub, new_con1[np.newaxis,:], axis=0)
                # new_A_ub2 = np.append(cur_node.A_ub, new_con2[np.newaxis,:], axis=0)
                # new_b_ub1 = np.append(cur_node.b_ub, np.array([-math.ceil(cur_node.x[idx])]), axis=0)
                # new_b_ub2 = np.append(cur_node.b_ub, np.array([math.floor(cur_node.x[idx])]), axis=0)

                # 将新约束条件加入队列，先加最优值大的那一支               
                r1 = linprog(self.c, Gub1, hub1, self.A_eq, self.b_eq, self.bounds)
                r2 = linprog(self.c, Gub2, hub2, self.A_eq, self.b_eq, self.bounds)

                node1.x = r1.x
                node1.z = r1.fun
                node2.x = r2.x
                node2.z = r2.fun

                # print(f"node:{node1.node_id}, z_lp:{node1.z}")
                # print(f"node:{node2.node_id}, z_lp:{node2.z}")

                
                self.addnode(cur_node,node1)
                self.addnode(cur_node,node2)

                if r1.success:
                    self.open_nodes.insert(0, node1)
                    # self.z_lps.append((node1.node_id, r1.fun))
                    self.z_lps.append(r1.fun)
                if r2.success:
                    self.open_nodes.insert(0, node2)
                    # self.z_lps.append((node2.node_id, r2.fun))
                    self.z_lps.append(r2.fun)
                # print(self.LOWER_BOUND, self.opt_val, self.UPPER_BOUND)
        with open(self.save_path, 'r') as file:
            data = json.load(file)
            data['z_lps'].extend(self.z_lps)
            data['lower_bounds'].extend(self.lower_bounds)
            
            if self.opt_val is not None:
                data.update({'opt_val': self.opt_val,
                            'opt_x': self.opt_x.tolist(),
                            'opt_node': cur_node.node_id})
            else:
                data['opt_val']="Not Found"

            data["node_num"] = self.get_node_id()
        
        with open(self.save_path, 'w') as file:
            json.dump(data, file)

    def show_solvetree(self):  # 按从上到下从左到右展示block,打印块名
        print('\n>>>>>>>>>>>> show solve tree >>>>>>>>>>>> ')
        if not self.head:
            print('Empty')
            return
        q = [self.head]
        nodelist = []
        while q:
            node = q.pop(0)
            nodelist.append(node)
            node.shownode()
            for i in node.next:
                q.append(i)
        if self.opt_node:
            print(f'opt_val: {self.opt_val}, opt_x: {self.opt_x}, opt_node: {self.opt_node.node_id}\n')
        print('>>>>>>>>>>>> solve tree end >>>>>>>>>>>>\n')




def test1():
    """ 此测试的真实最优解为 [4, 2] """
    c = np.array([40, 90])
    A = np.array([[9, 7], [7, 20]])
    b = np.array([56, 70])
    Aeq = None
    beq = None
    bounds = [(0, None), (0, None)]

    solver = BranchandBound(c, A, b, Aeq, beq, bounds)
    solver.solve()
    # solver.show_solvetree()
    print("Test 1's result:", solver.opt_val, solver.opt_x)


def test2():
    """ 此测试的真实最优解为 [2, 4] """
    c = np.array([3, 13])
    A = np.array([[2, 9], [11, -8]])
    b = np.array([40, 82])
    Aeq = None
    beq = None
    bounds = [(0, None), (0, None)]

    solver = BranchandBound(c, A, b, Aeq, beq, bounds)
    solver.solve()
    solver.show_solvetree()
    print("Test 2's result:", solver.LOWER_BOUND, solver.opt_val, solver.opt_x)


def test3():
    c = np.array([4, 3])
    A = np.array([[4, 5], [2, 1]])
    b = np.array([20, 6])
    Aeq = None
    beq = None
    bounds = [(0, None), (0, None)]

    solver = BranchandBound(c, A, b, Aeq, beq, bounds)
    solver.solve()
    solver.show_solvetree()
    print("Test 3's result:", solver.opt_val, solver.opt_x, solver.LOWER_BOUND)

def test4():
    c = np.array([4, 3])
    A = np.array([[3, 4], [4, 2]])
    b = np.array([12, 9])
    Aeq = None
    beq = None
    bounds = [(0, None), (0, None)]

    solver = BranchandBound(c, A, b, Aeq, beq, bounds)
    solver.solve()
    solver.show_solvetree()
    print("Test 4's result:", solver.opt_val, solver.opt_x, solver.LOWER_BOUND)

@get_time
def test_tsp():
    with open("testTSP\problem poolburma14.json", "r") as f:
        prblm_json = f.read().split('\n')[0]
        prblm = json.loads(prblm_json)
        c = np.array(prblm['c'])
        A = np.array(prblm['G_ub'])
        b = np.array(prblm['h_ub'])
        Aeq = np.array(prblm['A_eq'])
        beq = np.array(prblm['b_eq'])
        bounds = [tuple(bound) for bound in prblm['bounds']]

        solver = BranchandBound(c, A, b, Aeq, beq, bounds)
        solver.solve()
        # solver.show_solvetree()
        print("Test 1's result:", solver.opt_val, solver.opt_x)


def solve_ilp_by_pulp(c, G_ub, h_ub, A_eq, b_eq, bounds, conti_vars):
    """
    使用`pulp`求解整数规划问题
    :return: 最优目标值，如果不可解返回None
    """
    print("Solving by pulp...")
    ilp = pulp.LpProblem("ILP", pulp.LpMaximize)
    # vars = [pulp.LpVariable(f'x_{i//48}_{i%48}', lowBound=b[0], upBound=b[1], 
    #         cat=pulp.LpInteger) for i, b in enumerate(bounds)]
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
    v_names = []
    if ilp.status != pulp.LpStatusOptimal:
        return None
    # for i, v in enumerate(ilp.variables()):
    #     print(v.name, "=", v.varValue)
    #     v_names.append(v.name)
    obj_value = pulp.value(ilp.objective)
    print("PULP Optimal value:", obj_value)
    return obj_value

def test(prblm_path):
    try:
        with open(prblm_path, "r") as f:
            prblm_json = f.read().split('\n')[0]
            prblm = json.loads(prblm_json)
            c = np.array(prblm['c'])
            A = np.array(prblm['G_ub'])
            b = np.array(prblm['h_ub'])
            if 'A_eq' in prblm.keys():
                Aeq = np.array(prblm['A_eq'])
                beq = np.array(prblm['b_eq'])
            else:
                Aeq = None
                beq = None
            # Aeq = np.array(prblm['A_eq'])
            # beq = np.array(prblm['b_eq'])
            if 'bounds' in prblm.keys():
                bounds = [tuple(bound) for bound in prblm['bounds']]
            else:
                if 'type' in prblm.keys() and prblm['type'] == ZERO_ONE:
                    bounds = [(0, 1) for _ in range(len(c))]
                else:
                    bounds = [(0, None) for _ in range(len(c))]

            solver = BranchandBound(c, A, b, Aeq, beq, bounds,prblm_path=prblm_path)
            if 'conti_vars' in prblm.keys():
                solver.conti_vars = prblm['conti_vars']
            
            solver.solve()
            # solver.show_solvetree()
            print("Test 1's result:", solver.opt_val, solver.opt_x)
    except Exception:
        print(traceback.print_exc())
        # 遇到错误，跳过当前迭代并保存错误信息
        # ERROR_PATH = RESULT_PATH
        # with open(ERROR_PATH / f"error{time.time()}.txt", "w+") as f:
        # traceback.print_exc()
        print("Fatal Error! Terminate!")


if __name__ == '__main__':
    threadNum = 1
    worker_pool = mp.Pool(threadNum)
    print(threadNum)
    res = []
    pool_paths = []
    folder = Path("testMIPLIB2")
    for file_path in folder.glob('*'):
        pool_paths.append(file_path)
    print(pool_paths)
    for pool_path in pool_paths:
        res.append(worker_pool.apply_async(test, [pool_path]))
    while worker_pool._cache:
        print("number of jobs pending: ", len(worker_pool._cache))
        time.sleep(1)
    for r in res:
        r.wait()
    print('Waiting for all subprocesses done...')
    worker_pool.close()
    worker_pool.join()
    print('All subprocesses done.')
    # test("testMIPLIB2\int3_conti0_ub0_eq1_ej.json")