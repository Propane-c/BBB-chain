import configparser
import logging
import multiprocessing as mp
import time
import traceback
from pathlib import Path

from branchbound import bb_consensus as bb
import data.lpprblm as lpprblm
import simulation
from background import RESULT_PATH, Background
from environment import Environment
from data.lpprblm import NORMAL, ZERO_ONE

# import drawing


def get_time(f):
    def inner(*arg, **kwarg):
        s_time = time.time()
        res = f(*arg, **kwarg)
        e_time = time.time()
        print('耗时：{}秒'.format(e_time - s_time))
        return res

    return inner

def load_config():
    # 读取配置文件
    config = configparser.ConfigParser()
    config.optionxform = lambda option: option
    config.read('system_config.ini', encoding='utf-8')
    environ_settings = dict(config['EnvironmentSettings'])
    return config, environ_settings

def set_background(environ_settings):
    # 设置全局变量
    background = Background()
    background.set_consensus_type(environ_settings['consensus_type'])
    background.set_network_type(environ_settings['network_type'])
    # global_var.set_miner_num(int(environ_settings['miner_num']))
    background.set_ave_q(int(environ_settings['q_ave']))
    background.set_blocksize(int(environ_settings['blocksize']))
    background.set_show_fig(False)
    background.set_total_round(int(environ_settings['total_round']))
    return background

def set_logger(context: Background, log_level = logging.ERROR):
    """设置日志"""
    logging.basicConfig(
        filename = context.get_result_path() / 'events.log',
        level=log_level, filemode='w')

def set_network_param(config:configparser.ConfigParser, environ_settings):
    # 设置网络参数
    network_param = {}
    if environ_settings['network_type'] == 'network.TopologyNetwork':
        net_setting = 'TopologyNetworkSettings'
        bool_params = ['show_label', 'save_routing_graph']
        float_params = ['ave_degree', 'bandwidth_honest', 'bandwidth_adv']
        for bparam in bool_params:
            network_param.update({bparam: config.getboolean(net_setting, bparam)})
        for fparam in float_params:
            network_param.update({fparam: config.getfloat(net_setting, fparam)})
        network_param = {'TTL': config.getint(net_setting, 'TTL'),
                        'gen_net_approach': config.get(net_setting, 'gen_net_approach')}
    elif environ_settings['network_type'] == 'network.BoundedDelayNetwork':
        net_setting = 'BoundedDelayNetworkSettings'
        network_param = {k: float(v) for k, v in dict(config[net_setting]).items()}
    return network_param


@get_time
def run(pool_path=None):
    """
    单次运行入口
    """
    
    config, environ_settings = load_config()
    background = set_background(environ_settings)
    set_logger(background, logging.ERROR)
    network_param = set_network_param(config, environ_settings)
    # t = int(environ_settings['t'])

    q_ave = int(environ_settings['q_ave'])
    q_distr = environ_settings['q_distr']
    target = environ_settings['target']
    # adversary_ids = eval(environ_settings['adversary_ids'])
    adversary_ids = ()
    t = len(adversary_ids)
    total_round = int(environ_settings['total_round'])
    background.set_keyblock_strategy('pow')
    background.set_var_num('maxsat')
    background.set_solve_prob(0.5)
    background.set_safe_thre(1)
    background.set_miner_num(1)
    background.set_bb_difficulty(3)
    background.set_openblock_strategy(bb.OB_RAND)
    background.set_openprblm_strategy(bb.OP_RAND)
    if pool_path is not None:
        pool_path = Path(pool_path)
        background.set_result_path(background.get_result_path() / pool_path.stem)
    else:
        # pool_path = Path.cwd()/"Problem Pools"/"01"/f"problem pool{background.get_var_num()}.json"
        # pool_path = Path.cwd()/"Problem Pools\\fig1.json"
        # pool_path = Path.cwd()/"testMAXSAT\problem poolkbtree-kbtree9_7_3_5_80_1_1225.json"
        # pool_path = "E:\Files\A-blockchain\\branchbound\MAXSAT\json\problem poolvar162_pseudoBoolean-normalized-g9x9.opb.msat.wcnf.json"
        pool_path = Path.cwd()/"Problem Pools\\testTSP\problem poolburma14.json"
        # pool_path = Path.cwd()/"testMIPLIB2\\int24_conti24_ub24_eq10_gr4x6.json"
        # pool_path = Path.cwd()/"testMAXSAT\\var162_soft81_con162_pseudoBoolean-normalized-g9x9.opb.msat.json"

    prblm_pool = lpprblm.load_prblm_pool_from_json(pool_path)
    lp = prblm_pool[0]
    # lp = lpprblm.load_prblm_pool_from_json(
    #     ".\Problem Pools\\problem pool1007_1805.json")[0]
    # lp = lpprblm.load_prblm_pool_from_json(
    #     ".\Problem Pools\\1109\problem pool1109_1508.json")[0]
    # lp = lpprblm.load_prblm_pool_from_json(
    #     ".\Problem Pools\\1116\problem pool1116_105207.json")[0]
    # lp = lpprblm.load_prblm_pool_from_json(
    #     ".\Results\\20231215\\211531\problem poolprblm 1215_211531.json")[0]
    # lp =  lpprblm.prblm_generator(background.get_var_num(), ZERO_ONE, 5)
    # lpprblm.save_test_prblm_pool([lp], time.strftime("%m%d_%H%M%S"), 
    #     Path.cwd()/"Problem Pools"/time.strftime("%m%d"))
    # lpprblm.save_test_prblm_pool([lp], f'prblm {time.strftime("%m%d_%H%M%S")}', 
    #     background.get_result_path())
    # lp = lpprblm.test5()
    background.set_test_prblm(lp)
    quiet=False
    Z = Environment(background, t, q_ave, q_distr, target, 
                    adversary_ids, network_param, None, True)
    # Z.env_load_prblm_pool([lp, lp2])
    total_round = Z.exec(quiet=quiet)
    
    print(total_round)
    Z.view(quiet=quiet, pool_path=pool_path)
    # if len(Z.evaluation.mb_nums.keys()) == 0:
    #     return False
    # if Z.evaluation.mb_nums['B0'] >= 5:
    #     lpprblm.save_test_prblm_pool([lp], f'prblm {time.strftime("%m%d_%H%M%S")}', 
    #         background.get_result_path())
    #     Z.view(quiet=False)
    #     return True
    # print("mbNum: ", Z.evaluation.mb_nums['B0'])
    return False
    
    

def single_process_shortchain(
        repeat_num, iter_num, var_num, difficulties,
        miner_nums, adversary_num, prblm_pool_method, 
        safe_thre = 0.001, 
        record_block_times = False, 
        opblk_st:str = bb.OB_RAND, 
        opprblm_st:str = bb.OP_RAND, 
        
        solve_prob = 0.5):
    """ 短链仿真（单链只包含一个问题）"""
    try:
        _, environ_settings = load_config()
        background = set_background(environ_settings)
        set_logger(background)
        simulation.short_simulation(
            background, repeat_num, iter_num, var_num, difficulties, 
            miner_nums, adversary_num, prblm_pool_method, 
            record_block_times, safe_thre, solve_prob, opblk_st, 
            opprblm_st)
    except Exception:
        print(traceback.print_exc())
        # 遇到错误，跳过当前迭代并保存错误信息
        ERROR_PATH = background.get_result_path()
        with open(ERROR_PATH / f"error{time.time()}.txt", "w+") as f:
            traceback.print_exc(file = f)
        print("Fatal Error! Terminate!")
        
def single_run(pool_path):
    try:
        run(pool_path)
    except Exception:
        print(traceback.print_exc())
        # 遇到错误，跳过当前迭代并保存错误信息
        ERROR_PATH = RESULT_PATH
        with open(ERROR_PATH / f"error{time.time()}.txt", "w+") as f:
            traceback.print_exc(file = f)
        print("Fatal Error! Terminate!")

    
def single_process_longchain(
        iter_num, kb_strategy, safe_thre, 
        difficulty, var_num, miner_num, 
        prblm_pool_method, pool_size):
    """ 长链仿真（单链包含多个问题）"""
    _, environ_settings = load_config()
    background = set_background(environ_settings)
    set_logger(background)
    simulation.simu_mbkb_forkrate_longchain(
        background, iter_num, kb_strategy, 
        safe_thre, difficulty, var_num, miner_num, 
        prblm_pool_method, pool_size)

if __name__ == '__main__':
    """参数设置"""
    simu_type = "single_run"
    # simu_type = "long"
    # simu_type = "short"
    multiProcessOn = False
    threadNum = 2
    """
    short参数说明：
    repeat_num, iter_num, var_num, difficulties, miner_nums, adversary_num, 
    prblm_pool_method, opblk_st, opprblm_st, 
    safe_thre, record_block_times, solve_prob
    """
    rpt_num1 = 1
    rpt_num = 5
    args_list = [
        # [rpt_num, 2500, 100, [3],  [20],0, 'load'],
        # [rpt_num, 2500, 100, [5],  [20],0, 'load'],
        # [rpt_num, 2500, 100, [8],  [20],0, 'load'],
        # [rpt_num, 2500, 100, [10], [20],0, 'load'],
        
        # [rpt_num, 2500, 100, [8],  [10],0, 'load'],
        # [rpt_num, 2500, 100, [10], [10],0, 'load'],
        # [rpt_num, 2500, 100, [3],  [10],0, 'load'],
        # [rpt_num, 2500, 100, [5],  [10],0, 'load'],

        # [rpt_num, 2500, 100, [3],  [1],0, 'load'],
        # [rpt_num, 2500, 100, [5],  [1],0, 'load'],
        # [rpt_num, 2500, 100, [8],  [1],0, 'load'],
        # [rpt_num, 2500, 100, [10], [1],0, 'load'],

        # [rpt_num, 2500, 100, [3],  [3],0, 'load'],
        # [rpt_num, 2500, 100, [5],  [3],0, 'load'],
        # [rpt_num, 2500, 100, [8],  [3],0, 'load'],
        # [rpt_num, 2500, 100, [10], [3],0, 'load'],
        
        [rpt_num1, 1, 100, [3],  [3],0, 'load', 0.001, True],
        [rpt_num1, 1, 100, [5],  [3],0, 'load', 0.001, True],
        [rpt_num1, 1, 100, [7],  [3],0, 'load', 0.001, True],
        [rpt_num1, 1, 100, [9],  [3],0, 'load', 0.001, True],

        [rpt_num, 1, 100, [9],  [3],0, 'load', 0.001, True],
        [rpt_num, 1, 100, [3],  [10],0, 'load', 0.001, True],
        [rpt_num, 1, 100, [5],  [10],0, 'load', 0.001, True],
        [rpt_num, 1, 100, [7],  [10],0, 'load', 0.001, True],
        [rpt_num, 1, 100, [9],  [10],0, 'load', 0.001, True],

        # [rpt_num, 1500, 50, [3],  [20], 0, 'load'],
        # [rpt_num, 1500, 50, [5],  [20], 0, 'load'],
        # [rpt_num, 1500, 50, [8],  [20], 0, 'load'],
        # [rpt_num, 1500, 50, [10], [20], 0, 'load'],

        # [rpt_num, 1500, 50, [3],  [15], 0, 'load'],
        # [rpt_num, 1500, 50, [5],  [15], 0, 'load'],
        # [rpt_num, 1500, 50, [8],  [15], 0, 'load'],
        # [rpt_num, 1500, 50, [10], [15], 0, 'load'],

        # [rpt_num, 1500, 50, [3],  [5], 0, 'load', True],
        # [rpt_num, 1500, 50, [5],  [5], 0, 'load', True],
        # [rpt_num, 1500, 50, [8],  [5], 0, 'load', True],
        # [rpt_num, 1500, 50, [10], [5], 0, 'load', True],
 
        # [rpt_num, 1500, 50, [3],  [10], 0, 'load'],
        # [rpt_num, 1500, 50, [5],  [10], 0, 'load'],
        # [rpt_num, 1500, 50, [8],  [10], 0, 'load'],
        # [rpt_num, 1500, 50, [10], [10], 0, 'load'],

        # [rpt_num, 1500, 50, [3],  [3], 0, 'load'],
        # [rpt_num, 1500, 50, [5],  [3], 0, 'load'],
        # [rpt_num, 1500, 50, [8],  [3], 0, 'load'],
        # [rpt_num, 1500, 50, [10], [3], 0, 'load'],

        # [rpt_num, 1500, 50, [3],  [1], 0, 'load'],
        # [rpt_num, 1500, 50, [5],  [1], 0, 'load'],
        # [rpt_num, 1500, 50, [8],  [1], 0, 'load'],
        # [rpt_num, 1500, 50, [10], [1], 0, 'load'],

        # [rpt_num, 1500, 20, [3],  [1], 0, 'load'],
        # [rpt_num, 1500, 20, [5],  [1], 0, 'load'],
        # [rpt_num, 1500, 20, [8],  [1], 0, 'load'],
        # [rpt_num, 1500, 20, [10], [1], 0, 'load'],

        # [rpt_num, 1500, 20, [3],  [3], 0, 'load'],
        # [rpt_num, 1500, 20, [5],  [3], 0, 'load'],
        # [rpt_num, 1500, 20, [8],  [3], 0, 'load'],
        # [rpt_num, 1500, 20, [10], [3], 0, 'load'],

        # [rpt_num, 1500, 20, [3],  [5], 0, 'load', True],
        # [rpt_num, 1500, 20, [5],  [5], 0, 'load', True],
        # [rpt_num, 1500, 20, [8],  [5], 0, 'load', True],
        # [rpt_num, 1500, 20, [10], [5], 0, 'load', True],
 
        # [rpt_num, 1500, 20, [3],  [10], 0, 'load'],
        # [rpt_num, 1500, 20, [5],  [10], 0, 'load'],
        # [rpt_num, 1500, 20, [8],  [10], 0, 'load'],
        # [rpt_num, 1500, 20, [10], [10], 0, 'load'],

        # [rpt_num, 1500, 20, [3],  [15], 0, 'load'],
        # [rpt_num, 1500, 20, [5],  [15], 0, 'load'],
        # [rpt_num, 1500, 20, [8],  [15], 0, 'load'],
        # [rpt_num, 1500, 20, [10], [15], 0, 'load'],

        # [rpt_num, 1500, 20, [3],  [20], 0, 'load'],
        # [rpt_num, 1500, 20, [5],  [20], 0, 'load'],
        # [rpt_num, 1500, 20, [8],  [20], 0, 'load'],
        # [rpt_num, 1500, 20, [10], [20], 0, 'load'],

        # [rpt_num, 50, 50, [5], [5],   0, 'load', bb.OB_RAND, bb.OP_BEST],
        # [rpt_num, 10, 50, [5], [10],  0, 'load', bb.OB_RAND, bb.OP_BEST],
        # [rpt_num, 10, 50, [5], [15],  0, 'load', bb.OB_RAND, bb.OP_BEST],
        # [rpt_num, 10, 50, [5], [20],  0, 'load', bb.OB_RAND, bb.OP_BEST],
        # [rpt_num, 10, 50, [5], [30],  0, 'load', bb.OB_RAND, bb.OP_BEST],
        # [rpt_num, 10, 50, [5], [1,3], 0, 'load', bb.OB_RAND, bb.OP_BEST],

        # [rpt_num, 50, 50, [5], [5],  0, 'load', bb.OB_DEEP, bb.OP_RAND],
        # [rpt_num, 50, 50, [5], [10],  0, 'load', bb.OB_DEEP, bb.OP_RAND],
        # [rpt_num, 50, 50, [5], [15],  0, 'load', bb.OB_DEEP, bb.OP_RAND],
        # [rpt_num, 50, 50, [10], [5],  0, 'load', bb.OB_DEEP, bb.OP_RAND],
        # [rpt_num, 50, 50, [10], [10],  0, 'load', bb.OB_DEEP, bb.OP_RAND],
        # [rpt_num, 10, 50, [5], [1,3], 0, 'load', bb.OB_DEEP, bb.OP_RAND],

        # [rpt_num, 50, 50, [5], [5],   0, 'load', bb.OB_RAND, bb.OP_RAND],
        # [rpt_num, 10, 50, [5], [10],  0, 'load', bb.OB_RAND, bb.OP_RAND],
        # [rpt_num, 10, 50, [5], [15],  0, 'load', bb.OB_RAND, bb.OP_RAND],
        # [rpt_num, 10, 50, [5], [20],  0, 'load', bb.OB_RAND, bb.OP_RAND],
        # [rpt_num, 10, 50, [5], [30],  0, 'load', bb.OB_RAND, bb.OP_RAND],
        # [rpt_num, 10, 50, [5], [1,3], 0, 'load', bb.OB_RAND, bb.OP_RAND],

        # [rpt_num, 50, 50, [5], [5],   0, 'load', bb.OB_BREATH, bb.OP_RAND],
        # [rpt_num, 10, 50, [5], [10],  0, 'load', bb.OB_BREATH, bb.OP_RAND],
        # [rpt_num, 10, 50, [5], [15],  0, 'load', bb.OB_BREATH, bb.OP_RAND],
        # [rpt_num, 10, 50, [5], [20],  0, 'load', bb.OB_BREATH, bb.OP_RAND],
        # [rpt_num, 10, 50, [5], [30],  0, 'load', bb.OB_BREATH, bb.OP_RAND],
        # [rpt_num, 10, 50, [5], [1,3], 0, 'load', bb.OB_BREATH, bb.OP_RAND],

        # [rpt_num, 10, 50, [3,5], [5], 0,'load' ],
        # [rpt_num, 10, 50, [3,5], [10],0, 'load'],
        # [rpt_num, 10, 50, [3,5], [15],0, 'load'],
        # [rpt_num, 10, 50, [3,5], [20],0, 'load'],
        # [rpt_num, 20, 50, [3], [1,3,30],0, 'load'],
        # [rpt_num, 10, 50, [5], [1,3,30],0, 'load'],

        # [rpt_num, 10, 50, [8], [5], 0,'load'],
        # [rpt_num, 10, 50, [8], [10],0, 'load'],
        # [rpt_num, 10, 50, [8], [15],0, 'load'],
        # [rpt_num, 10, 50, [8], [20],0, 'load'],
        # [rpt_num, 10, 50, [8], [1,3,30],0, 'load'],

        # [rpt_num, 10, 50, [10], [5], 0, 'load'],
        # [rpt_num, 10, 50, [10],[10], 0, 'load'],
        # [rpt_num, 10, 50, [10],[15], 0, 'load'],
        # [rpt_num, 10, 50, [10],[20], 0, 'load'],
        # [rpt_num, 10, 50, [10], [1,3,30], 0, 'load'],

        # [rpt_num, 1500, 50, [3,4],   [20], 1, 'load', 0.005 ],
        # [rpt_num, 1500, 50, [3,4],   [20], 1, 'load', 0.001],
        # [rpt_num, 1500, 50, [3,4],   [20], 1, 'load', 0.0005],
        # [rpt_num, 1500, 50, [3,4],   [20], 1, 'load', 0.0001],

        # [rpt_num, 1500, 50, [5,6],   [20], 1, 'load', 0.005 ],
        # [rpt_num, 1500, 50, [5,6],   [20], 1, 'load', 0.001],
        # [rpt_num, 1500, 50, [5,6],   [20], 1, 'load', 0.0005],
        # [rpt_num, 1500, 50, [5,6],   [20], 1, 'load', 0.0001],

        # [rpt_num, 1500, 50, [7,8],   [20], 1, 'load', 0.005],
        # [rpt_num, 1500, 50, [7,8],   [20], 1, 'load', 0.001],
        # [rpt_num, 1500, 50, [7,8],   [20], 1, 'load', 0.0005],
        # [rpt_num, 1500, 50, [7,8],   [20], 1, 'load', 0.0001],

        # [rpt_num, 1500, 50, [9,10], [20], 1, 'load', 0.005],
        # [rpt_num, 1500, 50, [9,10], [20], 1, 'load', 0.001],
        # [rpt_num, 1500, 50, [9,10], [20], 1, 'load', 0.0005],
        # [rpt_num, 1500, 50, [9,10], [20], 1, 'load', 0.0001],

        # [rpt_num, 1500, 50, [11], [20], 1, 'load', 0.005],
        # [rpt_num, 1500, 50, [11], [20], 1, 'load', 0.001],
        # [rpt_num, 1500, 50, [11], [20], 1, 'load', 0.0005],
        # [rpt_num, 1500, 50, [11], [20], 1, 'load', 0.0001],

        # [rpt_num, 1, 50, [5],   [20], 1, 'load', 0.005],
        # [rpt_num, 1, 50, [5],   [20], 1, 'load', 0.003],
        # [rpt_num, 1, 50, [5],   [20], 1, 'load', 0.001],
        # [rpt_num, 1, 50, [5],   [20], 1, 'load', 0.0008],
        # [rpt_num, 1, 50, [5],   [20], 1, 'load', 0.0005],
        # [rpt_num, 1, 50, [5],   [20], 1, 'load', 0.0003],
        # [rpt_num, 1, 50, [5],   [20], 1, 'load', 0.0001],

    ]
    """
    long参数说明：
    iter_num, kb_strategy, dmin, difficulty, var_num, miner_num, 
    prblm_pool_method, pool_size
    """
    long_chian_args = [
        # [1, 'withmini', 1, 5, 20, 10, 'load', 1000],
        # [1, 'withmini', 1, 4, 20, 10, 'load', 1],
        # [1, 'withmini', 1, 5, 20, 10, 'load', 1],
        # [1, 'withmini', 1, 6, 20, 10, 'load', 1],
        # [2, 'withmini', 1, 7, 20, 10, 'load', 5],
        # [2, 'withmini', 1, 8, 20, 10, 'load', 5],
        # [2, 'withmini', 1, 9, 20, 10, 'load', 5],
        # [1, 'pow+withmini', 1, 5, 20, 10, 'load', 1000],
        # [1, 'pow+withmini', 1, 4, 20, 10, 'load', 1000],
        # # [1, 'pow+withmini', 1, 5, 20, 10, 'load', 1000],
        # [2, 'pow+withmini', 1, 6, 20, 10, 'load', 5],
        # [2, 'pow+withmini', 1, 7, 20, 10, 'load', 5],
        # [2, 'pow+withmini', 1, 8, 20, 10, 'load', 5],
        # [2, 'pow+withmini', 1, 9, 20, 10, 'load', 5],
        [10, 'pow', 1, 3, 20, 10, 'load', 1000],
        [10, 'pow', 1, 4, 20, 10, 'load', 1000],
        # # [1, 'pow', 1, 5, 20, 10, 'load', 1000],
        # [2, 'pow', 1, 6, 20, 10, 'load', 5],
        [10, 'pow', 1, 7, 20, 10, 'load', 1000],
        # [2, 'pow', 1, 8, 20, 10, 'load', 5],
        # [2, 'pow', 1, 9, 20, 10, 'load', 5],
    ]
    """仿真入口 """
    if multiProcessOn is False:
        if simu_type == "single_run":
            """单次运行测试"""
        
            i = 0
            # while not run():
            #     i+=1
            #     print(i)
            run()    
                      
    else:
        """多进程仿真"""
        worker_pool = mp.Pool(threadNum)
        print(threadNum)
        res = []
        if simu_type == "single_run":
            pool_paths = []
            folder = Path("testMIPLIB2")
            for file_path in folder.glob('*'):
                pool_paths.append(file_path)
            print(pool_paths)
            for pool_path in pool_paths:
                res.append(worker_pool.apply_async(single_run, [pool_path]))
        if simu_type == "short":
            for args in args_list:
                if not multiProcessOn:
                    single_process_shortchain(*args)
                    continue
                print(args)
                res.append(worker_pool.apply_async(single_process_shortchain, args))
        if simu_type == "long":
            for long_arg in long_chian_args:
                single_process_longchain(*long_arg)
                res.append(worker_pool.apply_async(single_process_longchain, long_arg))
        
        while worker_pool._cache:
            print("number of jobs pending: ", len(worker_pool._cache))
            time.sleep(20)
        for r in res:
            r.wait()
        print('Waiting for all subprocesses done...')
        worker_pool.close()
        worker_pool.join()
        print('All subprocesses done.')
