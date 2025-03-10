import os
from pathlib import Path
import xml.dom.minidom
import xml.etree.ElementTree as ET
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import data.lpprblm as lp

def load_exist_tsp(load_file_path=None):
    G, pos, n, distance_matrix = read_tsp_from_xml(load_file_path)
    draw_tsp_graph(G, pos, os.path.dirname(load_file_path))
    orig_prblm = get_tsp_lpprblm(n, distance_matrix)
    lp.solve_lp(orig_prblm)
    orig_prblm.fathomed = False
    orig_prblm.fthmd_state = False 
    return orig_prblm

def gen_random_tsp(node_num, save_dir=None):
    if save_dir is not None and not os.path.exists(save_dir):
        save_dir.mkdir(parents=True, exist_ok=True)
    G, pos,_,distance_matrix = random_tsp_save_xml(node_num, save_dir)
    draw_tsp_graph(G, pos, save_dir)
    orig_prblm = get_tsp_lpprblm(node_num, distance_matrix)
    lp.solve_lp(orig_prblm)
    orig_prblm.fathomed = False
    orig_prblm.fthmd_state = False 
    # solve_ilp_by_pulp(orig_prblm)
    return orig_prblm

def get_tsp_lpprblm(n, distance_matrix):
    c = np.array(distance_matrix).flatten()
    A_eq = np.zeros((2 * n, n ** 2))
    for i in range(n):
        A_eq[i, i*n:(i+1)*n] = 1  # leaving
        A_eq[n + i, i::n] = 1     # entering
    new_A_eq = np.zeros((2 * n, n ** 2 + n))
    new_A_eq[:, :n ** 2] = A_eq
    b_eq = np.ones(2 * n)
    bounds = [(0, 1) for _ in range(n ** 2)]
    for i in range(n):
        bounds[i*n + i] = (0, 0)
    # 子循环排除辅助变量u_i
    c = np.append(c, [0 for _ in range(n)])
    u_bounds = [(1, n-1) for _ in range(n)]
    bounds.extend(u_bounds)
    # 构造子循环排除约束
    num_subtour_constraints = (n-1) * (n-1)
    G_ub = np.zeros((num_subtour_constraints, n**2 + n))
    h_ub = np.full(num_subtour_constraints, n - 1)
    k = 0
    for i in range(1, n):
        for j in range(1, n):
            if i == j:
                continue
            G_ub[k, i*n + j] = n
            G_ub[k, n**2 + i] = 1
            G_ub[k, n**2 + j] = -1
            k += 1
    orig_prblm = lp.LpPrblm(((0, 0), 0), None, 0, c, G_ub, h_ub, new_A_eq, b_eq, bounds)
    
    # 设置初始解 - 简单地按顺序访问所有节点
    init_ix = np.zeros(n**2 + n)
    for i in range(n-1):
        init_ix[i*n + (i+1)] = 1 
    init_ix[(n-1)*n + 0] = 1 
    for i in range(1, n):
        init_ix[n**2 + i] = i
    init_iz = np.dot(c, init_ix)
    orig_prblm.init_ix = init_ix
    orig_prblm.init_iz = init_iz
    lp.save_prblm_pool([orig_prblm], Path.cwd() / "Problem Pools" / "testTSP", lp.TSP, True, 'burma14.json')
    return orig_prblm

def read_tsp_from_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    graph_section = root.find('graph')

    G = nx.Graph()
    pos = {}
    n = len(graph_section)
    distance_matrix = [[0 for _ in range(n)] for _ in range(n)]
    for i, vertex in enumerate(graph_section):
        x_tag = vertex.find('x')
        y_tag = vertex.find('y')
        if x_tag is not None and y_tag is not None:
            pos[i] = (float(x_tag.text), float(y_tag.text))
        for edge in vertex.findall('edge'):
            id = int(edge.text)
            cost = float(edge.attrib["cost"])
            distance_matrix[i][id] = cost
            G.add_edge(i, id, weight = cost)
    if len(pos) == 0:
        pos = nx.spring_layout(G, seed=42)
        for i, vertex in enumerate(graph_section):
            pos_x, pos_y = pos[i]
            pos_x_rounded = round(pos_x, 2)
            pos_y_rounded = round(pos_y, 2)
            x_elem = ET.SubElement(vertex, 'x')
            x_elem.text = str(pos_x_rounded)
            y_elem = ET.SubElement(vertex, 'y')
            y_elem.text = str(pos_y_rounded)
        tree.write(file_path)
    return G, pos, n, distance_matrix

def draw_tsp_graph(G, pos, save_dir):
    # 绘制图的布局
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=10, font_weight='bold')
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    output_svg_path = os.path.join(save_dir, "tsp_origin.svg")
    plt.savefig(output_svg_path, format="svg")

def random_tsp_save_xml(node_count, file_dir):
    # 创建一个空的无向图
    G = nx.Graph()
    # 添加节点
    G.add_nodes_from(range(node_count))
    pos = nx.random_layout(G)
    distance_matrix = [[0 for _ in range(node_count)] for _ in range(node_count)]
    def calculate_distance(i, j):
        x1, y1 = pos[i]
        x2, y2 = pos[j]
        x1, y1 = round(x1, 2)*100, round(y1, 2)*100
        x2, y2 = round(x2, 2)*100, round(y2, 2)*100
        return round(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5, 2)

    root = ET.Element("TSPInstance")
    name = ET.SubElement(root, "name")
    name.text = f"random_{node_count}_nodes"
    source = ET.SubElement(root, "source")
    source.text = "Generated"
    description = ET.SubElement(root, "description")
    description.text = f"{node_count}-Nodes Random TSP Problem"
    graph = ET.SubElement(root, "graph")
    for i in range(node_count):
        vertex = ET.SubElement(graph, "vertex")
        pos_x, pos_y = pos[i]
        ET.SubElement(vertex, "x").text = str(round(pos_x, 2))
        ET.SubElement(vertex, "y").text = str(round(pos_y, 2))
        for j in range(node_count):
            if i == j:
                continue
            cost = calculate_distance(i, j)
            edge = ET.SubElement(vertex, "edge", cost=f"{cost:.2e}")
            edge.text = str(j)
            if i>j:
                G.add_edge(i, j, weight = cost)
    xml_str = ET.tostring(root, encoding='UTF-8',xml_declaration=True).decode('UTF-8')
    pretty_xml_str = xml.dom.minidom.parseString(xml_str).toprettyxml(indent="  ")
    with open(file_dir / "TSP_problem.xml", "w", encoding="UTF-8") as f:
        f.write(pretty_xml_str)
    return G, pos, node_count, distance_matrix

def draw_tsp_solution(opt_x, prblm_dir = None):
    _, pos, n, _ = read_tsp_from_xml(prblm_dir / "TSP_problem.xml")
    # 遍历解向量 x，添加边到图中
    G = nx.DiGraph()
    # 添加节点
    G.add_nodes_from(range(n))
    opt_x_path = opt_x[:n * n]  # 前 n * n 项是路径变量
    opt_x_reshaped = opt_x_path.reshape((n, n))  # 重塑为 n x n 矩阵
    for i in range(n):
        for j in range(n):
            if opt_x_reshaped[i, j] == 1:  # 如果从城市 i 到城市 j 的路径被选中
                G.add_edge(i, j)
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=10, font_weight='bold')
    # 获取prblm_file_path的目录
    output_svg_path = os.path.join(prblm_dir, "tsp_solution.svg")
    plt.savefig(output_svg_path, format="svg")


def load_tsp(file_path=None):
    file_path = "E:\Files\A-blockchain\\branchbound\\tsp\\tsp\\burma14.xml"
    # file_path = "E:\Files\A-blockchain\\branchbound\\tsp\\tsp\hk48.xml"
    tree = ET.parse(file_path)
    root = tree.getroot()
    graph_section = root.find('graph')
    n = len(graph_section)
    distance_matrix = [[0 for _ in range(n)] for _ in range(n)]
    for i, vertex in enumerate(graph_section):
        for edge in vertex:
            id = int(edge.text) 
            distance_matrix[i][id] = float(edge.attrib['cost'])

    c = np.array(distance_matrix).flatten()
    A_eq = np.zeros((2 * n, n ** 2))
    for i in range(n):
        A_eq[i, i*n:(i+1)*n] = 1  # leaving
        A_eq[n + i, i::n] = 1     # entering
    new_A_eq = np.zeros((2 * n, n ** 2 + n))
    new_A_eq[:, :n ** 2] = A_eq
    b_eq = np.ones(2 * n)
    bounds = [(0, 1) for _ in range(n ** 2)]
    for i in range(n):
        bounds[i*n + i] = (0, 0)

    # 子循环排除辅助变量u_i
    c = np.append(c, [0 for _ in range(n)])
    u_bounds = [(1, n-1) for _ in range(n)]
    bounds.extend(u_bounds)
    # 构造子循环排除约束
    num_subtour_constraints = (n-1) * (n-1)
    G_ub = np.zeros((num_subtour_constraints, n**2 + n))
    h_ub = np.full(num_subtour_constraints, n - 1)
    k = 0
    for i in range(1, n):
        for j in range(1, n):
            if i == j:
                continue
            G_ub[k, i*n + j] = n
            G_ub[k, n**2 + i] = 1
            G_ub[k, n**2 + j] = -1
            k += 1
    orig_prblm = lp.LpPrblm(((0, 0), 0), None, 0, c, G_ub, h_ub, new_A_eq, b_eq, bounds)

    lp.save_prblm_pool([orig_prblm],'burma14',Path.cwd() / "testTSP", lp.TSP, True)
    # lp.solve_ilp_by_pulp(orig_prblm)