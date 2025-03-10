import numpy as np
import matplotlib.pyplot as plt
import pulp

def smart_grid_pulp():
    from pulp import LpProblem, LpMinimize, LpVariable, lpSum
    hours = 24  # 24-hour scheduling
    num_appliances = 7  # Total appliances in the system
    P_t = {
        5: np.array([1, 0.5] + [0] * (hours - 2)),  # Washing Machine: 1kWh first hour, 0.5kWh second hour
        6: np.array([0.8] + [0] * (hours - 1))  # Dishwasher: 0.8kWh for 1 hour
    }
    model = LpProblem("Home_Energy_Optimization", LpMinimize)
    x = {(a, h): LpVariable(f"x_{a}_{h}", lowBound=0) for a in range(num_appliances) for h in range(hours)}
    s = {t: [LpVariable(f"s_{t}_{h}", cat="Binary") for h in range(hours)] for t in [5, 6]}
    L = LpVariable("Peak_Load", lowBound=0)
    daily_requirements = np.array([2, 5, 2.88, 3, 5, 1.5, 0.8])  # From Table I
    model += L
    for h in range(hours):
        model += lpSum(x[a, h] for a in range(num_appliances)) <= L
    for h in range(1, hours+1):
        if h in [19, 20]:
            model += x[0, h-1] >= 1

        if h in [3, 4, 5, 21, 22]:
            model += x[1, h-1] >= 1
        model += x[2, h-1] == 0.12
        model += x[3, h-1] <= 1.5

        if 20 <= h or h <= 8:
            model += x[4, h-1] <= 3
            model += x[4, h-1] >= 0.1
        else:
            model += x[4, h-1] == 0 
    for t in [5, 6]:
        model += lpSum(s[t][h] for h in range(hours)) == 1 
    # Ensure power allocation follows P_t for time-shiftable appliances
    for t in [5, 6]:
        for h in range(hours):
            model += x[t, h] == lpSum(P_t[t][(h - start_h) % hours] * s[t][start_h] for start_h in range(hours))
    for a in range(num_appliances):
        model += lpSum(x[a, h] for h in range(hours)) == daily_requirements[a]
    model.solve()
    optimized_schedule = [sum(x[a, h].varValue for a in range(num_appliances)) for h in range(hours)]
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, 25), optimized_schedule, label="Optimized Load")
    plt.xlabel("Time (Hour)")
    plt.ylabel("Power Consumption (kWh)")
    plt.ylim(0, 1.5)
    plt.legend()
    plt.grid(True)
    plt.show()
    individual_schedules = {a: [x[a, h].varValue for h in range(hours)] for a in range(num_appliances)}
    apps = ["Oven", "Heater", "Fridge", "Water Boiler", "EV Charger", "Washing Machine", "Dishwasher"]
    colors = ["b", "g", "r", "c", "m", "y", "k"] 
    linestyles = [":", ":", ":", "--", "--", "-", "-"]  
    markers = ["o", "s", "D", "^", "v", "<", ">"] 
    plt.figure(figsize=(12, 6))
    for a, schedule in individual_schedules.items():
        plt.plot(range(1, 25), schedule, color=colors[a], linestyle=linestyles[a], marker=markers[a], label=f"{apps[a]}")
    plt.xlabel("Time (Hour)")
    plt.ylabel("Power Consumption (kWh)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
def smart_grid():
    hours = 24
    num_appliances = 7
    P_t = {
        5: np.array([[1, 0.5] + [0] * (hours - 2)] * hours),
        6: np.array([[0.8] + [0] * (hours - 1)] * hours)
    }

    daily_requirements = np.array([1, 3, 2.88, 3, 5, 1.5, 0.8])
    c = np.zeros(hours * num_appliances + 1)
    c[-1] = 1
    G_ub = np.zeros((hours, len(c)))  # 只需要保留峰值负载约束
    h_ub = np.zeros(hours)
    A_eq = np.zeros((num_appliances + hours * 2, len(c)))  # 保留每日能量需求和时间可调设备的约束
    b_eq = np.zeros(num_appliances + hours * 2)
    bounds = [(0, None) for _ in range(hours * num_appliances)] + [(0, None)]
    conti_vars = []

    # 定义决策变量和边界
    for a in range(num_appliances):
        for h in range(hours):
            if a == 0 and h in [18, 19]:  # Hob & Oven (19-20时段)
                bounds[a * hours + h] = (1, None)
            elif a == 1 and h in [2, 3, 4, 20, 21]:  # Heater (3-5时段和21-22时段)
                bounds[a * hours + h] = (1, None)
            elif a == 2:  # Fridge & Freezer (固定0.12kWh)
                bounds[a * hours + h] = (0.12, 0.12)
            elif a == 3:  # Water Boiler (0-1.5kWh)
                bounds[a * hours + h] = (0, 1.5)
            elif a == 4:  # EV Charger
                if 19 <= h or h <= 7:  # 20-8时段
                    bounds[a * hours + h] = (0.1, 3)
                else:
                    bounds[a * hours + h] = (0, 0)
            elif a in [5, 6]:  # 时间可调设备（洗衣机和洗碗机）
                bounds[a * hours + h] = (0, 1)
            else:
                bounds[a * hours + h] = (0, None)
                conti_vars.append(a * hours + h)

    # 峰值负载变量的边界
    bounds.append((0, None))
    conti_vars.append(len(c)-1)

    for t in [5, 6]:
        for h in range(hours):
            for start_h in range(hours):
                A_eq[num_appliances + t * hours + h, t * hours + h] += P_t[t][(h - start_h) % hours]

    for a in range(num_appliances):
        for h in range(hours):
            A_eq[a, a * hours + h] = 1
        b_eq[a] = daily_requirements[a]
    # 添加设备特定的约束
    for h in range(1, hours + 1):
        # Hob & Oven
        if h in [19, 20]:
            G_ub[h - 1, 0 * hours + h - 1] = -1
            h_ub[h - 1] = -1
        # Heater
        if h in [3, 4, 5, 21, 22]:
            G_ub[h - 1, 1 * hours + h - 1] = -1
            h_ub[h - 1] = -1
        # Fridge & Freezer
        A_eq[2, 2 * hours + h - 1] = 1
        b_eq[2] = 0.12 * hours
        # Water Boiler
        G_ub[h - 1, 3 * hours + h - 1] = 1.5
        # EV Charger
        if 20 <= h or h <= 8:
            G_ub[h - 1, 4 * hours + h - 1] = 1  # 上界约束
            G_ub[h - 1, 5 * hours + h - 1] = -1  # 下界约束
            h_ub[h - 1] = 3  # 下界值
        else:
            A_eq[3, 4 * hours + h - 1] = 1
            b_eq[3] = 0
    # 添加峰值负载约束
    for h in range(hours):
        G_ub[h, h::hours] = 1  # 每小时的总功耗
        G_ub[h, -1] = -1  # 减去峰值负载变量
        h_ub[h] = 0
    orig_prblm = LpPrblm(((0, 0), 0), None, 0, c, G_ub, h_ub, A_eq, b_eq, bounds)
    orig_prblm.conti_vars=conti_vars
    print(orig_prblm)
    solve_ilp_by_pulp(orig_prblm)
    optimized_schedule = [sum(orig_prblm.ix_pulp[i] for i in range(hours * num_appliances))] * hours
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 25), optimized_schedule, marker="o", linestyle="-", label="Optimized Load")
    plt.xlabel("Time (Hour)")
    plt.ylabel("Power Consumption (kWh)")
    plt.title("Optimized Hourly Power Consumption")
    plt.legend()
    plt.grid(True)
    plt.show()
