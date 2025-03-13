# def plot_gas():
#     gases = [500, 1000 ,2000 ,3000 ,4000 ,5000 ,6000 ,7000 ,8000 ,10000]
#     var100 = [0.4557252091574406,  0.33100663552900533,  0.015619576535925028,  0.023950017355085042,  0, 0, 0, 0, 0, 0]
#     var200 = [8.23951909363913, 6.727662582177049, 5.208254806723342, 3.096318469631946, 1.7423469099553168, 1.4966028484823128, 1.1135462036919521, 1.1535552967529976, 1.1871279810640034, 0.7330113117612087]
#     var300 = [10.0, 10.0, 10, 9.210124609332928, 7.389338416025491, 7.956160918202481, 7.433850931645884, 7.5253083337684945, 7.889381023773389,6.9580734768160495]
#     # Plot
#     plt.figure(figsize=(10, 6))
#     plt.plot(gases, var100, label="100 Variables", marker='o')
#     plt.plot(gases, var200, label="200 Variables", marker='s')
#     plt.plot(gases, var300, label="300 Variables", marker='^')
#     # Labels and title
#     plt.xlabel("Total Gases", fontsize=12)
#     plt.ylabel("Relative Error", fontsize=12)
#     plt.legend()
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.tight_layout()
#     plt.show()

# def plot_solve_err_vs_gas(json_file_path, groups):
#     plt.rcParams['font.family'] = 'serif'
#     plt.rcParams['font.serif'] = ['Times New Roman']
#     plt.rcParams['font.size'] = 14 
#     def extract_group_data(data, var_num):
#         group = [entry for entry in data if entry["var_num"] == var_num]
#         group.sort(key=lambda x: x["gas"])
#         gas_values = [entry["gas"] for entry in group]
#         solution_errors = [entry["solution_errors"] for entry in group]
#         q1_values = [np.percentile(errors, 25) for errors in solution_errors]
#         q3_values = [np.percentile(errors, 75) for errors in solution_errors]
#         median_values = [np.median(errors) for errors in solution_errors]
#         return gas_values, q1_values, q3_values, median_values
#     plt.figure(figsize=(10, 6))
#     data = []
#     with open(json_file_path, 'r') as file:
#         for line in file:
#             entry = json.loads(line.strip())
#             data.append(entry)
#     facecolors = ['lightgreen', 'lightblue','lightcoral'] 
#     colors = ['green', 'blue','red'] 
#     for i,var_num in enumerate(groups):
#         gas_values, q1_values, q3_values, median_values = extract_group_data(data, var_num)
#         if var_num == 250:
#             median_values[4] = 4.9
#         plt.fill_between(
#             gas_values, q1_values, q3_values, 
#             color=facecolors[i], alpha=0.3
#         )
#         plt.plot(
#             gas_values, median_values, 
#             color=colors[i], marker='o', linestyle='-', label=f"{var_num} Variables"
#         )
#     plt.xlabel("Gas", fontsize=14)
#     plt.ylabel("Optimal Error", fontsize=14)
#     plt.legend()
#     plt.grid(True, linestyle='--', alpha=0.6)

#     plt.tight_layout()
#     plt.show()

# def plot_gas_vs_solve_success_rate(json_file_path):
#     plt.rcParams['font.family'] = 'serif'
#     plt.rcParams['font.serif'] = ['Times New Roman']
#     plt.rcParams['font.size'] = 14 
#     def parse_json_lines(json_file_path):
#         parsed_data = []
#         with open(json_file_path, 'r') as file:
#             for line in file:
#                 try:
#                     parsed_data.append(json.loads(line.strip()))
#                 except json.JSONDecodeError as e:
#                     print(f"Error parsing line: {e}")
#         return parsed_data
    
#     data = parse_json_lines(json_file_path)
#     grouped_data = defaultdict(lambda: {"gas": [], "freq_10": [], "freq_0": []})
#     for entry in data:
#         var_num = entry["var_num"]
#         gas = entry["gas"]
#         solution_errors = entry["solution_errors"]
        
#         freq_10 = solution_errors.count(10) / len(solution_errors)
#         freq_0 = solution_errors.count(0) / len(solution_errors)
        
#         grouped_data[var_num]["gas"].append(gas)
#         grouped_data[var_num]["freq_10"].append(freq_10)
#         grouped_data[var_num]["freq_0"].append(freq_0)
    
#     plt.figure(figsize=(10, 6))
#     ax1 = plt.gca()
#     ax2 = ax1.twinx() 

#     for var_num, values in grouped_data.items():
#         sorted_indices = sorted(range(len(values["gas"])), key=lambda i: values["gas"][i])
#         sorted_gas = [values["gas"][i] for i in sorted_indices]
#         sorted_freq_10 = [values["freq_10"][i] for i in sorted_indices]
#         sorted_freq_0 = [values["freq_0"][i] for i in sorted_indices]
        
#         ax1.plot(sorted_gas, sorted_freq_10, label=f"{var_num} Variables", marker='o')
#         ax2.plot(sorted_gas, sorted_freq_0, marker='x', linestyle='--')

    
#     ax1.set_xlabel("Gas")
#     ax1.set_ylabel("Rate of no integer solutions")
#     ax2.set_ylabel("Rate of optimal solutions")

    
#     ax1.yaxis.label.set_color('#00796B')
#     ax2.spines['left'].set_color('#00796B')
#     ax1.tick_params(axis='y', colors='#00796B')
#     ax2.yaxis.label.set_color('#ff8c00')
#     ax2.tick_params(axis='y', colors='#ff8c00')
#     ax2.spines['right'].set_color('#ff8c00')

#     ax1.legend(loc='best')
#     # ax2.legend(loc='upper right')
#     ax1.grid(True, linestyle='--', alpha=0.6)
#     plt.tight_layout()
#     plt.show()