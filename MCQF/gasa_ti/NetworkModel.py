import copy
import datetime
import numpy as np
from PriorityGroup import GroupModel
from GaSaAlgorithm import genetic_algorithm, E2E_objective, reset_global_cache, simulated_annealing


LINK_BANDWIDTH = 1000e6 #100e6 # 100 Mbps

priority_groups = 3
priority_queues =  [['q7','q6','q5'],['q4','q3'],['q2','q1']]
CYCLE_TIME = [25,50,100, 'min_resource_per_group (%)', 40,50,50]

#########################################################################

def load_graph_stream_data(topo_fp, flows_filepath):
    with open(topo_filepath, 'r') as f:
        topo_raw = f.readlines()
    with open(flows_filepath, 'r') as g:
        flows_raw = g.readlines()
    return topo_raw, flows_raw

def process_flow_data(flow_data_raw):
    all_streams = list(map(lambda el: el.rstrip().split(','), flow_data_raw))
    return all_streams

def process_graph_data(graph_data_raw):
    nodes = []
    edges = []
    for el in graph_data_raw:
        el = el.rstrip().split(',')
        if el[0] == 'vertex':
            nodes.append(el)
        elif el[0] == 'edge':
            edges.append(el)
    return {'nodes': nodes, 'edges':edges}

def find_cycle_times(streams_data1, streams_data2, streams_data3, total_bandwidth, gcd_periods):

    max_packet_size1 = np.max(list(map(lambda el: int(el[-1]), streams_data1)))
    max_packet_size_buffered1 = 11/10*max_packet_size1
    max_packet_size2 = np.max(list(map(lambda el: int(el[-1]), streams_data2)))
    max_packet_size_buffered2 = 11/10*max_packet_size2
    max_packet_size3 = np.max(list(map(lambda el: int(el[-1]), streams_data3)))
    max_packet_size_buffered3 = 11/10*max_packet_size3

    min_cycle_time = max_packet_size_buffered1 / (total_bandwidth // 8)*1e6 # min_cycle_tim_in_ms
    min_cycle_time2 = max_packet_size_buffered2 / (total_bandwidth // 8)*1e6 # min_cycle_tim_in_ms
    min_cycle_time3 = max_packet_size_buffered3 / (total_bandwidth // 8)*1e6 # min_cycle_tim_in_ms

    mct1 = min_cycle_time
    mct2 = min_cycle_time2
    mct3 = min_cycle_time3

    init_time = int(np.ceil(min_cycle_time))
    temp_time = init_time
    possible_cycle_configuration = []
    while True:
        if not gcd_periods % temp_time:
            for i in range(2,11):
                for j in range(i+1,12):
                    if not gcd_periods % (i*j*temp_time):
                        possible_cycle_configuration.append([temp_time,i*temp_time,i*j*temp_time,
                                                             "min_resource_per_group (%): ",
                                                             round(np.ceil(mct1*100/temp_time)),
                                                             round(np.ceil(mct2*100/(i*temp_time))),
                                                             round(np.ceil(mct3*100/(i*j*temp_time)))])
                        possible_cycle_configuration.append([temp_time,j*temp_time,i*j*temp_time,
                                                             "min_resource_per_group (%): ",
                                                             round(np.ceil(mct1*100/temp_time)),
                                                             round(np.ceil(mct2*100/(j*temp_time))),
                                                             round(np.ceil(mct3*100/(i*j*temp_time)))])
        if 4*temp_time > gcd_periods:
            break
        temp_time+=1

    print(possible_cycle_configuration)
    resource_aware_cycle_configurations = list(filter(lambda el: sum(el[-3:]) < 100,possible_cycle_configuration))
    print("Resource aware configuration: ", resource_aware_cycle_configurations)

    cycle_times = CYCLE_TIME

    return cycle_times

def run_network_model(all_streams_data, topology_data, export_name=None, is_use_SA=False, cycle_times=None):

    periods_all_streams = [int(stream[-5]) for stream in all_streams_data ]
    hyper_period = np.lcm.reduce(periods_all_streams)
    gcd_periods = np.gcd.reduce(periods_all_streams)

    #TODO sort over deadlines
    sorted_all_streams_data = list(sorted(all_streams_data, key=lambda el: int(el[-3]),reverse=False))
    streams_number = len(sorted_all_streams_data)

    sn = streams_number

    # Divide into segments
    streams_data1 = sorted_all_streams_data[:int(0.5 * sn)]  # 50%
    streams_data2 = sorted_all_streams_data[int(0.5 * sn):int(0.8 * sn)]  # 30%
    streams_data3 = sorted_all_streams_data[int(0.8 * sn):]  # Remaining 20%

    cycle_times = CYCLE_TIME

    if cycle_times is None:
        cycle_times = find_cycle_times(streams_data1, streams_data2, streams_data3, LINK_BANDWIDTH, gcd_periods)
    else:
        cycle_times = cycle_times

    #TODO set to False
    if False:
        streams_data1 = sorted_all_streams_data
        streams_data2 = copy.deepcopy(sorted_all_streams_data[-1:])
        streams_data3 = copy.deepcopy(sorted_all_streams_data[-1:])
        cycle_times[4]= 100
        bw_pg1 = 1
        bw_pg2 = 0.0001
        bw_pg3 = 0.0001
    else:
        free_bw_resources = 100 - sum(cycle_times[-3:])
        bw_pg1 = cycle_times[4]/100
        bw_pg2 = cycle_times[5]/100
        bw_pg3 = cycle_times[6]/100

        bw_sum = 0.9*LINK_BANDWIDTH
        print("bw_sum", bw_sum)
        bw_pg1 += bw_pg1/bw_sum*(free_bw_resources/100)
        bw_pg2 += bw_pg2/bw_sum*(free_bw_resources/100)
        bw_pg3 += bw_pg3/bw_sum*(free_bw_resources/100)

        delta_amount_flows1 = int((bw_pg1-1/3) * len(sorted_all_streams_data))
        delta_amount_flows2 = int((bw_pg2-1/3) * len(sorted_all_streams_data))
        delta_amount_flows3 = int((bw_pg3-1/3) * len(sorted_all_streams_data))

        for i in range(delta_amount_flows1):
            if i+1 > abs(delta_amount_flows3) and streams_data2:
                streams_data1.append(streams_data2.pop(0))
            else:
                if streams_data3:
                    streams_data2.append(streams_data3.pop(0))
                    streams_data1.append(streams_data2.pop(0))

    PG1 = GroupModel(topology_data, streams_data1, 1, priority_queues, cycle_times[0], None, bandwidth=(bw_pg1*LINK_BANDWIDTH/8), export_name=export_name)
    PG2 = GroupModel(topology_data, streams_data2, 2, priority_queues, cycle_times[1], None, bandwidth=(bw_pg2*LINK_BANDWIDTH/8), export_name=export_name)
    PG3 = GroupModel(topology_data, streams_data3, 3, priority_queues, cycle_times[2], None, bandwidth=(bw_pg3*LINK_BANDWIDTH/8), export_name=export_name)

    if not is_use_SA:
        best_individual1 = genetic_algorithm(PG1) #genetic_algorithm(PG1)
        best_individual2 = genetic_algorithm(PG2) #genetic_algorithm(PG2)
        best_individual3 = genetic_algorithm(PG3) #genetic_algorithm(PG3)
    else:
        best_individual1 = simulated_annealing(PG1) #genetic_algorithm(PG1)
        best_individual2 = simulated_annealing(PG2) #genetic_algorithm(PG2)
        best_individual3 = simulated_annealing(PG3) #genetic_algorithm(PG3)

    NM = [PG1,PG2,PG3]
    group_solution = [ best_individual1,best_individual2,best_individual3 ]

    return NM, group_solution

if __name__ == '__main__':

    for i in range(11,12):

        reset_global_cache()

        topo_filepath = "..\TC\TCs2\TC"+str(i)+"_topo.txt"
        flows_filepath = "..\TC\TCs2\TC"+str(i)+"_flows.txt"

        now = datetime.datetime.now()
        date_and_hour = now.strftime("_%Y-%m-%d_%H-%M-%S")
        result_name = topo_filepath.split('\\''')[-1].split('.')[0]+date_and_hour

        graph_data_raw, flow_data_raw = load_graph_stream_data(topo_filepath, flows_filepath)
        td = process_graph_data(graph_data_raw)
        asd = process_flow_data(flow_data_raw)[:]

        NM, group_solution = run_network_model(asd, td, result_name)

        for PrioGroup, sol, pgidx in zip(NM,group_solution,[1,2,3]):

            obj = E2E_objective(sol, PrioGroup.space, True, True, bandwidth=PrioGroup.bandwidth,find_ratio_scheduled=True)
            PrioGroup.show_full_routes_and_schedules(sol)
            PrioGroup.export_full_routes_and_schedules(sol, "k: "+str(pgidx)+" "+str(obj[1])+'\n')
            PrioGroup.export_full_routes_and_schedules_csv(sol, "k: "+str(pgidx)+" "+str(obj[1])+'\n')
            print("done")
