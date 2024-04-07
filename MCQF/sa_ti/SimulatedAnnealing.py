import time
import copy
import numba
import random
#import Stream
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from PriorityGroup import GroupModel
from timeit import default_timer as timer

# Simulated Annealing Parameters
TEMPERATURE_0 = 50
COOLING_RATE = 99
TERMINATION_CONDITION='2e4iter'

MEAS_CONVERGENCE = False
MEAS_TIME = False
CACHE_STREAM_ROUTE = {}
TOTAL_OBJECT_COUNT = {"cost_function": 0}

GLOBAL_COUNTER = {'calls':0}
#########################################################################

def E2E_objective(all_routed_streams, all_streams, is_normalize=True, is_stats=False, bandwidth = None,
                  find_ratio_scheduled=False):

    GLOBAL_COUNTER['calls'] +=1

    objective_val = 0

    if is_stats: delay_violations = 0
    for idx, (route, stream) in enumerate(zip(all_routed_streams, all_streams)):

        scale_delay = 1
        if is_normalize:
            scale_delay = (np.floor(stream[0].delay_cycles))
        objective_val += (1/len(all_routed_streams)) * (E2E_delay(route, stream)/scale_delay)

        if E2E_delay(route, stream) > int(np.floor(stream[0].delay_cycles)):
            objective_val += 2
            if is_stats: delay_violations +=1

    if bandwidth is None:
        bandwidth = 600

    timey = timer()
    utilization, pc = BU_bandwidth_utilization3(all_routed_streams, all_streams, bandwidth)
    objective_val += 3*pc

    timey = timer()-timey
    timex = timer()
    timex = timer()-timex

    if is_stats: bu_violation = 0
    for cycle_utilization in utilization:
        for link_utilization_in_cycle_c in cycle_utilization.values():
            if link_utilization_in_cycle_c > bandwidth:
                objective_val += 3
                if is_stats: bu_violation += 1

    if np.random.rand() < 0.0025: print("calculation time: ", round(timey, 4))

    if is_stats:
        print("current_solution_violates_delay_stream_delay:", delay_violations, "times")
        print("current_solution_violates_bandwith_on_cycle_link:", bu_violation, "times")
        print(objective_val)
        print(utilization)

    if find_ratio_scheduled:
        scheduled_percentage = find_scheduled_percentage(all_routed_streams, all_streams, False, bandwidth=bandwidth)
        print("Scheduled_percentage", scheduled_percentage)
        return objective_val, scheduled_percentage

    return objective_val

def reset_global_cache():
    global CACHE_STREAM_ROUTE
    CACHE_STREAM_ROUTE = {}
    print(CACHE_STREAM_ROUTE)

def BU_bandwidth_utilization33(all_routes, all_streams, penalty_bw_threshold=None, ignore_streams=[]):
    penalty_counter = 0
    penalty_streams ={}

    periods_all_streams = [int(2*round(stream[0].period_cycles/2)) for stream in all_streams ]
    no_cycles = np.lcm.reduce(periods_all_streams)
    all_cycles = [{} for i in range(no_cycles) ]
    all_links = [s[1][r[0]][1][1:-1] for s, r in zip(all_streams, all_routes)]
    for links, stream, route in zip(all_links, all_streams, all_routes):
        if stream[0].id in ignore_streams:
            continue
        try:
            if CACHE_STREAM_ROUTE.get(ck := ';'.join([str(stream[0].id),str(stream[0].k),str(route)])) is not None:
                package = int(CACHE_STREAM_ROUTE[ck][0])
                for sq, pk in CACHE_STREAM_ROUTE[ck][1:]:
                    if all_cycles[sq].get(pk) is None:
                        all_cycles[sq][pk] = package
                    else:
                        all_cycles[sq][pk] = all_cycles[sq][pk]+package
                        if all_cycles[sq][pk] > penalty_bw_threshold:
                            penalty_counter += 1
                            penalty_streams[stream[0].id] = 1

                continue
        except Exception as e:
            print(e)

        for idx,link in enumerate(links):
            link_propagation_delay = 0
            link_schedule_time = 0
            link_schedule_time += route[-1] # add planned injection time
            link_schedule_time += sum(route[1:1+idx])
            link_schedule_time += (1+idx)*link_propagation_delay
            assigned_queue = route[1+idx]
            for cyc_idx, cycle in enumerate(all_cycles[link_schedule_time::int(stream[0].period_cycles)]):
                current_cycle = link_schedule_time + cyc_idx*int(stream[0].period_cycles)
                sending_queue = current_cycle+assigned_queue
                sending_port = stream[1][0][1][-1] if idx+1 == len(links) else links[idx+1]
                if sending_queue > no_cycles-1:
                    continue
                port_key = str(link)+'->'+str(sending_port)
                #if len(CACHE_STREAM_ROUTE) < int(1e7):
                if len(CACHE_STREAM_ROUTE) < int(1):
                    if CACHE_STREAM_ROUTE.get(';'.join([str(stream[0].id),str(stream[0].k),str(route)])) is None:
                        CACHE_STREAM_ROUTE[';'.join([str(stream[0].id),str(stream[0].k),str(route)])] = [
                                                     int(stream[0].size_byte),[sending_queue, port_key]]
                    else:
                        CACHE_STREAM_ROUTE[';'.join([str(stream[0].id),str(stream[0].k),str(route)])].append([
                                                                               sending_queue, port_key])

                if all_cycles[sending_queue].get(port_key) is None:
                    all_cycles[sending_queue][str(port_key)]=int(stream[0].size_byte)
                else:
                    all_cycles[sending_queue][str(port_key)]+=int(stream[0].size_byte)
                    if all_cycles[sending_queue][port_key] > penalty_bw_threshold:
                        penalty_counter += 1
                        penalty_streams[stream[0].id] = 1

    if penalty_bw_threshold is not None:
        return all_cycles, len(penalty_streams.keys())
    return all_cycles

def BU_bandwidth_utilization3(all_routes, all_streams, penalty_bw_threshold=None):
    penalty_counter = 0

    periods_all_streams = [int(2*round(stream[0].period_cycles/2)) for stream in all_streams ]
    no_cycles = np.lcm.reduce(periods_all_streams)
    all_cycles = [{} for i in range(no_cycles) ]
    all_links = [s[1][r[0]][1][1:-1] for s, r in zip(all_streams, all_routes)]
    for links, stream, route in zip(all_links, all_streams, all_routes):
        try:
            if CACHE_STREAM_ROUTE.get(ck := ';'.join([str(stream[0].id),str(stream[0].k),str(route)])) is not None:
                package = int(CACHE_STREAM_ROUTE[ck][0])
                for sq, pk in CACHE_STREAM_ROUTE[ck][1:]:
                    if all_cycles[sq].get(pk) is None:
                        all_cycles[sq][pk] = package
                    else:
                        all_cycles[sq][pk] = all_cycles[sq][pk]+package
                        if all_cycles[sq][pk] > penalty_bw_threshold: penalty_counter += 1

                continue
        except Exception as e:
            print(e)

        for idx,link in enumerate(links):
            link_propagation_delay = 0
            link_schedule_time = 0
            link_schedule_time += route[-1] # add planned injection time
            link_schedule_time += sum(route[1:1+idx])
            link_schedule_time += (1+idx)*link_propagation_delay
            assigned_queue = route[1+idx]
            for cyc_idx, cycle in enumerate(all_cycles[link_schedule_time::int(stream[0].period_cycles)]):
                current_cycle = link_schedule_time + cyc_idx*int(stream[0].period_cycles)
                sending_queue = current_cycle+assigned_queue
                sending_port = stream[1][0][1][-1] if idx+1 == len(links) else links[idx+1]
                if sending_queue > no_cycles-1:
                    continue
                port_key = str(link)+'->'+str(sending_port)
                #if len(CACHE_STREAM_ROUTE) < int(1e7):
                if len(CACHE_STREAM_ROUTE) < int(1):
                    if CACHE_STREAM_ROUTE.get(';'.join([str(stream[0].id),str(stream[0].k),str(route)])) is None:
                        CACHE_STREAM_ROUTE[';'.join([str(stream[0].id),str(stream[0].k),str(route)])] = [
                                                     int(stream[0].size_byte),[sending_queue, port_key]]
                    else:
                        CACHE_STREAM_ROUTE[';'.join([str(stream[0].id),str(stream[0].k),str(route)])].append([
                                                                               sending_queue, port_key])

                if all_cycles[sending_queue].get(port_key) is None:
                    all_cycles[sending_queue][str(port_key)]=int(stream[0].size_byte)
                else:
                    all_cycles[sending_queue][str(port_key)]+=int(stream[0].size_byte)
                    if all_cycles[sending_queue][port_key] > penalty_bw_threshold: penalty_counter += 1
    if penalty_bw_threshold is not None:
        return all_cycles, penalty_counter
    return all_cycles

def find_BU_utilization_offenders(all_routes, all_streams, bandwidth, misscheduled_ids):


    periods_all_streams = [int(2*round(stream[0].period_cycles/2)) for stream in all_streams ]
    no_cycles = np.lcm.reduce(periods_all_streams)
    all_cycles = [{} for i in range(no_cycles) ]
    utilization_ids = [{} for i in range(no_cycles)]
    all_links = [s[1][r[0]][1][1:-1] for s, r in zip(all_streams, all_routes)]
    for links, stream, route in zip(all_links, all_streams, all_routes):

        if stream[0].id in misscheduled_ids:
            continue

        for idx,link in enumerate(links):
            link_propagation_delay = 0
            link_schedule_time = 0
            link_schedule_time += route[-1] # add planned injection time
            link_schedule_time += sum(route[1:1+idx])
            link_schedule_time += (1+idx)*link_propagation_delay
            assigned_queue = route[1+idx]
            for cyc_idx, cycle in enumerate(all_cycles[link_schedule_time::int(stream[0].period_cycles)]):
                current_cycle = link_schedule_time + cyc_idx*int(stream[0].period_cycles)
                sending_queue = current_cycle+assigned_queue
                sending_port = stream[1][0][1][-1] if idx+1 == len(links) else links[idx+1]
                if sending_queue > no_cycles-1:
                    continue
                port_key = str(link)+'->'+str(sending_port)
                if all_cycles[sending_queue].get(port_key) is None:
                    all_cycles[sending_queue][str(port_key)]=int(stream[0].size_byte)
                    utilization_ids[sending_queue][str(port_key)] = [[stream[0].id,stream[0].size_byte]]
                else:
                    all_cycles[sending_queue][str(port_key)]+=int(stream[0].size_byte)
                    utilization_ids[sending_queue][str(port_key)].append([stream[0].id,stream[0].size_byte])

    return all_cycles, utilization_ids

def BU_bandwidth_utilization4(all_routes, all_streams):

    periods_all_streams = [int(2*round(stream[0].period_cycles/2)) for stream in all_streams ]
    no_cycles = np.lcm.reduce(periods_all_streams)
    all_cycles = [{} for i in range(no_cycles) ]
    all_links = [s[1][r[0]][1][1:-1] for s, r in zip(all_streams, all_routes)]
    for links, stream, route in zip(all_links, all_streams, all_routes):
        for idx,link in enumerate(links):
            link_propagation_delay = 0
            link_schedule_time = 0
            link_schedule_time += route[-1] # add planned injection time
            link_schedule_time += sum(route[1:1+idx])
            link_schedule_time += (1+idx)*link_propagation_delay
            assigned_queue = route[1+idx]
            for cyc_idx, cycle in enumerate(all_cycles[link_schedule_time::int(stream[0].period_cycles)]):
                current_cycle = link_schedule_time + cyc_idx*int(stream[0].period_cycles)
                sending_queue = current_cycle+assigned_queue
                sending_port = stream[1][0][1][-1] if idx+1 == len(links) else links[idx+1]
                if sending_queue > no_cycles-1:
                    continue
                port_key = str(link)+'->'+str(sending_port)
                if all_cycles[sending_queue].get(port_key) is None:
                    all_cycles[sending_queue][str(port_key)]=int(stream[0].size_byte)
                else:
                    all_cycles[sending_queue][str(port_key)]+=int(stream[0].size_byte)
    return all_cycles

def BU_bandwidth_utilization(all_routes, all_streams):

    periods_all_streams = [int(2*round(stream[0].period_cycles/2)) for stream in all_streams ]
    no_cycles = np.lcm.reduce(periods_all_streams)
    all_links = [s[1][r[0]][1][1:-1] for s, r in zip(all_streams, all_routes)]
    all_periods = list(map(lambda el: int(el[0].period_cycles), all_streams))
    all_sending_ports = list(map(lambda el: int(el[1][0][1][-1]), all_streams))
    all_packet_sizes = list(map(lambda el: int(el[0].size_byte), all_streams))

    all_links_array = np.zeros((len(all_links),max(list(map(lambda el: len(el), all_links)))))
    for idx, link in enumerate(all_links):
        all_links_array[idx][:len(link)] = link
    all_links_lengths = np.array(list(map(lambda el: len(el),all_links)))

    all_routes_array = np.zeros((len(all_routes),max(list(map(lambda el: len(el), all_routes)))))
    for idx, route in enumerate(all_routes):
        all_routes_array[idx][:len(route)] = route
    all_routes_lengths = np.array(list(map(lambda el: len(el),all_routes)))


    timex = timer()
    all_cycles, all_keys = BU_utilization_helper(no_cycles,all_links_array, all_links_lengths, np.array(all_periods),
                     np.array(all_sending_ports), np.array(all_packet_sizes), all_routes_array, all_routes_lengths)

    all_cycles_dict = [{} for el in range(len(all_cycles))]

    timex = timer()
    for idx, el in enumerate(all_keys):
        while key := all_keys[idx].pop(0):
            all_cycles_dict[idx][key] = all_cycles[idx].pop(0)

    return all_cycles_dict

def BU_utilization_helper2(no_cycles, all_links,all_periods,all_sending_ports,all_packet_sizes,all_routes):

    all_cycles = [{} for i in range(no_cycles) ]
    all_keys = [[] for i in range(no_cycles)]

    for links, period, end_port, packet_size, route in zip(all_links, all_periods,all_sending_ports,all_packet_sizes, all_routes):
        for idx,link in enumerate(links):
            link_propagation_delay = 0
            link_schedule_time = 0
            link_schedule_time += route[-1] # add planned injection time
            link_schedule_time += sum(route[1:1+idx])
            link_schedule_time += (1+idx)*link_propagation_delay
            assigned_queue = route[1+idx]
            for cyc_idx, cycle in enumerate(all_cycles[link_schedule_time::int(period)]):
                current_cycle = link_schedule_time + cyc_idx*int(period)
                sending_queue = current_cycle+assigned_queue
                sending_port = end_port if idx+1 == len(links) else links[idx+1]
                if sending_queue > no_cycles-1:
                    continue
                port_key = str(link)+'->'+str(sending_port)
                #if all_cycles[sending_queue].get(port_key) is None:
                if port_key not in all_keys[sending_queue]:
                    all_cycles[sending_queue][str(port_key)]=int(packet_size)
                    all_keys[sending_queue].append(port_key)
                else:
                    all_cycles[sending_queue][str(port_key)]+=int(packet_size)

    return all_cycles

def BU_utilization_helper(no_cycles, all_links_array, all_links_lengths, all_periods,
                          all_sending_ports, all_packet_sizes, all_routes_array, all_routes_lengths):

    all_cycles = [[0 for j in range(len(all_sending_ports))] for i in range(no_cycles)]
    all_keys = [['' for j in range(len(all_sending_ports))] for i in range(no_cycles)]

    for (links_trail, links_length, period, end_port,
         packet_size, route_trail, route_length) in zip(all_links_array, all_links_lengths,
                                                        all_periods, all_sending_ports, all_packet_sizes,
                                                        all_routes_array, all_routes_lengths):

        links = links_trail[:links_length]
        route = route_trail[:route_length]
        for idx, link in enumerate(links):
            link_propagation_delay = 0
            link_schedule_time = 0
            link_schedule_time += route[-1] # add planned injection time
            link_schedule_time += int(sum(route[1:1+idx]))
            link_schedule_time += (1+idx) * link_propagation_delay
            assigned_queue = route[1+idx]
            new_port_key_idx = 0
            for cyc_idx, cycle in enumerate(all_cycles[int(link_schedule_time)::int(period)]):
                current_cycle = int(link_schedule_time + cyc_idx * int(period))
                sending_queue = int(current_cycle + assigned_queue)
                sending_port = int(end_port) if idx+1 == len(links) else int(links[idx+1])
                if sending_queue > no_cycles-1:
                    continue
                port_key = str(int(link)) + '->' + str(int(sending_port))
                if port_key not in all_keys[int(sending_queue)]:
                    all_cycles[sending_queue][new_port_key_idx] = int(packet_size)
                    all_keys[sending_queue][new_port_key_idx] = port_key
                    new_port_key_idx += 1
                    all_cycles[sending_queue][idx] = int(packet_size)
                else:
                    idx2 = all_keys[sending_queue].index(port_key)
                    all_cycles[sending_queue][idx2] += int(packet_size)

    return all_cycles, all_keys

def BU_bandwith_utilization2(all_routes, all_streams):

    periods_all_streams = [int(2*round(stream[0].period_cycles/2)) for stream in all_streams ]
    no_cycles = np.lcm.reduce(periods_all_streams)
    all_cycles = [{} for i in range(no_cycles) ]
    all_links = [s[1][r[0]][1][1:-1] for s, r in zip(all_streams, all_routes)]
    for links, stream, route in zip(all_links, all_streams, all_routes):
        for idx,link in enumerate(links):
            link_propagation_delay = 0
            link_schedule_time = 0
            link_schedule_time += route[-1] # add planned injection time
            link_schedule_time += sum(route[1:1+idx])
            link_schedule_time += (1+idx)*link_propagation_delay
            assigned_queue = route[1+idx]
            for cyc_idx, cycle in enumerate(all_cycles[link_schedule_time::int(stream[0].period_cycles)]):
                current_cycle = link_schedule_time + cyc_idx*int(stream[0].period_cycles)
                sending_queue = current_cycle+assigned_queue
                if sending_queue > no_cycles-1:
                    continue
                if all_cycles[sending_queue].get(str(link)) is None:
                    all_cycles[sending_queue][str(link)]=int(stream[0].size_byte)
                else:
                    all_cycles[sending_queue][str(link)]+=int(stream[0].size_byte)

    return all_cycles

def E2E_delay(route, stream):
    e2e = 0
    stream_len = stream[1][route[0]][0]
    e2e_propagation_cycles = sum(route[1:1+stream_len])
    e2e_propagation_cycles += route[-1]
    e2e_propagation_cycles +=1
    return e2e_propagation_cycles

def load_graph_stream_data(topo_fp, flows_fp):
    with open(topo_fp, 'r') as f:
        topo_raw = f.readlines()
    with open(flows_fp, 'r') as g:
        flows_raw = g.readlines()
    return topo_raw, flows_raw

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

def process_flow_data(flow_data_raw):
    all_streams = list(map(lambda el: el.rstrip().split(','), flow_data_raw))
    return all_streams

def generate_population(PG):
    population = []
    init_member = copy.deepcopy(PG.solution)
    population.append(init_member)
    for _ in range(POPULATION_SIZE):
        PG.GenerateRandomIndividual()

        individual = copy.deepcopy(PG.solution)
        population.append(individual)
    return population

# Function to evaluate an individual
def evaluate(PG, individual):
    bandwidth = PG.bandwidth

    return E2E_objective(individual, PG.space, bandwidth=bandwidth)

def find_scheduled_percentage(all_routed_streams, all_streams, is_stats=False, bandwidth = None):
    num_misscheduled_streams = 0
    misscheduled_ids = []

    if is_stats: delay_violations = 0
    for idx, (route, stream)  in enumerate(zip(all_routed_streams, all_streams)):

        if E2E_delay(route, stream) > int(np.floor(stream[0].delay_cycles)):
            num_misscheduled_streams += 1
            misscheduled_ids.append(stream[0].id)

    if bandwidth is None:
        bandwidth = 600

    utilization_counter = find_BU_utilization_offenders(all_routed_streams, all_streams, bandwidth, misscheduled_ids)
    bu_utilization, id_utilization = utilization_counter
    bu_violators = calculate_misscheduled_bu_number(bu_utilization, id_utilization, bandwidth)
    print(bu_violators)
    total_unschedulable_streams = num_misscheduled_streams+len(bu_violators)
    total_streams = len(all_streams)
    percent_schedulable = 1 -total_unschedulable_streams/total_streams

    return (100*round(float(percent_schedulable),2), '%', "unscheduled", total_unschedulable_streams,
            "total", total_streams, "e2e too large", misscheduled_ids, "bandwidth violation", bu_violators )

def calculate_misscheduled_bu_number(bu, id, bw):
    print(bu, id)
    misscheduled_id = []
    for cycle_stats, cycle_id in zip(bu,id):
        for key in cycle_stats.keys():
            new_cycle_id = [el for el in cycle_id[key] if el[0] not in misscheduled_id ]
            new_bandwidth = sum([ int(el[1]) for el in new_cycle_id])
            if new_bandwidth > bw:
                threshold_size = new_bandwidth
                for packet in sorted(new_cycle_id, key = lambda el: int(el[1]),reverse=True):
                    flow_id = packet[0]
                    packet_size = int(packet[1])
                    threshold_size -= packet_size
                    misscheduled_id.append(flow_id)
                    if threshold_size < bw:
                        break
    print(misscheduled_id)
    return misscheduled_id

def simulated_annealing(network_model, T_0=1e4, cooling_rate=0.99, is_log_stats=False, bandwidth=None):

    nm = network_model
    streams = network_model.space
    bw = nm.bandwidth

    t = T_0
    t = 40
    cr = cooling_rate
    sol = copy.deepcopy(nm.solution[:])
    #sol_temp = sol_0
    sol_final = copy.deepcopy(sol[:])
    cost_final = E2E_objective(sol_final, streams, bandwidth=bw)
    print("Cost init:", cost_final)
    e2e_current = cost_final
    minute = 0
    no_iterations = 2*int(1e3)//1
    improving_moves = 0
    jump_stats_new = 0
    jump_stats_old = 0

    if is_log_stats: algo_temp_jumps = 0
    old_solution = copy.deepcopy(nm.solution[:])
    e2e_current = E2E_objective(old_solution, streams, bandwidth=bw)
    termination_time = timer()
    for iter_idx in range(no_iterations):
        if (duration := timer()-termination_time) > 2.4*60*60:
            break
        #########################################################################

        if MEAS_CONVERGENCE:
            pgidx = nm.k_priority
            if (timer() - termination_time)/27 > minute/nm.k_priority :
                minute+=1
                nm.solution = copy.deepcopy(sol_final)
                with open('temp_obj.txt','a') as f:
                    f.write(str(iter_idx) + " iteration, objective: "
                         + (str(round(cost_final,5))) + '\n')
                obj = E2E_objective(nm.solution, nm.space, True, True, bandwidth=nm.bandwidth, find_ratio_scheduled=True)
                nm.export_full_routes_and_schedules(nm.solution, "k: "+str(pgidx)+" "+str(obj[1])+'\n', convergence= minute)
                nm.export_full_routes_and_schedules_csv(nm.solution, obj[-1][-1]+obj[-1][-3] , convergence = minute)
                if minute >= 50:
                    break
        ################################################################################

        if True in [np.isclose((x:=iter_idx/no_iterations),0.05*i) for i in range(21) ]:
            print(int(100*x),"% finished, ", "running for:", round(duration/60), "(mins), group:", nm.k_priority)

        old_solution = copy.deepcopy(nm.solution[:])
        is_new = nm.GenerateNeighbour()
        new_solution = copy.deepcopy(nm.solution[:])
        e2e_temp = E2E_objective(new_solution, streams, bandwidth=bw)
        if np.random.rand() < 0.0025: print("eval cost | local opt cost:", round(e2e_temp,3),'|', round(e2e_current,3))
        if e2e_temp < e2e_current:
            improving_moves +=1
            if np.random.rand() < .025: print("iteration: ", iter_idx, "improving moves count: ", improving_moves)
            e2e_current = e2e_temp
            if e2e_temp < cost_final:
                sol_final = copy.deepcopy(new_solution[:])
                cost_final = e2e_temp
        else:
            if np.exp(-(e2e_temp-e2e_current)/t) < np.random.rand():
                jump_stats_new+=1
                if is_log_stats: algo_temp_jumps +=1
                nm.solution[:] = copy.deepcopy(old_solution[:])
            else :
                jump_stats_old+=1
            t *= cr
        if np.random.rand() < 0.0025:
            print("subopt solution taken | not taken:", jump_stats_old,"|", jump_stats_new)

        ##############################################################################

        if MEAS_TIME:
            if GLOBAL_COUNTER['calls'] > 1250:
                with open('temp_obj.txt', 'a') as f:
                    f.write(str(iter_idx) + " iteration, time(s): "
                         + (str(round(timer()-termination_time,3))) + '\n')
                    GLOBAL_COUNTER['calls'] = 0
                    break

        ##############################################################################

    #print(sol_final is sol)
    E2E_objective(sol_final[:], streams, True, True)
    if is_log_stats: print("Algo stats jumps:", algo_temp_jumps/no_iterations)

    return sol_final[:], cost_final

if __name__ == '__main__':

    # Run single group
    graph_data_raw, flow_data_raw = load_graph_stream_data(topo_filepath, flows_filepath)
    topology_data = process_graph_data(graph_data_raw)
    all_streams_data = process_flow_data(flow_data_raw)


    GM = GroupModel(topology_data, all_streams_data, priority_groups, priority_queues, cycle_times, None)

    # Run the genetic algorithm
    for _ in range(5): print()
    print("Best Individual:", best_individual)
    GM.show_solution(best_individual)

    GM.show_full_routes_and_schedules(best_individual)
    E2E_objective(best_individual, GM.space, True, True)
    print("done")

    pass
