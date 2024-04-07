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

# Choose Algorithm
ASAGA = False # GA
COMPLEX_GENES = True

# ASAGA Algorithm Parameters
POPULATION_SIZE = 16
GENERATIONS = 64
GENERATIONS = 4
NO_ITERATIONS = 40
MUTATION_RATE = 0.05
NO_MUTATION_GENES = 2
SA_INIT_TEMP = 1

# Genetic Algorithm Parameters
if not ASAGA:
    POPULATION_SIZE = 50
    GENERATIONS = 100
    MUTATION_RATE = 0.1
    NO_MUTATION_GENES = 2

MEAS_CONVERGENCE = False
MEAS_TIME = False
MEAS_TIME_SA = False

if MEAS_CONVERGENCE:
    if not ASAGA:
        GENERATIONS = 50
    else:
        GENERATIONS = 50

CACHE_STREAM_ROUTE = {}
TOTAL_OBJECT_COUNT = {"cost_function": 0}
GLOBAL_COUNTER = {'calls':0}

#########################################################################

def E2E_objective(all_routed_streams, all_streams, is_normalize=True, is_stats=False, bandwidth = None,
                  find_ratio_scheduled=False):

    GLOBAL_COUNTER['calls'] +=1

    if np.random.rand() < 0.0025:
        print( "[ LOG ] : GLOBAL_COUNTER", GLOBAL_COUNTER['calls'])

    objective_val = 0

    if is_stats: delay_violations = 0
    for idx, (route, stream)  in enumerate(zip(all_routed_streams, all_streams)):

        scale_delay = 1
        if is_normalize:
            scale_delay = (np.floor(stream[0].delay_cycles))
        objective_val += (1/len(all_routed_streams)) * (E2E_delay(route, stream)/scale_delay)

        delayed_streams = []
        if E2E_delay(route, stream) > int(np.floor(stream[0].delay_cycles)):
            objective_val += 2
            if is_stats: delay_violations +=1
            delayed_streams.append(stream[0].id)

    if bandwidth is None:
        bandwidth = 600

    timey = timer()
    utilization, pc = BU_bandwidth_utilization3(all_routed_streams, all_streams, bandwidth, ignore_streams=delayed_streams)
    objective_val += 2*pc

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
    GLOBAL_COUNTER['calls'] = 0

def BU_bandwidth_utilization3(all_routes, all_streams, penalty_bw_threshold=None, ignore_streams=[]):
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


def E2E_delay(route, stream):

    e2e = 0
    # first end station propagation to the switch is an additional one cycle
    e2e_propagation_cycles = 1

    stream_len = stream[1][route[0]][0]
    e2e_propagation_cycles += sum(route[1:1+stream_len])
    e2e_propagation_cycles += route[-1]

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

# Function to generate an initial population
def generate_population(PG):
    population = []
    init_member = copy.deepcopy(PG.solution)
    second_member = copy.deepcopy(PG.get_balanced_naive_solution())
    third_member = copy.deepcopy(PG.get_balanced_naive_solution(True))

    population.append(init_member)
    population.append(second_member[:])

    for _ in range(POPULATION_SIZE-3):
        PG.GenerateRandomIndividual()

        individual = copy.deepcopy(PG.solution)
        population.append(individual)
    return population

# Function to evaluate an individual
def evaluate(PG, individual):
    bandwidth = PG.bandwidth

    return E2E_objective(individual, PG.space, bandwidth=bandwidth)

# Function to perform mutation on an individual
def mutate2(PG, individual):

    if random.random() < MUTATION_RATE:
        individual = PG.MutateSolution(individual, 1+np.random.randint(NO_MUTATION_GENES))

    return copy.deepcopy(individual)

def mutate(PG, individual, sa_mut = False):

    if not ASAGA:
        return mutate2(PG, individual)

    if (random.random() < MUTATION_RATE) or sa_mut:
        PG.solution = copy.deepcopy(individual[:])
        individual = copy.deepcopy(simulated_annealing(PG))
    if random.random() < MUTATION_RATE:
        individual = PG.MutateSolution(individual, np.random.randint(NO_MUTATION_GENES))
    if COMPLEX_GENES:
        if np.random.random() < 0.5:
            individual = PG.MutateSolution(individual, np.random.randint(NO_MUTATION_GENES))

    return copy.deepcopy(individual)

# Function to perform pairing of individuals for crossover
def pair(population):
    pairs = []
    for _ in range(POPULATION_SIZE // 2):
        parent1 = random.choice(population)
        parent2 = random.choice(population)
        pairs.append((parent1, parent2))
    return pairs

# Function to perform crossover
def crossover(parent1, parent2):
    child1, child2 = [], []

    if COMPLEX_GENES:
        return crossover2(parent1, parent2)

    for el1,el2 in zip(parent1,parent2):
        child1.append([])
        child2.append([])
        for val1,val2 in zip(el1,el2):
            child1[-1].append(random.choice([val1,val2]))
            child2[-1].append(random.choice([val1,val2]))

    return child1, child2

def crossover2(parent1, parent2):
    child1, child2 = [], []
    for el1,el2 in zip(parent1,parent2):
        idx1, idx2 = np.random.randint(2), np.random.randint(2)
        el = [el1,el2]
        child1.append(el[idx1])
        child2.append(el[idx2])

    return child1, child2

# Function to run the genetic algorithm
def genetic_algorithm(priority_group):
    PG = priority_group
    population = generate_population(PG)

    termination_time = timer()

    if MEAS_CONVERGENCE:
        obj = E2E_objective(PG.solution, PG.space, True, True, bandwidth=PG.bandwidth, find_ratio_scheduled=True)
        pgidx = PG.k_priority
        PG.export_full_routes_and_schedules(PG.solution, "k: "+str(pgidx)+" "+str(obj[1])+'\n', convergence= 99)
        PG.export_full_routes_and_schedules_csv(PG.solution, obj[-1][-1]+obj[-1][-3] , convergence = 99)

    most_fit_individual_score = 1e10
    timex = time.time()
    for generation in range(GENERATIONS):
        if True in [np.isclose((x:=generation/GENERATIONS),0.05*i) for i in range(21) ]:
            duration = time.time()-timex
            print(int(100*x),"% finished, ", "running for:", round(duration/60), "(mins), group:", PG.k_priority)
        # Evaluate the fitness of each individual in the population
        fitness_scores = [evaluate(PG, individual) for individual in population]

        # Select the best individuals for the next generation
        elites = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=False)[:len(population)//2]
        next_generation = [individual for individual, _ in elites]

        # Perform pairing and crossover to create the rest of the next generation
        pairs = pair(population)
        for parent1, parent2 in pairs:
            child1, child2 = crossover(parent1, parent2)
            next_generation.extend([child1, child2])

        # Perform mutation on the next generation
        next_generation = [mutate( PG, individual,sa_mut=idx<2 ) for (idx,individual) in enumerate(next_generation)]
        #timet = timer()-timet

        # Replace the current population with the next generation
        population = next_generation

        best_current_generation_individual = min(zip(population, fitness_scores), key=lambda x: x[1])
        if best_current_generation_individual[1] < most_fit_individual_score:
            best_all_generation_individual = copy.deepcopy(best_current_generation_individual)

        if MEAS_TIME:
            if GLOBAL_COUNTER['calls'] > 1250:
                with open('temp_obj.txt', 'a') as f:
                    f.write(str(generation) + " generation, time(s): "
                            + (str(round(timer() - termination_time, 3))) + '\n')
                    GLOBAL_COUNTER['calls'] = 0
                    break

        if MEAS_CONVERGENCE:
            PG.solution = copy.deepcopy(best_all_generation_individual[0])
            with open('temp_obj.txt','a') as f:
                f.write(str(generation) + " generation, objective: "
                     + (str(round(best_all_generation_individual[1],5))) + '\n')

            obj = E2E_objective(PG.solution, PG.space, True, True, bandwidth=PG.bandwidth, find_ratio_scheduled=True)
            pgidx = PG.k_priority
            PG.export_full_routes_and_schedules(PG.solution, "k: "+str(pgidx)+" "+str(obj[1])+'\n', convergence= generation)
            PG.export_full_routes_and_schedules_csv(PG.solution, obj[-1][-1]+obj[-1][-3] , convergence = generation)
            if timer() - timex > 50*60:
                break

    # Return the best individual from the last generation
    best_individual = min(zip(population, fitness_scores), key=lambda x: x[1])[0]
    best_last_generation_individual = min(zip(population, fitness_scores), key=lambda x: x[1])


    return best_individual

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


def simulated_annealing(network_model, T_0=1e2, cooling_rate=0.99, is_log_stats=False, bandwidth=None):

    nm = network_model
    streams = network_model.space
    bw = nm.bandwidth

    t = T_0
    t = 4
    t = SA_INIT_TEMP
    cr = cooling_rate
    sol = copy.deepcopy(nm.solution[:])
    comparative_solution = copy.deepcopy(nm.solution[:])

    sol_final = copy.deepcopy(sol[:])
    cost_final = E2E_objective(sol_final, streams, bandwidth=bw)

    e2e_current = cost_final

    no_iterations = 2*int(1e1)//1
    no_iterations = NO_ITERATIONS

    improving_moves = 0
    jump_stats_new = 0
    jump_stats_old = 0

    if is_log_stats: algo_temp_jumps = 0
    old_solution = copy.deepcopy(nm.solution[:])
    e2e_current = E2E_objective(copy.deepcopy(old_solution[:]), streams, bandwidth=bw)
    termination_time = timer()
    if MEAS_TIME_SA:
        no_iterations = int(10e4)
    for iter_idx in range(no_iterations):

        old_solution = copy.deepcopy(nm.solution[:])
        is_new = nm.GenerateNeighbour()
        new_solution = copy.deepcopy(nm.solution[:])
        e2e_temp = E2E_objective(copy.deepcopy(new_solution[:]), streams, bandwidth=bw)
        if e2e_temp < e2e_current:
            improving_moves +=1
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

        ##############################################################################

        if MEAS_TIME_SA:
            if GLOBAL_COUNTER['calls'] > 1250:
                with open('temp_obj.txt', 'a') as f:
                    f.write(str(iter_idx) + " iteration, time(s): "
                         + (str(round(timer()-termination_time,3))) + '\n')
                    GLOBAL_COUNTER['calls'] = 0
                    break

        ##############################################################################

    if np.random.rand() < .1: print("improving moves count: ", improving_moves)
    return sol_final[:]


if __name__ == '__main__':

    # Run single group
    graph_data_raw, flow_data_raw = load_graph_stream_data(topo_filepath, flows_filepath)
    topology_data = process_graph_data(graph_data_raw)
    all_streams_data = process_flow_data(flow_data_raw)
    GM = GroupModel(topology_data, all_streams_data, priority_groups, priority_queues, cycle_times, None)

    # Run the genetic algorithm
    best_individual = genetic_algorithm(GM)
    for _ in range(5): print()
    print("Best Individual:", best_individual)

    GM.show_solution(best_individual)

    GM.show_full_routes_and_schedules(best_individual)
    E2E_objective(best_individual, GM.space, True, True)
    print("done")

    pass

