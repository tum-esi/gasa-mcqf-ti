import os
import copy
import datetime
import numpy as np
from random import choice
np.set_printoptions(linewidth=100)
import heapq

class GroupModel():

    def __init__(self, topology_data, streams_data, idx_priority_group, queue_list, cycle_time, n_neighbours,
                 bandwidth = 1000e6, export_name = None, injection_cycles=10):
        """ Initialize network model"""

        self.k_priority = idx_priority_group
        self.queue_index = len(queue_list[idx_priority_group-1])
        self.cycle_times = cycle_time
        self.n_neighbours = 5
        self.injection_cycles = injection_cycles // idx_priority_group + 1
        self.queue_list = queue_list
        self.bandwidth = bandwidth*cycle_time/1e6

        self.graph = self.generate_graph(topology_data)
        self.paths = self.graph.quick_paths_access
        self.space = self.generate_rs_space(streams_data)
        self.GenerateInitialSolution()

        self.show_full_routes_and_schedules()
        self.export_name = export_name

    def MutateSolution(self, individual, no_mutation_steps=1):
        cache_solution = copy.deepcopy(self.solution)
        self.solution = copy.deepcopy(individual)
        for _ in range(no_mutation_steps):
            self.GenerateNeighbour()
        retval = copy.deepcopy(self.solution)
        self.solution = copy.deepcopy(cache_solution)
        return copy.deepcopy(retval)

    def GenerateNeighbour(self):

        is_changed = False

        stream_index = np.random.random_integers(len(self.solution))-1
        stream = self.space[stream_index]
        assignment = self.solution[stream_index]
        space = self.solution_space[stream_index]

        path_possibility = len(stream[1])
        queue_possibility = stream[0].q_n-1
        delay_possibility = space[-1][1]+1

        if path_possibility == 1 and queue_possibility == 1 and delay_possibility == 1:
            is_changed = False
            return is_changed

        # TODO check selection of path, queue, delay
        random_pertrubation_space = [ True for i in range(len(assignment))]
        if path_possibility == 1:
            random_pertrubation_space[0] = False
        if queue_possibility == 1:
            random_pertrubation_space[1:-1] = [ False for i in range(len(assignment[1:-1]))]
        else:
            current_stream = stream[1][assignment[0]]
            path_len = current_stream[0]
            random_pertrubation_space[1:-1] = [ True if el < path_len else False
                                                for el in range(len(assignment[1:-1]))]
        if delay_possibility == 1:
            random_pertrubation_space[-1] = False

        random_index = np.random.random_integers(sum(random_pertrubation_space))-1
        temp = 0
        for i in range(len(random_pertrubation_space)):
            if random_pertrubation_space[i]:
                random_pertrubation_space[i] = temp
                temp +=1
            else:
                random_pertrubation_space[i] = -1
        chosen_index = random_pertrubation_space.index(random_index)
        bot, top = space[chosen_index][0], space[chosen_index][1]+1
        try:
            new_value  = choice([i for i in range(bot, int(top)) if i not in [assignment[chosen_index]]])
        except Exception as e:
            print(e)
        assignment[chosen_index] = new_value

        is_changed = True
        return is_changed

    def GenerateRandomIndividual(self):

        is_changed = False

        for idx in range(len(self.solution)):

            stream_index = idx
            stream = self.space[stream_index]
            assignment = self.solution[stream_index]
            space = self.solution_space[stream_index]
            path_possibility = len(stream[1])
            queue_possibility = stream[0].q_n-1
            delay_possibility = space[-1][1]+1

            if path_possibility == 1 and queue_possibility == 1 and delay_possibility == 1:
                is_changed = False
                return is_changed

            random_pertrubation_space = [ True for i in range(len(assignment))]
            if path_possibility == 1:
                random_pertrubation_space[0] = False
            if queue_possibility == 1:
                random_pertrubation_space[1:-1] = [ False for i in range(len(assignment[1:-1]))]
            if delay_possibility == 1:
                random_pertrubation_space[-1] = False

            random_index = np.random.random_integers(sum(random_pertrubation_space))-1
            temp = 0
            for i in range(len(random_pertrubation_space)):
                if random_pertrubation_space[i]:
                    new_value  = choice(range(space[i][0],space[i][1]+1))
                    assignment[i] = new_value

        is_changed = True
        return is_changed

    def GenerateInitialSolution(self):
        self.solution = []
        self.solution_space = []
        for el in self.space:
            path_assignment = 0
            max_links_assignment = el[1][-1][0]
            no_links_assignment = el[1][0][0]
            no_links_unusable = max_links_assignment-no_links_assignment
            queue_assignment = [1]*(no_links_assignment) + [1]*(no_links_unusable)
            delay_assignment = 0
            max_delay_cycles = 0
            self.solution.append([path_assignment,*queue_assignment, delay_assignment])
            self.solution_space.append([[0,len(el[1])-1],
                                       *[[1,el[0].q_n-1] for i in range(no_links_assignment)],
                                       *[[1,el[0].q_n-1] for i in range(no_links_unusable)],
                                       [0,min(el[0].period_cycles-1, self.injection_cycles)]])

    def get_balanced_naive_solution(self, time_injection_balance=False):
        self.solution = []
        self.solution_space = []
        for el in self.space:
            path_assignment = 0
            max_links_assignment = el[1][-1][0]
            no_links_assignment = el[1][0][0]
            no_links_unusable = max_links_assignment-no_links_assignment
            queue_assignment = [1]*(no_links_assignment) + [1]*(no_links_unusable)
            queue_assignment[0] = choice(range(1,el[0].q_n))
            delay_assignment = 0
            if time_injection_balance:
                try:
                    delay_assignment = choice(range(int(min(el[0].period_cycles, self.injection_cycles))))
                except:
                    delay_assignment = 0

            max_delay_cycles = 0
            self.solution.append([path_assignment,*queue_assignment, delay_assignment])
            self.solution_space.append([[0,len(el[1])-1],
                                       *[[1,el[0].q_n-1] for i in range(no_links_assignment)],
                                       *[[1,el[0].q_n-1] for i in range(no_links_unusable)],
                                       [0,int(min(el[0].period_cycles-1, self.injection_cycles))]])
        return copy.deepcopy(self.solution[:])

    def generate_graph(self, topology):

        nodes = topology['nodes']
        edges = topology['edges']

        all_vertices = []
        lut_vertices = {}
        for idx, node in enumerate(nodes):
            if node[1] == 'SWITCH':
                all_vertices.append(Vertex(idx,'sw',node[2],node[4], node[6]))
                lut_vertices[node[2]] = idx
            if node[1] == 'PLC':
                all_vertices.append(Vertex(idx,'es',node[2],node[4], node[6]))
                lut_vertices[node[2]] = idx

        all_edges = []
        for idx, edge in enumerate(edges):
            if '.' in edge[2]:
                node_start, port_start = edge[2].split('.')
            else:
                node_start = edge[2]
                port_start = 'ES'
            if '.' in edge[3]:
                node_end, port_end = edge[3].split('.')
            else:
                node_end = edge[3]
                port_end = 'ES'

            node_start_idx = lut_vertices[node_start]
            node_end_idx = lut_vertices[node_end]
            all_edges.append(Link(node_start_idx,node_end_idx, port_start, port_end, self.k_priority, self.queue_index,
                                  name=edge[-1]))


        graph = Graph(all_vertices, all_edges, self.n_neighbours)
        self.all_vertices = all_vertices
        self.lut_vertices = lut_vertices
        return graph

    def generate_rs_space(self, streams_data):

        rs_space = []
        for idx,stream in enumerate(streams_data):
            name = stream[3]
            talker = stream[5]
            receiver = stream[6]
            period = stream[8]
            cycle = self.cycle_times
            max_delay = stream[10]
            size_in_bytes = stream[12]
            stream = Stream(int(stream[2]), name, self.k_priority, self.queue_index, talker,receiver, size_in_bytes, period , cycle, max_delay)
            rs_space.append((stream, self.paths[talker+'->'+receiver]))

        return rs_space

    def show_solution(self, individual):

        for idx,rs in enumerate(individual):
            print(self.space[idx])
            print(self.space[idx][1][rs[0]], "queue assignment:", rs[1:-1], "planned time injection: ", rs[-1])

    def show_full_routes_and_schedules(self, sol = None):

        if sol is None:
            sol = copy.deepcopy(self.solution)

        for route, stream in zip(sol, self.space):
            hops = stream[1][route[0]][0]
            path = stream[1][route[0]][1]
            schedule = route[1:1+hops]
            cycle_injection = route[-1]
            path_ids = [self.graph.all_vertices[el].id for el in path]
            current_cycle = -1
            priority_group = -1 + stream[0].k
            queues = self.queue_list[priority_group]
            queues_chosen = []
            for queue_delay in schedule:
                current_cycle += queue_delay
                queues_chosen.append(queues[current_cycle % len(queues)])
            print(stream[0], "| injection delay:", cycle_injection ,path_ids, queues_chosen )
        print("\n")

    def export_full_routes_and_schedules(self, sol = None, scheduled_line='', convergence = None):

        if convergence is not None:
            self.export_name = self.export_name.split('__')[0]
            self.export_name += '__'+str(convergence)

        if sol is None:
            sol = copy.deepcopy(self.solution)

        with open("results/export_log_"+self.export_name+".txt", 'a+') as f:
            for route, stream in zip(sol, self.space):
                hops = stream[1][route[0]][0]
                path = stream[1][route[0]][1]
                schedule = route[1:1+hops]
                cycle_injection = route[-1]
                path_ids = [self.graph.all_vertices[el].id for el in path]
                current_cycle = -1
                priority_group = -1 + stream[0].k
                queues = self.queue_list[priority_group]
                queues_chosen = []
                for queue_delay in schedule:
                    current_cycle += queue_delay
                    queues_chosen.append(queues[current_cycle % len(queues)])
                f.write(str(stream[0])+"| injection delay:"+str(cycle_injection)+str(path_ids)+str(queues_chosen))
                f.write('\n')

        if scheduled_line:
            with open("results/export_log_"+self.export_name+".txt", 'r') as file:
                contents = file.read()
            updated_contents = (scheduled_line + " cycle_time: "+ str(self.cycle_times) +
                                "us bandwidth: " + str(round(self.bandwidth,2)) + "bytes " + contents)
            with open("results/export_log_"+self.export_name+".txt", 'w') as file:
                if updated_contents[3:4] == '3':
                    total = [int(updated_contents.split('\n')[j].split(',')[i]) for j in [0,1,2] for i in [3, 5]]
                    total_scheduled = total[1]+total[3]+total[5]-total[0]-total[2]-total[4]
                    total_streams = total[1] + total[3] + total[5]
                    header = (self.export_name.split('_')[0]+" scheduled/num_streams "+
                              str(total_scheduled)+"/"+str(total_streams)+" = "+
                              str(round(100*total_scheduled/total_streams,2))+"%\n")
                    updated_contents = header+updated_contents
                file.write(updated_contents)

    def E2E_delay_in_cycles(self,route, stream):
        e2e = 0
        stream_len = stream[1][route[0]][0]
        e2e_propagation_cycles = sum(route[1:1+stream_len])
        e2e_propagation_cycles += route[-1]

        e2e_propagation_cycles +=1
        return e2e_propagation_cycles

    def export_full_routes_and_schedules_csv(self, sol = None, misscheduled_idx=[], convergence=None ):

        if sol is None:
            sol = copy.deepcopy(self.solution)

        with open("results/export_log_"+self.export_name+".csv", 'a+') as f:
            for idx, (route, stream) in enumerate(zip(sol, self.space)):
                if stream[0].id in misscheduled_idx:
                    continue
                name = stream[0].name
                hops = stream[1][route[0]][0]
                maxE2Edelay = str(self.E2E_delay_in_cycles(route,stream)*self.cycle_times)
                last_hop_delay = round(int(stream[0].size_byte) / (self.bandwidth / 10), 2)
                deadline = str(int(stream[0].delay_cycles*self.cycle_times))

                path = stream[1][route[0]][1]
                schedule = route[1:1+hops]
                cycle_injection = route[-1]
                path_ids = [self.graph.all_vertices[el].id for el in path]
                link_ids = []
                port_ids = []
                for (n1,n2) in zip(path[0:-1],path[1:]):
                    for lidx, link in enumerate(self.graph.all_links):
                        if [n1, n2] == link.e:
                            link_ids.append(link.name)
                            port_ids.append(link.p[0])
                            break
                        if [n2,n1] == link.e:
                            link_ids.append(link.name)
                            port_ids.append(link.p[1])
                            break
                current_cycle = -1+cycle_injection
                priority_group = -1 + stream[0].k
                queues = self.queue_list[priority_group]

                queues_chosen = []
                for queue_delay in schedule:
                    current_cycle += queue_delay
                    queues_chosen.append(queues[current_cycle % len(queues)])
                current_cycle += 1

                path_str = '|'.join(['|'.join([el[3]+'-'+el[0],el[1],el[2].strip('q')]) for el in zip(path_ids[1:],link_ids[1:],queues_chosen,port_ids[1:]) ])
                path_str = path_ids[0]+'|'+path_str+'|'+path_ids[-1]

                f.write(','.join([name,maxE2Edelay,deadline,path_str]))
                f.write('\n')
        try:
            if self.k_priority == 3:
                with open("results/export_log_"+self.export_name+".csv", 'r') as file:
                    contents = file.read()
                    contents = contents.split('\n')
                    contents = contents[:-1]
                    contents = sorted(contents,key = lambda el: int(el.split(',')[0].split('_')[-1]) )
                with open("results/res_"+self.export_name+"_sorted.csv",'w') as file:
                    file.write("FlowName,maxE2E(us),Deadline(us),Path(SourceName|LinkID|Qnumber)\n")
                    file.write('\n'.join(contents))
                os.remove("results/export_log_"+self.export_name+".csv")
        except Exception as e:
            print("[ ERROR ] Could not sort: " + str(e))


class RoutedStream():

    def __init__(self, route, stream):
        pass

class Stream():

    def __init__(self, id, name,  k_priority, no_queue, v_s_node, v_d_node, b_size_byte, t_ms, t_cycle, d_max_delay):

        self.id = id
        self.name = name
        self.k = k_priority
        self.q_n = no_queue
        self.v_s = v_s_node
        self.v_d = v_d_node
        self.size_byte = b_size_byte
        self.period_cycles = float(t_ms)/t_cycle
        self.delay_cycles = float(d_max_delay)/t_cycle

    def Arrival(self, cycle):
        arrival = self.b if not cycle % self.t_ms else 0
        return arrival

    def __repr__(self):
        return "Stream: id {} | k_group {} | q_n {} | {}->{} | period cyc {}| deadline {} | size {} ".format(
            self.id, self.k, self.q_n, self.v_s, self.v_d, self.period_cycles, self.delay_cycles, self.size_byte)

    def __str__(self):
        return "Stream: id {} | k_group {} | q_n {} | {}->{} | period {}| deadline {} | size {}".format(
            self.id, self.k, self.q_n, self.v_s, self.v_d, self.period_cycles, self.delay_cycles, self.size_byte)


class AllRoutes():

    pass
    def latency_T(self):

        latency = 0
        for link in self.route:
            latency += link.cycle_propagation_delay + link.assigned_queing_delay

        latency += route.link

class Vertex():
    def __init__(self, device_idx, device_type, device_id, address, port_number):

        self.device_idx = device_idx
        self.dt = device_type
        self.id = device_id
        self.address = address
        self.port = port_number
        assert(device_type == 'es' or device_type == 'sw'), (
               ' node/vertex must be either an ES ( end station ) or a SW (switch) ')

    def __repr__(self):
        return "Vertex: {} | {}".format(self.device_idx, self.address)

    def __str__(self):
        return "Vertex: Idx {} | Type {} | Id {} | Address {} | Ports {}".format(
                      self.device_idx,self.dt,self.id,self.address,self.port)

class Link():

    def __init__(self, node_u, node_v, port_u, port_v, k_priority, queue_index,name=''):
        self.e = [node_u, node_v]
        self.p = [port_u, port_v]
        self.k = k_priority
        self.q = queue_index
        self.name = name

    def __str__(self):
        return "edge_{}->{}, k_priority_group: {}, queue_index: {}".format(self.e[0],self.e[1],self.k,self.q)

    def __repr__(self):
        return "edge_{}->{}, ports_{}->{}, k_priority_group: {}, queue_index: {}".format(
                self.e[0],self.e[1],self.p[0],self.p[1],self.k,self.q)

class Graph:

    def __init__(self, all_vertices, all_links, n_neighbours):

        self._is_space_created = False
        self.ES_routes_space = None

        self.all_vertices = all_vertices
        self.all_links = all_links
        self.n = n_neighbours

        num_vertices = len(all_vertices)
        self.num_vertices = num_vertices

        # Create and Fill the Adjacency Matrix
        self.adjacency_matrix = [[float('inf')] * num_vertices for _ in range(num_vertices)]
        for link in all_links:
            self.add_link(*link.e)

        # Generate Search Space
        self.end_stations = list(filter(lambda el: el.dt == 'es', all_vertices))
        self.ES_routes_space = []
        for i, start_vertex in enumerate(self.end_stations):
            for j, end_vertex in enumerate(self.end_stations):
                if i >= j:
                    continue
                self.ES_routes_space.append([start_vertex, end_vertex,
                                            self.n_shortest_paths(start_vertex, end_vertex, self.n)])

        self.add_backward_ES_routes()
        self._is_space_created = True

        self.quick_paths_access = {}
        for el in self.ES_routes_space:
            self.quick_paths_access[el[0].id+'->'+el[1].id]=el[2]

    def add_backward_ES_routes(self):

        backward_routes = []
        for route in self.ES_routes_space:
            backward_routes.append([route[1],route[0],[(el[0], el[1][::-1]) for el in route[2]]])

        self.ES_routes_space += backward_routes
        print("created_graph")


    def add_link(self, u, v):
        self.adjacency_matrix[u][v] = 1
        self.adjacency_matrix[v][u] = 1

    def n_shortest_paths(self, start, end, n):
        # Priority queue to store the shortest paths
        paths = []
        # Heap to store the nodes to be explored
        heap = [(-1, start.device_idx, [start.device_idx])]
        while heap and len(paths) < n:
            # Pop the node with the smallest distance from the heap
            dist, node, path = heapq.heappop(heap)
            # If the current node is the destination, add the path to the result
            if node == end.device_idx:
                paths.append((dist, path))
                continue
            # Explore the neighbors of the current node
            for neighbor in range(self.num_vertices):
                #print(neighbor)
                if self.adjacency_matrix[node][neighbor] != float('inf'):
                    # Calculate the new distance
                    new_dist = dist + self.adjacency_matrix[node][neighbor]
                    # Prevent paths from making loops
                    if neighbor in path:
                        continue
                    # Create a new path by extending the current path
                    new_path = path + [neighbor]
                    # Add the new path to the heap
                    heapq.heappush(heap, (new_dist, neighbor, new_path))
        return paths

    def get_path(self, start_vertex, end_vertex):

        self.ES_routes_space

    def __repr__(self):
        return str(np.array(self.adjacency_matrix))