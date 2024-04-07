import datetime
from NetworkModel import run_network_model
from GaAlgorithm import (load_graph_stream_data, process_graph_data, process_flow_data,
                              reset_global_cache, E2E_objective)


RUNNAME = '1gbps GA-TI 125 250 500 ER'
from time import time

if __name__ == '__main__':

    for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]: # erg

        reset_global_cache()
        t0 = time()
        print("the time of start is", t0)

        topo_filepath = "..\TC\TCs\TC" + str(i) + "_topo.txt"
        flows_filepath = "..\TC\TCs\TC" + str(i) + "_flows.txt"

        # ErdosRenyi
        now = datetime.datetime.now()
        date_and_hour = now.strftime("_%Y-%m-%d_%H-%M-%S_50_125_250_1Gbits_GA_TI")
        result_name = topo_filepath.split('\\''')[-1].split('.')[0] + date_and_hour + RUNNAME

        graph_data_raw, flow_data_raw = load_graph_stream_data(topo_filepath, flows_filepath)

        td = process_graph_data(graph_data_raw)
        asd = process_flow_data(flow_data_raw)[:]

        NM, group_solution = run_network_model(asd, td, result_name, is_use_SA=False)

        for PrioGroup, sol, pgidx in zip(NM, group_solution, [1, 2, 3]):
            obj = E2E_objective(sol, PrioGroup.space, True, True, bandwidth=PrioGroup.bandwidth, find_ratio_scheduled=True)
            PrioGroup.export_full_routes_and_schedules(sol, "k: " + str(pgidx) + " " + str(obj[1]) + '\n')
            PrioGroup.export_full_routes_and_schedules_csv(sol, obj[-1][-1]+obj[-1][-3] )
            print("done")
            t2 = time()
            with open('results/time'+ str(i) + '_time.txt', 'w') as file:
                file.write("start time is in secs:\n")
                file.write(f"{t0}\n")
                file.write("stop time is in secs:\n")
                file.write(f"{t2}\n")
                file.write("total runtime is in secs:\n")
                file.write(f"{t2 - t0}\n")
