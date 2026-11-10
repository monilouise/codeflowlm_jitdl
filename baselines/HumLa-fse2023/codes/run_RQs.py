from data_stream.real_data_stream import data_id_2name
from main import run_HumLa, compute_human_PFs, compute_human_churn
from datetime import datetime
import shutil

VERBOSE, IS_PLOT = 0, False  # for run_HumLa
print("Please note that 100 runs with 10k test steps can take lots of time")
print("So one can set smaller N_TEST and SEEDs to have a quick check")
#N_TEST, SEEDS = 1720, range(5)  # setup in the manuscript (ant-ivy)
#N_TEST, SEEDS = 732, range(5)  # setup in the manuscript (commons-bcel)
#N_TEST, SEEDS = 727, range(5)  # setup in the manuscript (commons-codec)
#N_TEST, SEEDS = 1023, range(5)  # setup in the manuscript (commons-net)
#N_TEST, SEEDS = 812, range(5)  # setup in the manuscript (giraph)
#N_TEST, SEEDS = 534, range(5)  # setup in the manuscript (gora)
#N_TEST, SEEDS = 1064, range(5)  # setup in the manuscript (opennlp)
#N_TEST, SEEDS = 996, range(5)  # setup in the manuscript (commons-digester) (987 CP)
#N_TEST, SEEDS = 774, range(5)  # setup in the manuscript (commons-jcs)
#N_TEST, SEEDS = 578, range(5)  # setup in the manuscript (commons-validator)
#N_TEST, SEEDS = 1687, range(5)  # setup in the manuscript (commons-collections) (1658 CP)
#N_TEST, SEEDS = 2822, range(5)  # setup in the manuscript (commons-lang)
#N_TEST, SEEDS = 1619, range(5)  # setup in the manuscript (commons-compress)
#N_TEST, SEEDS = 3712, range(5)  # setup in the manuscript (commons-math)
#N_TEST, SEEDS = 521, range(5)  # setup in the manuscript (commons-scxml)
#N_TEST, SEEDS = 572, range(5)  # setup in the manuscript (commons-beanutils) (560 CP)
#N_TEST, SEEDS = 1082, range(5)  # setup in the manuscript (parquet-mr)
#N_TEST, SEEDS = 1740, range(5)  # setup in the manuscript (commons-configuration)
#N_TEST, SEEDS = 975, range(5)  # setup in the manuscript (commons-vfs)
N_TEST, SEEDS = 23690, range(5)  # setup in the manuscript (all_projects)

#N_TEST, SEEDS = 1000, range(2)  # for a quick run
#N_TEST, SEEDS = 611, range(100)

def RQ1(str_RQ1="RQ1.1", project_id=0, target_project_id=None):
    """ HumLa
    Replication of RQ1 that runs the waiting-time method and HumLa at varying levels of human noise and human effort.
    Input arguments:
        str_RQ1:        "RQ1.1", "RQ1.2", and "RQ1.3"
        project_id:     0, 1, ..., 15
    """

    str_RQ1 = str_RQ1.upper()
    if str_RQ1 == "RQ1.1":  # RQ1.1: HumLa vs waiting-time method
        #print("\n" + "==" * 30)
        #print("RQ1.1 for %s: PF of JIT-SDP with the waiting-time method vs HumLa:"
        #      % data_id_2name(project_id))
        # waiting-time method
        #run_HumLa({"has_human": False}, project_id, N_TEST, SEEDS, VERBOSE, IS_PLOT)
        # HumLa
        if target_project_id is not None:
            print("target project: %s" % data_id_2name(target_project_id))
        run_HumLa({"has_human": True, "human_err": 0, "human_eff": 1}, project_id, N_TEST, SEEDS, VERBOSE, IS_PLOT, target_project_id=target_project_id)

    elif str_RQ1 == "RQ1.2":  # RQ1.2: HumLa at varying levels of human noise
        print("\n" + "==" * 30)
        print("RQ1.2 for %s: average PF of HumLa at different human noise across test steps:"
              % data_id_2name(project_id))
        if target_project_id is not None:
            print("target project: %s" % data_id_2name(target_project_id))
        human_noises = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        for hm, noise in enumerate(human_noises):
            human_dict = {"has_human": True, "human_err": noise, "human_eff": 1}
            run_HumLa(human_dict, project_id, N_TEST, SEEDS, VERBOSE, IS_PLOT, target_project_id=target_project_id)

    elif str_RQ1 == "RQ1.3":  # RQ1.3: HumLa at varying levels of human effort
        print("\n" + "==" * 30)
        print("RQ1.3 for %s: PF of HumLa at different human efforts:"
              % data_id_2name(project_id))
        human_efforts = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        for hm, effort in enumerate(human_efforts):
            human_dict = {"has_human": True, "human_err": 0, "human_eff": effort}
            run_HumLa(human_dict, project_id, N_TEST, SEEDS, VERBOSE, IS_PLOT, target_project_id=target_project_id)
    else:
        raise Exception("Error in str_RQ=%s." % str_RQ1)


def RQ2(str_RQ2="RQ2.1", project_id=0, target_project_id=None):
    """ Eco-HumLa
    Replicate RQ2 that investigates the proposed Eco-HumLa.

    Input argument:
        str_RQ2:        "RQ2.1", "RQ2.2", and "RQ2.3"
        project_id:     0, 1, ..., 13
    """
    str_RQ2 = str_RQ2.upper()
    if str_RQ2 == "RQ2.1":  # RQ2.1: Eco-HumLa
        print("\n" + "==" * 30)
        print("RQ2.1 for %s: PF of Eco-HumLa vs HumLa at different efforts across #seeds=%d"
              % (data_id_2name(project_id), len(SEEDS)))
        if target_project_id is not None:
            print("target project: %s" % data_id_2name(target_project_id))
        # Eco-HumLa
        human_dict = {"has_human": True, "human_err": 0, "human_eff": "auto_ecohumla2"}
        run_HumLa(human_dict, project_id, N_TEST, SEEDS, VERBOSE, IS_PLOT, target_project_id=target_project_id)

        # HumLa at different human efforts
        ##human_efforts = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        #for hm, effort in enumerate(human_efforts):
        #    human_dict = {"has_human": True, "human_err": 0, "human_eff": effort}
        #    run_HumLa(human_dict, project_id, N_TEST, SEEDS, VERBOSE, IS_PLOT, target_project_id=target_project_id)

    elif str_RQ2 == "RQ2.2":  # RQ2.2: cumulative code churns from HumLa and Eco-HumLa
        print("\n" + "==" * 30)
        print("RQ2.2 for %s: cumulative code churn (in kilo) across #seeds=%d"
              % (data_id_2name(project_id), len(SEEDS)))
        # Eco-HumLa
        compute_human_churn("auto_ecohumla2", project_id, N_TEST, SEEDS)

        # HumLa at different levels of human effort
        human_efforts = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        for hm, effort in enumerate(human_efforts):
            compute_human_churn(effort, project_id, N_TEST, SEEDS)

    elif str_RQ2 == "RQ2.3":  # RQ2.3: recall 1 and false alarm for humans to investigate
        # ECo-HumLa
        eco_humla_recall1, eco_humla_false_alarm = compute_human_PFs("auto_ecohumla2", project_id, N_TEST, SEEDS)
        # HumLa at 50%-human effort
        humla50_recall1, humla50_false_alarm = compute_human_PFs(0.5, project_id, N_TEST, SEEDS)

        print("\n" + "==" * 25)
        print("RQ2.3 for %s: recall-1 and false-alarm for humans across #seeds=%d"
              % (data_id_2name(project_id), len(SEEDS)))
        print("\t Eco-HumLa: recall-1 = %.2f, false-alarm = %.2f" % (eco_humla_recall1, eco_humla_false_alarm))
        print("\t HumLa at 50%%-human effort: recall 1 = %.2f, false alarm = %.2f" % (humla50_recall1, humla50_false_alarm))

    else:
        raise Exception("Error in str_RQ=%s." % str_RQ2)

#TODO: Acrescentar |R1-R0| como métrica
if __name__ == "__main__":
    project_id = 33
    
    for i in range(21, 33):
        target_project_id = i
        #target_project_id = None
        RQ1("RQ1.1", project_id, target_project_id=target_project_id)
        #RQ1("RQ1.2", project_id) #tirar
        #RQ1("RQ1.3", project_id) #tirar
        #RQ2("RQ2.1", project_id, target_project_id=target_project_id)
        #RQ2("RQ2.2", project_id)
        #RQ2("RQ2.3", project_id)
        print("\nSuccess on %s\n" % datetime.today())
        dir_to_delete = r"C:\Users\Monique\Documents\UFPE\Doutorado\Tese\HumLa-fse2023\results\rslt.save\all_projects"
        if shutil.os.path.isdir(dir_to_delete):
            shutil.rmtree(dir_to_delete)
            print(f"Diretório apagado: {dir_to_delete}")
        else:
            print(f"Diretório não encontrado: {dir_to_delete}")
        
        

