import sys

sys.path.append("../")

from config import config
from nc_model import LGD, clean_GPU_memory, create_folder
import numpy as np
import optuna

# set configs
config.datname = "blogcatalog"
config.cur_ddir = "{}social_networks/".format(config.datadir)
config.graph_type = "knn"

config.hpm_opt = True
config.is_print = False
config.is_prepare = False
config.is_sav_model = False

max_evals = 200

# fix hyper-parameters
ncaps = 4
nhidden = 16
routit = 7

# meta-preparation
meta_mdir = "{}{}_meta/".format(config.modeldir, config.datname)
create_folder(meta_mdir)
logfile = open("{}logfile.txt".format(meta_mdir), "w+")

# meta-initialization
trail_step = 0
META_SAV_LIST = []


def log_trial(str, file, end="\r\n"):
    print(str, end=end)
    print(str, file=file, end=end)
    file.flush()


def dic_to_class(dic):
    class hyperpm(object):
        for k, v in dic.items():
            if int(v) == v:
                locals()[k] = int(v)
            elif v > 1:
                locals()[k] = float("%.2e" % float(v))
            elif v > 0.1:
                locals()[k] = float("%.1e" % float(v))
            elif (k == "lr" or k == "reg"):
                locals()[k] = float("%.e" % float(v))
            else:
                locals()[k] = float("%.1e" % float(v))

    # delete the irrelevant attributes
    delattr(hyperpm, "k")
    delattr(hyperpm, "v")
    return hyperpm


def class_to_dic(cls):
    dic = {}
    for a in dir(cls):
        if not a.startswith("__"):
            dic[a] = getattr(cls, a)
    return dic


def dic_to_str(dic, end="\r\n"):
    str_line = ""
    for k, v in dic.items():
        str_line += "{}={} ".format(k, str(v))
    str_line += end
    return str_line


def class_to_str(cls):
    str_line = ""
    for a in dir(cls):
        if not a.startswith("__"):
            str_line += "{}={}, ".format(a, getattr(cls, a))
    return str_line[:-2]


def get_rnd_seed():
    # return a random seed in 6 numbers
    return np.random.randint(99999, 999999)


def objective(trial):
    # define params
    global trail_step, META_SAV_LIST
    trail_step += 1
    # sampling space
    hyperpm_dic = {"reg": trial.suggest_uniform("reg", 1e-4, 5e-1),
                   "lr": trial.suggest_uniform("lr", 1e-3, 5e-1),
                   "nlayer": trial.suggest_int("nlayer", 1, 3),
                   "dropout": trial.suggest_float("dropout", 0, 1, step=0.05),
                   "gm_update_rate": trial.suggest_uniform("gm_update_rate", 0.1, 0.9),
                   "latent_nnb_k": trial.suggest_int("latent_nnb_k", 1, 20),
                   "space_lambda": trial.suggest_uniform("space_lambda", 0.1, 1),
                   "div_lambda": trial.suggest_uniform("div_lambda", 0.001, 0.1),
                   "ncaps": ncaps,
                   "nhidden": nhidden,
                   "routit": routit}
    # run model
    hyperpm = dic_to_class(hyperpm_dic)
    config.rnd_seed = get_rnd_seed()
    val_acc, tst_acc, epochs = LGD(config, hyperpm)
    # log result
    result_dic = class_to_dic(hyperpm)
    log_trial("trail={}/{}, meta_seed={}, meta_val_acc={:.4f}%, meta_tst_acc={:.4f}%, meta_epochs={} @ {}".format(
        trail_step, max_evals, config.rnd_seed, val_acc * 100, tst_acc * 100, epochs, dic_to_str(result_dic)), file=logfile)
    # save result
    result_dic["seed"] = config.rnd_seed
    result_dic["val_acc"] = val_acc
    result_dic["tst_acc"] = tst_acc
    META_SAV_LIST.append(result_dic)
    clean_GPU_memory()
    return val_acc


def main():
    log_trial("Tuning Hyper-params on dataset-{} in {} evals".format(config.datname, max_evals), file=logfile)
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction="maximize")
    study.optimize(objective, n_trials=max_evals)
    best_trial = study.best_trial
    best_hyperpm_dic = best_trial.params
    best_hyperpm_dic["ncaps"] = ncaps
    best_hyperpm_dic["nhidden"] = nhidden
    best_hyperpm_dic["routit"] = routit
    best_hyperpm = dic_to_class(best_hyperpm_dic)
    log_trial("Final-Selection:\r\n{}\r\nbest-val-acc={:.4f}%".format(class_to_str(best_hyperpm), best_trial.value * 100), file=logfile)
    np.save("{}meta_results.npy".format(meta_mdir), META_SAV_LIST)  # save all the experimental results


if __name__ == '__main__':
    main()
