import sys

sys.path.append("../")

from config import config
from nc_model import LGD, clean_GPU_memory
import numpy as np

config.datname = "blogcatalog"
config.cur_ddir = "{}social_networks/".format(config.datadir)
config.graph_type = "knn"  # algorithm for building new graphs from latent space
config.rnd_seed = 330269

config.record_tst = True
config.is_print = True
config.is_sav_model = True


class hyperpm(object):
    routit = 7
    dropout = 0.4
    lr = 7e-3
    reg = 0.01
    nlayer = 1
    ncaps = 4
    nhidden = 64 // ncaps
    latent_nnb_k = 10
    gm_update_rate = 0.6
    space_lambda = 1
    div_lambda = 0.019


def set_rdsplits():
    # load random-splits
    config.is_rdsp = True
    rdsps = np.load("../data/dataset/rdsp_idx/{}_rdspDidx.npy".format(config.datname), allow_pickle=True)
    assert len(rdsps) == 10
    split = rdsps[0]  # take the 1st one as example
    config.rd_trn_idx = split["trn_idx"]
    config.rd_val_idx = split["val_idx"]
    config.rd_tst_idx = split["tst_idx"]


def eval():
    # set_rdsplits()
    val_acc, tst_acc, epochs = LGD(config, hyperpm)
    print("dataset={}, val-acc={:.4f}%, tst-acc={:.4f}%, epochs={}".format(config.datname, val_acc * 100, tst_acc * 100, epochs))
    clean_GPU_memory()


def main():
    eval()


if __name__ == "__main__":
    main()
