class hyperpm(object):
    routit = None
    dropout = None
    lr = None
    reg = None
    nlayer = None
    ncaps = None
    nhidden = None
    latent_nnb_k = None
    gm_update_rate = None
    space_lambda = None
    div_lambda = None


class config(object):
    datname = None
    datadir = "../data/dataset/"
    modeldir = "../data/model/"
    cur_ddir = None
    cur_mdir = None

    nepoch = 1000  # max-training-epochs
    early = 100  # patience for early-stopping
    nbsz = 50  # size of the sampled node neighbors for efficient training
    rnd_seed = None

    hpm_opt = False  # whether in the mode of hyper-params' opt
    cpu = False
    record_tst = True
    is_print = True
    is_sav_model = False

    is_rdsp = False
    rd_trn_idx = None
    rd_val_idx = None
    rd_tst_idx = None
