
from easydict import EasyDict as edict

config = edict()
config.dataset = "kface" # training dataset
config.embedding_size = 512 # embedding size of model
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 4 # batch size per GPU
config.lr = 0.1
config.output = "/home/user3/ElasticFace/output/kface" # train model output folder
config.global_step=0 # step to resume
config.s=64.0
config.m=0.50
config.std=0.05


config.loss="ElasticArcFace"  #  Option : ElasticArcFace, ArcFace, ElasticCosFace, CosFace, MLLoss, ElasticArcFacePlus, ElasticCosFacePlus

if (config.loss=="ElasticArcFacePlus"):
    config.s = 64.0
    config.m = 0.50
    config.std = 0.0175
elif (config.loss=="ElasticArcFace"):
    config.s = 64.0
    config.m = 0.50
    config.std = 0.05
if (config.loss=="ElasticCosFacePlus"):
    config.s = 64.0
    config.m = 0.35
    config.std = 0.02
elif (config.loss=="ElasticCosFace"):
    config.s = 64.0
    config.m = 0.35
    config.std = 0.05


# type of network to train [iresnet100 | iresnet50]
config.network = "iresnet100"
config.SE=False # SEModule


if config.dataset == "emoreIresNet":
    config.rec = "datasets/faces_emore"
    config.num_classes = 85742
    config.num_image = 5822653
    config.num_epoch =  26
    config.warmup_epoch = -1
    config.val_targets =  ["lfw"]
    # config.val_targets =  ["lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw"]
    config.eval_step=5686
    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < -1 else 0.1 ** len(
            [m for m in [8, 14,20,25] if m - 1 <= epoch])  # [m for m in [8, 14,20,25] if m - 1 <= epoch])
    config.lr_func = lr_step_func

elif config.dataset == "webface":
    config.rec = "/home/user3/ElasticFace/align_test_binary"
    config.num_classes = 25
    config.num_image = 1845
    config.num_epoch = 20   #  [22, 30, 35]
    config.warmup_epoch = -1
    config.val_targets = ['testing']
    config.eval_step= 958 #33350
    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < config.warmup_epoch else 0.1 ** len(
            [m for m in [22, 30, 40] if m - 1 <= epoch])
    config.lr_func = lr_step_func

elif config.dataset == "kface":
    config.rec = "/home/user3/ElasticFace/train_binary"
    config.num_classes = 400
    config.num_image = 12800
    config.num_epoch = 20   #  [22, 30, 35]
    config.warmup_epoch = -1
    config.val_targets = ["kface"]
    config.eval_step= 1916
    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < config.warmup_epoch else 0.1 ** len(
            [m for m in [22, 30, 40] if m - 1 <= epoch])
    config.lr_func = lr_step_func