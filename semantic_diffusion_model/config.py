from easydict import EasyDict as edict

__C = edict()
cfg = __C

#
# Dataset Config
#

__C.DATASETS = edict()
__C.DATASETS.DATADIR = 'augmented_camus/2CH_ED_augmented'
# inputs_dir = "/home/data/farid/simulated_images_cs_Demir_Yichen_Daniel/Daniel/Daniel/IFL-SET1-FILTERED/"
__C.DATASETS.SAVEDIR = 'output/diffusion' #'output/sdm_train'
__C.DATASETS.DATASET_MODE = 'camus'

__C.TRAIN = edict()
__C.TRAIN.DIFFUSION = edict()
__C.TRAIN.DIFFUSION.LEARN_SIGMA = True # We are learning the variances in the diffusion process
__C.TRAIN.DIFFUSION.NOISE_SCHEDULE = "cosine"
__C.TRAIN.DIFFUSION.TIMESTEP_RESPACING = ''
__C.TRAIN.DIFFUSION.USE_KL = False
__C.TRAIN.DIFFUSION.PREDICT_XSTART = False # why not?
__C.TRAIN.DIFFUSION.RESCALE_TIMESTEPS = False
__C.TRAIN.DIFFUSION.RESCALE_LEARNED_SIGMAS = False

__C.TRAIN.IMG_SIZE = 128 #256
__C.TRAIN.NUM_CLASSES = 5
__C.TRAIN.LR = 1e-4
__C.TRAIN.ATTENTION_RESOLUTIONS = "32,16,8"
__C.TRAIN.CHANNEL_MULT = None
__C.TRAIN.DROPOUT = 0.0
__C.TRAIN.DIFFUSION_STEPS = 1000 
__C.TRAIN.SCHEDULE_SAMPLER = "uniform"
__C.TRAIN.NUM_CHANNELS = 128 #256
__C.TRAIN.NUM_HEADS = 1
__C.TRAIN.NUM_HEADS_UPSAMPLE = -1
__C.TRAIN.NUM_HEAD_CHANNELS = 64 
__C.TRAIN.NUM_RES_BLOCKS = 2
__C.TRAIN.RESBLOCK_UPDOWN = True
__C.TRAIN.USE_SCALE_SHIFT_NORM = True
__C.TRAIN.USE_CHECKPOINT = True
__C.TRAIN.CLASS_COND = True
__C.TRAIN.WEIGHT_DECAY = 0.0
__C.TRAIN.LR_ANNEAL_STEPS = 50000 #1000
__C.TRAIN.BATCH_SIZE = 12 #1
__C.TRAIN.MICROBATCH = -1
__C.TRAIN.EMA_RATE = "0.9999"
__C.TRAIN.DROP_RATE = 0.0
__C.TRAIN.LOG_INTERVAL = 1000 #10 # After 10 steps are completed it checks the logger.
__C.TRAIN.SAVE_INTERVAL = 5000 # After the 1000 steps are completed it saves the model.
__C.TRAIN.RESUME_CHECKPOINT = None #"output/sdm_checkpoint/ema_0.9999_050000_2ch_ed_256.pt"  # optional, if you want to resume training from a checkpoint
__C.TRAIN.USE_FP16 = True
__C.TRAIN.DISTRIBUTED_DATA_PARALLEL = True # this was true before, but we need it to be false so it runs in the cuda gpu 
__C.TRAIN.USE_NEW_ATTENTION_ORDER = True
__C.TRAIN.FP16_SCALE_GROWTH = 1e-3
__C.TRAIN.NUM_WORKERS = 10
__C.TRAIN.DETERMINISTIC = False
__C.TRAIN.NO_INSTANCE = True
__C.TRAIN.RANDOM_CROP = False
__C.TRAIN.RANDOM_FLIP = False
__C.TRAIN.IS_TRAIN = False
__C.TRAIN.CHECKPOINT_DIR = "output/b-maps"


__C.TEST = edict()
__C.TEST.S = 1.0
__C.TEST.USE_DDIM = False
__C.TEST.DETERMINISTIC = True
__C.TEST.INFERENCE_ON_TRAIN = True
__C.TEST.BATCH_SIZE = 1
__C.TEST.CLIP_DENOISED = True
__C.TEST.NUM_SAMPLES = 10 #1000
# __C.TEST.RESULTS_DIR = "output/results/DS100_results_10samples"
__C.TEST.RESULTS_DIR = "output/results/b-maps"
