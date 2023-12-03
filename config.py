# Super parameters
clamp = 2.0
channels_in = 3
dwt_channels = 48
log10_lr = -4.5
lr = 10 ** log10_lr
weight_decay = 0
init_scale = 0.01

# Model:
device_ids = [0]

# Train:
batch_size = 8
cropsize = 256
betas = (0.5, 0.999)
weight_step = 200
gamma = 0.5

# Val:
cropsize_val = 256
batchsize_val = 2
shuffle_val = False
val_freq = 10


# Dataset
DIV_TRAIN_PATH = '/home/Dataset/DIV2K/DIV2K_train_HR'
DIV_VAL_PATH = '/home/Dataset/DIV2K/DIV2K_valid_HR'
DIV_format_train = 'png'
DIV_format_val = 'png'

pub_TRAIN_PATH = '/home/Dataset/pub/train'
pub_VAL_PATH = '/home/Dataset/pub/val'
pub_format_train = 'png'
pub_format_val = 'png'


# Display and logging:
loss_display_cutoff = 2.0
loss_names = ['train loss', 'val loss', 'lr', 'attack method', 'Current EP time']
silent = False
live_visualization = False
progress_bar = True


# Saving checkpoints:

MODEL_PATH = 'model/'
checkpoint_on_error = True
SAVE_freq = 50

IMAGE_PATH = 'image/'
IMAGE_PATH_host= IMAGE_PATH + 'host/'
IMAGE_PATH_secret = IMAGE_PATH + 'secret/'
IMAGE_PATH_container = IMAGE_PATH + 'container/'
IMAGE_PATH_extracted = IMAGE_PATH + 'extracted/'

