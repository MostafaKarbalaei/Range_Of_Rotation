import os, sys, math, argparse, socket, importlib
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import provider
import data_loader

# ======================
# Args
# ======================
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--model', default='pointnet_cls_rotation_torch')
parser.add_argument('--log_dir', default='log_rotation')
parser.add_argument('--num_point', type=int, default=1024)
parser.add_argument('--max_epoch', type=int, default=250)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)  # برای SGD
parser.add_argument('--optimizer', default='adam', choices=['adam','momentum'])
parser.add_argument('--decay_step', type=int, default=200000)
parser.add_argument('--decay_rate', type=float, default=0.7)

parser.add_argument('--num_angles', type=int, choices=[6,18,24,32,54,100], default=18)
parser.add_argument('--no_transformation_loss', action='store_true')
parser.add_argument('--no_input_transform', action='store_true')
parser.add_argument('--no_feature_transform', action='store_true')

parser.add_argument('--dataset', type=str, choices=['shapenet','modelnet','modelnet10','shapenet_chair'], default='modelnet')
parser.add_argument('--all_directions', action='store_true')
parser.add_argument('--enable_y_axis', action='store_true')
parser.add_argument('--num_y_rotation_angles', type=int, default=4)
parser.add_argument('--use_angle_loss', action='store_true')

FLAGS = parser.parse_args()
USE_TRANS_LOSS   = (not FLAGS.no_transformation_loss)
USE_INPUT_TRANS  = (not FLAGS.no_input_transform)
USE_FEATURE_TRANS= (not FLAGS.no_feature_transform)
ALL_DIRECTIONS   = FLAGS.all_directions
USE_ANGLE_LOSS   = FLAGS.use_angle_loss

BN_INIT_DECAY        = 0.5
BN_DECAY_DECAY_RATE  = 0.5
BN_DECAY_DECAY_STEP  = float(FLAGS.decay_step)
BN_DECAY_CLIP        = 0.99

# ======================
# Logging dir
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'models'))
log_para_name = f"{FLAGS.model}_{FLAGS.dataset}_rot_angles_{FLAGS.num_angles}_batch_{FLAGS.batch_size}_opt_{FLAGS.optimizer}_lr_{FLAGS.learning_rate}_trans_loss_{USE_TRANS_LOSS}_input_trans_{USE_INPUT_TRANS}_feature_trans_{USE_FEATURE_TRANS}_y_{FLAGS.enable_y_axis}"
LOG_DIR = os.path.join(FLAGS.log_dir, log_para_name)

while os.path.exists(LOG_DIR):
    LOG_DIR = LOG_DIR + "_1"
os.makedirs(LOG_DIR, exist_ok=True)


model_module = importlib.import_module(FLAGS.model)
os.system(f'cp models/{FLAGS.model}.py {LOG_DIR}/')
os.system(f'cp {os.path.basename(__file__)} {LOG_DIR}/')
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train_rot.txt'), 'w')
def log_string(s):
    print(s); LOG_FOUT.write(s+'\n'); LOG_FOUT.flush()

# ======================
# Helper: LR & BN schedules (exact TF parity)
# ======================
def tf_exponential_decay(lr_base, global_sample_idx, decay_step, decay_rate, staircase=True, min_lr=1e-5):
    if staircase:
        p = math.floor(global_sample_idx / decay_step)
    else:
        p = global_sample_idx / decay_step
    lr = lr_base * (decay_rate ** p)
    return max(lr, min_lr)

def get_torch_bn_momentum(global_sample_idx, batch_size):
    # TF:
    # bn_momentum_tf = exponential_decay(BN_INIT_DECAY, global_step*BATCH_SIZE, BN_DECAY_DECAY_STEP, BN_DECAY_DECAY_RATE, staircase=True)
    # bn_decay_tf    = min(BN_DECAY_CLIP, 1 - bn_momentum_tf)
    # PyTorch momentum = 1 - bn_decay_tf
    bn_momentum_tf = tf_exponential_decay(BN_INIT_DECAY, global_sample_idx, BN_DECAY_DECAY_STEP, BN_DECAY_DECAY_RATE, staircase=True, min_lr=BN_INIT_DECAY)  # بدون کلیپ
    bn_decay_tf = min(BN_DECAY_CLIP, 1.0 - bn_momentum_tf)
    torch_bn_momentum = 1.0 - bn_decay_tf

    torch_bn_momentum = max(torch_bn_momentum, 1.0 - BN_DECAY_CLIP)
    return torch_bn_momentum

# ======================
# Dataset
# ======================
NUM_POINT = FLAGS.num_point
X_train, X_test, _, _ = data_loader.get_pointcloud(dataset=FLAGS.dataset, NUM_POINT=NUM_POINT)

ENABLE_Y_AXIS = FLAGS.enable_y_axis
NUM_Y_ROTATION_ANGLES = FLAGS.num_y_rotation_angles
Y_ANGLE_INCREMENT = 2 * np.pi / NUM_Y_ROTATION_ANGLES
NUM_CLASSES_WITHOUT_Y = FLAGS.num_angles
if ENABLE_Y_AXIS:
    NUM_CLASSES = FLAGS.num_angles * NUM_Y_ROTATION_ANGLES
    log_string(f"Enable y-axis labels: NUM_CLASSES = {NUM_CLASSES}")
else:
    NUM_CLASSES = FLAGS.num_angles

BATCH_SIZE = FLAGS.batch_size
if ALL_DIRECTIONS:
    BATCH_SIZE = BATCH_SIZE * NUM_CLASSES

# ======================
# Model
# ======================
device = torch.device(f'cuda:{FLAGS.gpu}' if torch.cuda.is_available() else 'cpu')
model = model_module.PointNetRotation(num_angles=NUM_CLASSES,
                                      use_input_trans=USE_INPUT_TRANS,
                                      use_feature_trans=USE_FEATURE_TRANS,
                                      dropout_keep_prob=0.7).to(device)

criterion = nn.CrossEntropyLoss()

if FLAGS.optimizer == 'momentum':
    optimizer = torch.optim.SGD(model.parameters(), lr=FLAGS.learning_rate, momentum=FLAGS.momentum, weight_decay=0.0)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate, weight_decay=0.0)

writer_tr = SummaryWriter(os.path.join(LOG_DIR, 'train'))
writer_te = SummaryWriter(os.path.join(LOG_DIR, 'test'))

global_step_samples = 0  

def prepare_rotated_batch(current_data, all_dirs=False, enable_y=False):

    data = current_data
    if all_dirs:
        data = np.repeat(data, NUM_CLASSES, axis=0)

    if all_dirs:
        labels = np.tile(np.arange(NUM_CLASSES), int(data.shape[0]//NUM_CLASSES))
    else:
        labels = np.random.randint(NUM_CLASSES, size=data.shape[0])

    if enable_y:
        y_angles = (labels // NUM_CLASSES_WITHOUT_Y) * Y_ANGLE_INCREMENT
        data = provider.rotation_multiprocessing_wrapper(provider.rotate_point_cloud_by_angle_list, data, y_angles)

    labels_general = labels % NUM_CLASSES_WITHOUT_Y
    if NUM_CLASSES_WITHOUT_Y == 6:
        data = provider.rotation_multiprocessing_wrapper(provider.rotate_point_by_label, data, labels_general)
    elif NUM_CLASSES_WITHOUT_Y == 18:
        data = provider.rotation_multiprocessing_wrapper(provider.rotate_point_by_label, data, labels_general)
    elif NUM_CLASSES_WITHOUT_Y == 32:
        data = provider.rotation_multiprocessing_wrapper(provider.rotate_point_by_label_32, data, labels_general)
    elif NUM_CLASSES_WITHOUT_Y == 54:
        data = provider.rotation_multiprocessing_wrapper(provider.rotate_point_by_label_54, data, labels_general)
    elif NUM_CLASSES_WITHOUT_Y:
        data = provider.rotation_multiprocessing_wrapper(provider.rotate_point_by_label_n, data, labels_general, NUM_CLASSES_WITHOUT_Y)
    else:
        raise NotImplementedError

    data, labels, _ = provider.shuffle_data(data, np.squeeze(labels))
    return data.astype(np.float32), labels.astype(np.int64)

def train_epoch(epoch_idx):
    global global_step_samples
    model.train()
    data, labels = prepare_rotated_batch(X_train, all_dirs=ALL_DIRECTIONS, enable_y=ENABLE_Y_AXIS)
    num_batches = data.shape[0] // BATCH_SIZE

    total_correct, total_seen, loss_sum = 0, 0, 0.0

    for b in range(num_batches):
        start, end = b*BATCH_SIZE, (b+1)*BATCH_SIZE
        batch = provider.jitter_point_cloud(data[start:end])  
        target = labels[start:end]

        lr = tf_exponential_decay(FLAGS.learning_rate, global_step_samples, FLAGS.decay_step, FLAGS.decay_rate, staircase=True, min_lr=1e-5)
        for g in optimizer.param_groups: g['lr'] = lr

        torch_bn_m = get_torch_bn_momentum(global_step_samples, FLAGS.batch_size)
        from models.pointnet_cls_rotation_torch import set_bn_momentum as _set_bn_m
        _set_bn_m(model, torch_bn_m)

        batch_t  = torch.from_numpy(batch).float().to(device)     # (B,N,3)
        target_t = torch.from_numpy(target).long().to(device)     # (B,)

        optimizer.zero_grad()
        logits, end_points = model(batch_t, is_training=True)
        loss = criterion(logits, target_t)

        if USE_TRANS_LOSS and USE_FEATURE_TRANS:
            from models.pointnet_cls_rotation_torch import transform_regularizer
            loss = loss + transform_regularizer(end_points, reg_weight=0.001)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            correct = (pred == target_t).sum().item()
            total_correct += correct
            total_seen    += batch_t.shape[0]
            loss_sum      += loss.item()

        global_step_samples += batch_t.shape[0]

    acc = total_correct / max(1, total_seen)
    loss_mean = loss_sum / max(1, num_batches)
    writer_tr.add_scalar('loss', loss_mean, epoch_idx)
    writer_tr.add_scalar('accuracy', acc, epoch_idx)
    log_string(f'Epoch {epoch_idx:03d} | train loss {loss_mean:.4f} | acc {acc:.4f}')

def eval_epoch(epoch_idx):
    model.eval()
    data, labels = prepare_rotated_batch(X_test, all_dirs=False, enable_y=ENABLE_Y_AXIS)
    num_batches = data.shape[0] // FLAGS.batch_size

    total_correct, total_seen, loss_sum = 0, 0, 0.0
    total_seen_class = np.zeros((NUM_CLASSES,), dtype=np.int64)
    total_correct_class = np.zeros((NUM_CLASSES,), dtype=np.int64)

    with torch.no_grad():
        for b in range(num_batches):
            start, end = b*FLAGS.batch_size, (b+1)*FLAGS.batch_size
            batch = data[start:end]
            target = labels[start:end]
            batch_t  = torch.from_numpy(batch).float().to(device)
            target_t = torch.from_numpy(target).long().to(device)

            logits, _ = model(batch_t, is_training=False)
            loss = nn.functional.cross_entropy(logits, target_t)
            pred = logits.argmax(dim=1)

            correct = (pred == target_t).sum().item()
            total_correct += correct
            total_seen    += batch_t.shape[0]
            loss_sum      += loss.item() * batch_t.shape[0]

            # per-class
            for i in range(batch_t.shape[0]):
                l = int(target[i])
                total_seen_class[l] += 1
                total_correct_class[l] += int(pred[i].item() == l)

    acc = total_correct / max(1, total_seen)
    loss_mean = loss_sum / max(1, total_seen)
    avg_class_acc = np.mean((total_correct_class / np.maximum(1, total_seen_class)))
    writer_te.add_scalar('loss', loss_mean, epoch_idx)
    writer_te.add_scalar('accuracy', acc, epoch_idx)
    writer_te.add_scalar('avg_class_acc', avg_class_acc, epoch_idx)
    log_string(f'Epoch {epoch_idx:03d} | eval  loss {loss_mean:.4f} | acc {acc:.4f} | avg_cls_acc {avg_class_acc:.4f}')

def main():
    log_string(f"Logging to {LOG_DIR}")
    for epoch in range(FLAGS.max_epoch):
        train_epoch(epoch)
        eval_epoch(epoch)
        if epoch % 10 == 0:
            ckpt = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(ckpt, os.path.join(LOG_DIR, 'model.pth'))
            log_string(f"Checkpoint saved at epoch {epoch}")

if __name__ == "__main__":
    main()
