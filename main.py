from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io as io
import os
import random
import time
import socket
from math import inf

from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.data import DataLoader
from dual import DCT
from data import get_patch_training_set, get_test_set
from torch.autograd import Variable
from psnr import MPSNR
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef
from torch.utils.data.distributed import DistributedSampler
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None



# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=8, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=8, help='training batch size')
parser.add_argument('--patch_size', type=int, default=64, help='training patch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--ChDim', type=int, default=31, help='output channel number')
parser.add_argument('--alpha', type=float, default=0.1, help='alpha')
parser.add_argument('--nEpochs', type=int, default=0, help='(Deprecated) previously used as resume epoch; use --resume_epoch now')
parser.add_argument('--total_epochs', type=int, default=200, help='Total number of training epochs to run (end epoch).')
parser.add_argument('--resume_epoch', type=int, default=0, help='Epoch number to resume from (loads checkpoint epoch_<n>.pth). 0 means start fresh.')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.01')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--save_folder', default='TrainedNet/', help='Directory to keep training outputs.')
parser.add_argument('--outputpath', type=str, default='result/', help='Path to output img')
parser.add_argument('--mode', default=1, type=int, help='Train or Test.')
parser.add_argument('--local_rank', default=0, type=int, help='Local rank for distributed training')
parser.add_argument('--use_distribute', type=int, default=1, help='None')
parser.add_argument('--data_root', type=str, default='/data/CAVEdata12/', help='Root directory containing train/ and test/ subfolders.')
parser.add_argument('--train_prop', type=float, default=1.0, help='Proportion of training images to use (0-1].')
parser.add_argument('--virtual_length', type=int, default=20000, help='Number of random patch samples per epoch (dataset length).')
parser.add_argument('--no_progress', action='store_true', help='Disable tqdm progress bars')
parser.add_argument('--loss_smooth', type=int, default=50, help='Window for smoothed loss display in progress bar')
parser.add_argument('--eval_interval', type=int, default=1, help='Run evaluation every N epochs (set >1 to reduce frequency).')
parser.add_argument('--reconstruct_lr', action='store_true', help='Reconstruct low-res HSI on-the-fly from HR if channel mismatch occurs.')
opt = parser.parse_args()

print(opt)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_random_seed(opt.seed)

use_dist = opt.use_distribute == 1
if use_dist:
    dist.init_process_group(backend="nccl", init_method='env://')
else:
    # Backward compatibility: map deprecated --nEpochs to resume_epoch if provided
    if opt.resume_epoch == 0 and opt.nEpochs > 0:
        opt.resume_epoch = opt.nEpochs

print('===> Loading datasets from', opt.data_root)
train_set = get_patch_training_set(opt.upscale_factor, opt.patch_size, root_dir=opt.data_root, train_prop=opt.train_prop, virtual_length=opt.virtual_length)
if use_dist:
    sampler = DistributedSampler(train_set)
test_set = get_test_set(root_dir=opt.data_root)

if use_dist:
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, sampler = sampler, pin_memory=True)
else:
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle = True, pin_memory=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False, pin_memory=True)

print('===> Building model')
print("===> distribute model")


if use_dist:
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
else:
    local_rank = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DCT(opt.ChDim,opt.upscale_factor).to(device)

if use_dist:
    model = torch.nn.parallel.DistributedDataParallel(model,find_unused_parameters=True,device_ids=[local_rank],output_device=local_rank)
print('# network parameters: {}'.format(sum(param.numel() for param in model.parameters())))
optimizer = optim.Adam(model.parameters(), lr=opt.lr)
scheduler = MultiStepLR(optimizer, milestones=[100, 150, 175, 190, 195], gamma=0.5)

if local_rank == 0:
    print(f"[Dataset] Training base images: {getattr(train_set,'base_count', 'NA')}  | Virtual length: {len(train_set)}  | BatchSize: {opt.batchSize}  | Iter/Epoch: {len(training_data_loader)}")
    print(f"[Dataset] Test images: {len(test_set)}  | Test Iterations: {len(testing_data_loader)}")



if opt.resume_epoch > 0:
    ckpt_path = os.path.join(opt.save_folder.rstrip('/'), f'epoch_{opt.resume_epoch}.pth')
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
    if use_dist:
        dist.barrier()
    map_location = {f'cuda:{0}': f'cuda:{local_rank}'} if device.type == 'cuda' else 'cpu'
    load_dict = torch.load(ckpt_path, map_location=map_location)
    optimizer.load_state_dict(load_dict['adam'])
    model.load_state_dict(load_dict['param'])
    # Optionally override LR with saved LR
    for pg in optimizer.param_groups:
        pg['lr'] = load_dict.get('lr', pg['lr'])
    print(f"Resumed from {ckpt_path} (epoch {load_dict.get('epoch')})")

criterion = nn.L1Loss(reduction='none')


current_time = datetime.now().strftime('%b%d_%H-%M-%S')
CURRENT_DATETIME_HOSTNAME = '/' + current_time + '_' + socket.gethostname()
tb_logger = SummaryWriter(log_dir='./tb_logger/' + 'unfolding2' + CURRENT_DATETIME_HOSTNAME)
current_step = 0

def mkdir(path):
 
    folder = os.path.exists(path)
 
    if not folder:
        os.makedirs(path)
        print("---  new folder...  ---")
        print("---  "+path+"  ---")
    else:
        print("---  There exsits folder "+ path + " !  ---")
        
def normalize_dir(p):
    # Ensure trailing slash and create directory
    if not p.endswith('/'):
        p = p + '/'
    mkdir(p)
    return p

# Normalize output directories (important for Kaggle /kaggle/working storage)
opt.save_folder = normalize_dir(opt.save_folder)
opt.outputpath = normalize_dir(opt.outputpath)

def train(epoch, optimizer, scheduler):
    epoch_loss = 0
    global current_step
    model.train()
    iterator = training_data_loader
    use_bar = (not opt.no_progress) and (tqdm is not None) and (local_rank == 0)
    pbar = tqdm(iterator, total=len(training_data_loader), disable=not use_bar, desc=f"Epoch {epoch} [train]", ncols=120)
    recent_losses = []
    for iteration, batch in enumerate(pbar if use_bar else iterator, 1):
        Y, Z, X = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        optimizer.zero_grad(set_to_none=True)
        Y = Variable(Y).float(); Z = Variable(Z).float(); X = Variable(X).float()
        # Runtime channel/order correction if corrupted (unexpected channel count)
        if Y.shape[1] != opt.ChDim and Z.shape[1] == opt.ChDim:
            Y, Z = Z, Y
        if Y.shape[1] != opt.ChDim:
            if opt.reconstruct_lr and X.shape[1] == opt.ChDim:
                # Rebuild low-res from HR target
                scale = opt.upscale_factor
                Y = torch.nn.functional.interpolate(X, scale_factor=1.0/scale, mode='bicubic', align_corners=False)
            else:
                raise RuntimeError(f"HSI tensor channel mismatch (expected {opt.ChDim} got {Y.shape[1]}). Use --reconstruct_lr or fix dataset.")
        if Z.shape[1] != 3:
            if Z.shape[1] > 3:
                Z = Z[:, :3, :, :]
            else:
                # derive RGB proxy by selecting 3 bands from Y (after upsample) if possible
                if Y.shape[1] >= 3:
                    Z = Y[:, [0, Y.shape[1]//2, -1], :, :]
                else:
                    raise RuntimeError(f"RGB guidance has {Z.shape[1]} channels; cannot adapt.")
        HX = model(Y, Z)
        loss = criterion(HX, X).mean()
        epoch_loss += loss.item()
        if local_rank == 0:
            tb_logger.add_scalar('total_loss', loss.item(), current_step)
        current_step += 1
        loss.backward()
        optimizer.step()
        if use_bar:
            recent_losses.append(loss.item())
            if len(recent_losses) > opt.loss_smooth:
                recent_losses.pop(0)
            smooth = sum(recent_losses)/len(recent_losses)
            lr_cur = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'smooth': f"{smooth:.4f}",
                'lr': f"{lr_cur:.2e}"
            })
        elif iteration % 100 == 0 and local_rank == 0:
            print(f"Epoch {epoch} Iter {iteration}/{len(training_data_loader)} | Loss {loss.item():.4f}")
    avg = epoch_loss / max(1, len(training_data_loader))
    if local_rank == 0:
        print(f"===> Epoch {epoch} Complete: Avg. Loss: {avg:.4f}")
    return avg

def test():
    avg_psnr = 0
    avg_time = 0
    model.eval()
    iterator = testing_data_loader
    use_bar = (not opt.no_progress) and (tqdm is not None) and (local_rank == 0)
    pbar = tqdm(iterator, total=len(testing_data_loader), disable=not use_bar, desc='[eval]', ncols=120)
    with torch.no_grad():
        for batch in (pbar if use_bar else iterator):
            Y, Z, X = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            Y = Variable(Y).float(); Z = Variable(Z).float(); X = Variable(X).float()
            start_time = time.time()
            if Y.shape[1] != opt.ChDim and Z.shape[1] == opt.ChDim:
                Y, Z = Z, Y
            if Y.shape[1] != opt.ChDim:
                if opt.reconstruct_lr and X.shape[1] == opt.ChDim:
                    scale = opt.upscale_factor
                    Y = torch.nn.functional.interpolate(X, scale_factor=1.0/scale, mode='bicubic', align_corners=False)
                else:
                    raise RuntimeError(f"[Eval] HSI channel mismatch {Y.shape[1]} vs {opt.ChDim}")
            if Z.shape[1] != 3:
                if Z.shape[1] > 3:
                    Z = Z[:, :3, :, :]
                else:
                    if Y.shape[1] >= 3:
                        Z = Y[:, [0, Y.shape[1]//2, -1], :, :]
                    else:
                        raise RuntimeError(f"[Eval] RGB guidance has {Z.shape[1]} channels")
            HX = model(Y, Z)
            end_time = time.time()
            X_np = torch.squeeze(X).permute(1, 2, 0).cpu().numpy()
            HX_np = torch.squeeze(HX).permute(1, 2, 0).cpu().numpy()
            psnr = MPSNR(HX_np, X_np)
            avg_psnr += psnr
            avg_time += (end_time - start_time)
            if local_rank == 0:
                im_name = batch[3][0]
                (path, filename) = os.path.split(im_name)
                io.savemat(os.path.join(opt.outputpath, filename), {'HX': HX_np})
            if use_bar:
                pbar.set_postfix({'psnr_avg': f"{(avg_psnr / max(1, pbar.n)):.3f}"})
    psnr_final = avg_psnr / max(1, len(testing_data_loader))
    if local_rank == 0:
        print(f"===> Avg. PSNR: {psnr_final:.4f} dB | Avg. time: {avg_time / max(1,len(testing_data_loader)):.4f} s")
    return psnr_final


def checkpoint(epoch):
    # Save inside folder with clean naming
    model_out_path = os.path.join(opt.save_folder, f'epoch_{epoch}.pth')
    if epoch % 1 == 0 and local_rank == 0:
        save_dict = dict(
            lr=optimizer.state_dict()['param_groups'][0]['lr'],
            param=model.state_dict(),
            adam=optimizer.state_dict(),
            epoch=epoch
        )
        torch.save(save_dict, model_out_path)
        print(f"Checkpoint saved to {model_out_path}")

start_epoch = opt.resume_epoch + 1
end_epoch = opt.total_epochs
if opt.mode == 1:
    for epoch in range(start_epoch, end_epoch + 1):
        avg_loss = train(epoch, optimizer, scheduler)
        checkpoint(epoch)
        run_eval = (epoch % opt.eval_interval == 0) or (epoch == end_epoch)
        if run_eval:
            try:
                avg_psnr = test()
                if local_rank == 0:
                    tb_logger.add_scalar('psnr', avg_psnr, epoch)
            except RuntimeError as e:
                if local_rank == 0:
                    print(f"[Warn] Evaluation skipped due to error: {e}")
        torch.cuda.empty_cache()
        scheduler.step()

else:
    test()
