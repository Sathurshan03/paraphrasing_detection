from .arch.RNN_FastText import RNN_FastText
import torch
import torch.distributed as dist
import os

import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .data.dataset_paraphrase import DatasetParaphrase
from .data.data_server import DataServer
from .params import DEBUG, USE_DDP

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def save_checkpoint(state, epoch, is_best, model_name):
    model_name = f"checkpoints/{model_name}_epoch_{epoch}.pth"
    torch.save(state, model_name)
    if is_best:
        torch.save(state, model_name.replace('.pth', '_best.pth'))

def cleanup_ddp():
    dist.destroy_process_group()

def train(data_loader, val_data_loader, model_l, optimizer_l, criterion_l, device_l, epochs=10, rank=0):

    loss_epochs_t = []
    val_losses_t = []

    for epoch in range(epochs):
        data_loader.sampler.set_epoch(epoch)
        model_l.train()
        loss_epoch = 0
        tqdm_loader = tqdm(data_loader, disable=(rank != 0))
        for batch_index, batch in enumerate(tqdm_loader):
            sentence1 = batch['sentence1'].to(device_l)        # (batch, seq, 300)
            sentence2 = batch['sentence2'].to(device_l)
            mask_1 = batch['sentence1_mask'].to(device_l)      # (batch, seq) bool
            mask_2 = batch['sentence2_mask'].to(device_l)
            labels = batch['label'].float().to(device_l)       # (batch,)

            optimizer_l.zero_grad()

            outputs = model_l(sentence1, sentence2, mask_1=mask_1, mask_2=mask_2).squeeze(-1)

            loss = criterion_l(outputs, labels)
            loss_epoch += loss.item()
            loss.backward()
            optimizer_l.step()

            tqdm_loader.set_description(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

        avg_loss_epoch = loss_epoch / len(data_loader)
        val_loss = validate(val_data_loader, model_l, criterion_l, device_l, rank)

        if rank == 0:
            print(f"Epoch {epoch+1}/{epochs} — train: {avg_loss_epoch:.4f}  val: {val_loss:.4f}")
            is_best = val_loss < min(val_losses_t, default=float('inf'))
            # Unwrap DDP to save just the model weights
            raw_state = model_l.module.state_dict()
            save_checkpoint(raw_state, epoch, is_best, 'rnn_fasttext')

        val_losses_t.append(val_loss)
        loss_epochs_t.append(avg_loss_epoch)

    return loss_epochs_t, val_losses_t

def test():
    pass

def validate(val_loader_l, model_l, criterion_l, device_l, rank=0):

    model_l.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(val_loader_l, disable=(rank != 0))):
            sentence1 = batch['sentence1'].to(device_l)
            sentence2 = batch['sentence2'].to(device_l)
            mask_1 = batch['sentence1_mask'].to(device_l)
            mask_2 = batch['sentence2_mask'].to(device_l)
            labels = batch['label'].float().to(device_l)

            outputs = model_l(sentence1, sentence2, mask_1=mask_1, mask_2=mask_2).squeeze(-1)
            loss = criterion_l(outputs, labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader_l)

    if rank == 0:
        print(f"Validation Loss: {avg_val_loss:.4f}")

    return avg_val_loss

def predict():
    pass


def ddp_worker(rank, world_size, train_ds, val_ds, epochs=50):
    setup_ddp(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    # Each rank builds its own DataLoader with a DistributedSampler
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader  = DataLoader(train_ds, batch_size=8, sampler=train_sampler,
                               num_workers=4, pin_memory=True, collate_fn=DataServer.collate_fn)
    val_loader    = DataLoader(val_ds,   batch_size=8, shuffle=False,
                               num_workers=4, pin_memory=True, collate_fn=DataServer.collate_fn)
    model     = RNN_FastText().to(device)
    model     = DDP(model, device_ids=[rank])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.HuberLoss()
    loss_epochs, val_losses = train(train_loader, val_loader, model, optimizer, criterion, device, epochs, rank)
    # Only rank 0 saves plots / loss files
    if rank == 0:
        with open('loss_epochs.txt', 'w') as f:
            f.writelines(f"{v}\n" for v in loss_epochs)
        with open('val_losses.txt', 'w') as f:
            f.writelines(f"{v}\n" for v in val_losses)
        plt.plot(loss_epochs, label='Training Loss')
        plt.plot(val_losses,  label='Validation Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
        plt.savefig('loss_plot.png')
    cleanup_ddp()




if __name__ == "__main__":

    if not USE_DDP:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        dataset = DatasetParaphrase('dataset.csv')
        train_loader, val_loader, test_loader = dataset.get_data_loaders(batch_size=8, num_workers=6, pin_memory=True)

        model = RNN_FastText()
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.HuberLoss()

        loss_epochs, val_losses = train(train_loader, val_loader, model, optimizer, criterion, device, epochs=50)

        # save these to file

        with open('loss_epochs.txt', 'w') as f:
            for epoch in loss_epochs:
                f.write(f"{epoch}\n")

        with open('val_losses.txt', 'w') as f:
            for loss in val_losses:
                f.write(f"{loss}\n")

        # Plot and save to file

        plt.plot(loss_epochs, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('loss_plot.png')

    else:
        world_size = torch.cuda.device_count()   # 6 on your node
        print(f"Launching DDP across {world_size} GPUs")
        # Load dataset ONCE in the main process, then pass Dataset objects to workers
        dataset = DatasetParaphrase('dataset.csv')
        train_ds, val_ds, _ = dataset.get_datasets()
        torch.multiprocessing.spawn(
            ddp_worker,
            args=(world_size, train_ds, val_ds),
            nprocs=world_size,
            join=True,
        )