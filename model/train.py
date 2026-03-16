from arch.RNN_FastText import RNN_FastText
import torch

import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

from data.dataset_paraphrase import DatasetParaphrase
from data.data_server import DataServer

def save_checkpoint(state, epoch, is_best, model_name):
    model_name = f"checkpoints/{model_name}_epoch_{epoch}.pth"
    torch.save(state, model_name)
    if is_best:
        torch.save(state, model_name.replace('.pth', '_best.pth'))

def train(data_loader, val_data_loader, model_l, optimizer_l, criterion_l, device_l, epochs=10):

    loss_epochs_t = []
    val_losses_t = []

    for epoch in range(epochs):
        model_l.train()
        loss_epoch = 0
        tqdm_loader = tqdm(data_loader)
        for batch_index, (data, targets) in enumerate(tqdm_loader):
            sentence1 = data['sentence1'].to(device_l)
            sentence2 = data['sentence2'].to(device_l)
            labels = targets['label'].float().to(device_l)

            optimizer_l.zero_grad()

            outputs = model_l(sentence1, sentence2).squeeze()

            loss = criterion_l(outputs, labels)
            loss_epoch += loss.item()
            loss.backward()
            optimizer_l.step()

            tqdm_loader.set_description(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

        avg_loss_epoch = loss_epoch / len(data_loader)
        val_loss = validate(val_data_loader, model_l, criterion_l, device_l)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss_epoch:.4f}")

        is_best = val_loss < min(val_losses_t, default=float('inf'))

        val_losses_t.append(val_loss)
        loss_epochs_t.append(avg_loss_epoch)
        save_checkpoint(state=model_l.state_dict(), epoch=epoch, is_best=is_best, model_name='rnn_fasttext')

    return loss_epochs_t, val_losses_t

def test():
    pass

def validate(val_loader_l, model_l, criterion_l, device_l):

    model_l.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_index, (data, targets) in enumerate(tqdm(val_loader_l)):
            sentence1 = data['sentence1'].to(device_l)
            sentence2 = data['sentence2'].to(device_l)
            labels = targets['label'].float().to(device_l)

            outputs = model_l(sentence1, sentence2).squeeze()
            loss = criterion_l(outputs, labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader_l)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    return avg_val_loss

def predict():
    pass


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = DatasetParaphrase('dataset.csv')
    train_loader, val_loader, test_loader = dataset.get_data_loaders(batch_size=16, num_workers=6, pin_memory=True)

    model = RNN_FastText()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.HuberLoss()

    loss_epochs, val_losses = train(train_loader, val_loader, model, optimizer, criterion, device, epochs=50)