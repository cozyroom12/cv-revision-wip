import torch.nn as nn
import os
import torch
import torchvision
import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
from torchvision.models.efficientnet import efficientnet_b0
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from torch.utils.data import DataLoader
from pracCustomDataset import SportsDataset
import argparse
import pandas as pd


class SportsClassifier:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

    def train(self, train_loader, val_loader, epochs, optimizer, criterion, start_epoch=0):
        """self, train_loader, val_loader, epochs, optimizer, criterion, start_epoch=0"""
        best_val_acc = 0.0
        print("Training. . .")

        for epoch in range(start_epoch, epochs):
            train_loss = 0.0
            val_loss = 0.0
            train_acc = 0.0
            val_acc = 0.0

            self.model.train()
            train_loader_iter = tqdm(train_loader, desc=(f'Epoch: {epoch+1}/{epochs}'), leave=False)

            for index, (data, target) in enumerate(train_loader_iter):
                data, target = data.float().to(self.device), target.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                train_acc += (preds == target).sum().item()

                train_loader_iter.set_postfix({'Loss' : loss.item()})

            train_loss /= len(train_loader)
            train_acc /= len(train_loader.dataset)

            # eval()
            self.model.eval()
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.float().to(self.device), target.to(self.device)
                    outputs = self.model(data)
                    preds = outputs.argmax(dim=1, keepdim=True)
                    val_acc += preds.eq(target.view_as(preds)).sum().item()
                    val_loss += criterion(outputs, target).item()

            val_loss /= len(val_loader)
            val_acc /= len(val_loader.dataset)

            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            print(f"Epoch [{epoch + 1}/{epochs}], Train loss: {train_loss:.4f}, "
                  f"Val loss: {val_loss:.4f}, Train ACC: {train_acc:.4f}, Val ACC: {val_acc:.4f}")

            if val_acc > best_val_acc:
                torch.save(self.model.state_dict(), './sports_efficientnet_b0_best.pt')

            # save the model state & optimizer state per each epoch
            torch.save({
                'epoch':epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses' : self.train_losses,
                'train_accs' : self.train_accs,
                'val_losses' : self.val_losses,
                'val_accs' : self.val_accs,

            }, './weight/sports_efficientnet_b0_checkpoint.pt')

        torch.save(self.model.state_dict(), './sports_efficientnet_b0_last.pt')

        self.save_results_to_csv()
        self.plot_loss()
        self.plot_accuracy()

    def save_results_to_csv(self):
        df = pd.DataFrame({
            'Train Loss' : self.train_losses,
            'Train Accuracy': self.train_accs,
            'Validation Loss' : self.val_losses,
            'Validation Accuracy' : self.val_accs
        })
        df.to_csv('train_val_results_sports.csv', index=False)

    def plot_loss(self):
        plt.figure()
        plt.plot(self.train_losses, label='Train loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('sports_loss_plot.png')

    def plot_accuracy(self):
        plt.figure()
        plt.plot(self.train_accs, label='Train Accuracy')
        plt.plot(self.val_accs, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig('sports_acc_plot.png')

    def run(self, args):
        self.model = efficientnet_b0(pretrained=True)
        self.model.classifier[1] = nn.Linear(1280, out_features=100)
        self.model.to(self.device)

        train_transforms = A.Compose([
            A.Resize(224, 224),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
            A.RandomShadow(),
            A.RandomRain(),
            A.RandomFog(),
            A.RGBShift(r_shift_limit=5, g_shift_limit=5, b_shift_limit=5, p=0.5),
            A.HorizontalFlip(),
            A.RandomBrightness(),
            A.RandomRotate90(),
            ToTensorV2()
        ])

        val_transforms = A.Compose([
            A.Resize(224, 224),
            ToTensorV2()
        ])

        train_dataset = SportsDataset(args.train_dir, transform=train_transforms)
        val_dataset = SportsDataset(args.val_dir, transform=val_transforms)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        epochs = args.epochs
        criterion = CrossEntropyLoss().to(self.device)
        optimizer = AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        start_epoch = 0

        if args.resume_training :
            checkpoint = torch.load(args.checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.train_losses = (checkpoint['train_losses'])
            self.train_accs = (checkpoint['train_accs'])
            self.val_losses = (checkpoint['val_losses'])
            self.val_accs = (checkpoint['val_accs'])
            start_epoch = checkpoint['epoch']  # 이거 빼먹으면 안된다~

        self.train(train_loader, val_loader, epochs, optimizer, criterion, start_epoch)


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='./sports_data/train/',
                        help='directory path to training dataset')
    parser.add_argument('--val_dir', type=str, default='./sports_data/valid',
                        help='directory path to valid dataset')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=124,
                        help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2,
                        help='weight decay')
    parser.add_argument('--resume_training', action='store_true',
                        help='resume training from the last checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default='./weight/sports_efficientnet_b0_checkpoint.pt',
                        help='path to checkpoint file')
    parser.add_argument('--checkpoint_folder_path', type=str,
                        default='./weight/')


    args = parser.parse_args()
    # weight_folder_path = args.checkpoint_folder_path
    # print(weight_folder_path)  # default 값이 그대로 나옴
    # os.makedirs(weight_folder_path, exist_ok=True)


    classifier = SportsClassifier()
    classifier.run(args)