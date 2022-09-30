import glob
import json
import os
import pickle
import warnings
from collections import defaultdict
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
warnings.filterwarnings(action="ignore")

training = False
image_size = 1024
dataset_dir = "data/train_dataset"


class EyeDataset(Dataset):
    """Dataset class organizing the loading and receiving of images and corresponding markups.
    """
    
    def __init__(self, data_folder: str, transform = None):
        self.class_ids = {"vessel": 1}
        
        self.data_folder = data_folder
        self.transform = transform
        self._image_files = glob.glob(f"{data_folder}/*.png")
    
    @staticmethod
    def read_image(path: str) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image / 255, dtype=np.float32)
        
        return image
    
    @staticmethod
    def parse_polygon(coordinates, image_size):
        mask = np.zeros(image_size, dtype=np.float32)
        
        if len(coordinates) == 1:
            points = [np.int32(coordinates)]
            cv2.fillPoly(mask, points, 1)
        else:
            points = [np.int32([coordinates[0]])]
            cv2.fillPoly(mask, points, 1)
            
            for polygon in coordinates[1:]:
                points = [np.int32([polygon])]
                cv2.fillPoly(mask, points, 0)
        
        return mask
    
    @staticmethod
    def parse_mask(shape: dict, image_size: tuple) -> np.ndarray:
        """Method for parsing figures from geojson file.
        """
        
        mask = np.zeros(image_size, dtype=np.float32)
        coordinates = shape["coordinates"]
        if shape["type"] == "MultiPolygon":
            for polygon in coordinates:
                mask += EyeDataset.parse_polygon(polygon, image_size)
        else:
            mask += EyeDataset.parse_polygon(coordinates, image_size)
        
        return mask
    
    def read_layout(self, path: str, image_size: tuple) -> np.ndarray:
        """Method for reading geojson markup and converting to numpy mask.
        """
        
        with open(path, "r", encoding="cp1251") as f:
            json_contents = json.load(f)
        
        num_channels = 1 + max(self.class_ids.values())
        mask_channels = [np.zeros(image_size, dtype=np.float32) for _ in range(num_channels)]
        mask = np.zeros(image_size, dtype=np.float32)
        
        if type(json_contents) == dict and json_contents["type"] == "FeatureCollection":
            features = json_contents["features"]
        elif type(json_contents) == list:
            features = json_contents
        else:
            features = [json_contents]
        
        for shape in features:
            channel_id = self.class_ids["vessel"]
            mask = self.parse_mask(shape["geometry"], image_size)
            mask_channels[channel_id] = np.maximum(mask_channels[channel_id], mask)
        
        mask_channels[0] = 1 - np.max(mask_channels[1:], axis=0)
        
        return np.stack(mask_channels, axis=-1)
    
    def __getitem__(self, idx: int) -> dict:
        image_path = self._image_files[idx]
        json_path = image_path.replace("png", "geojson")
        
        image = self.read_image(image_path)
        
        mask = self.read_layout(json_path, image.shape[:2])
        
        sample = {"image": image, "mask": mask}
        
        if self.transform is not None:
            sample = self.transform(**sample)
        
        return sample
    
    def __len__(self):
        return len(self._image_files)
    
    def make_report(self):
        reports = []
        if (not self.data_folder):
            reports.append("Dataset path not specified.")
        if (len(self._image_files) == 0):
            reports.append("Images for recognition not found.")
        else:
            reports.append(f"Found {len(self._image_files)} images.")
        cnt_images_without_masks = sum([1 - len(glob.glob(filepath.replace("png", "geojson"))) for filepath in self._image_files])
        if cnt_images_without_masks > 0:
            reports.append(f"Found {cnt_images_without_masks} images without markup.")
        else:
            reports.append(f"All images have a markup file.")
        
        return reports


class DatasetPart(Dataset):
    """Wrapper over the dataset class for splitting it into parts.
    """
    
    def __init__(self, dataset: Dataset, indices: np.ndarray, transform: A.Compose = None):
        self.dataset = dataset
        self.indices = indices
        
        self.transform = transform
    
    def __getitem__(self, idx: int) -> dict:
        sample = self.dataset[self.indices[idx]]
        
        if self.transform is not None:
            sample = self.transform(**sample)
        
        return sample
    
    def __len__(self) -> int:
        return len(self.indices)


if training:
    train_list = [
        A.RandomCrop(width=1550, height=1550),
        A.LongestMaxSize(image_size, interpolation=cv2.INTER_CUBIC),
        A.Rotate(limit=15),
        A.PadIfNeeded(image_size, image_size),
        ToTensorV2(transpose_mask=True)
    ]

    eval_list = [
        A.RandomCrop(width=1550, height=1550),
        A.LongestMaxSize(image_size, interpolation=cv2.INTER_CUBIC),
        A.Rotate(limit=15),
        A.PadIfNeeded(image_size, image_size),
        ToTensorV2(transpose_mask=True)
    ]

    transforms = {"train": A.Compose(train_list), "test": A.Compose(eval_list)}

    dataset = EyeDataset(dataset_dir)
    for report in dataset.make_report():
        print(report)

    train_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.25)
    print(f"Splitting into train/test: {len(train_indices)}/{len(test_indices)}.")

    train_dataset = DatasetPart(dataset, train_indices, transform=transforms["train"])
    valid_dataset = DatasetPart(dataset, test_indices, transform=transforms["test"])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 4, shuffle=True, drop_last=True
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, 4, shuffle=True, drop_last=True
    )


class UnetTrainer:
    """The class that implements model training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        device: str,
        metric_functions: List[Tuple[str, Callable]] = [],
        epoch_number: int = 0,
        lr_scheduler: Optional[Any] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        
        self.device = device
        
        self.metric_functions = metric_functions
        
        self.epoch_number = epoch_number
    
    @torch.no_grad()
    def evaluate_batch(
        self, val_iterator: Iterator, eval_on_n_batches: int
    ) -> Optional[Dict[str, float]]:
        predictions = []
        targets = []
        
        losses = []
        
        for real_batch_number in range(eval_on_n_batches):
            try:
                batch = next(val_iterator)
                
                xs = batch["image"].to(self.device)
                ys_true = batch["mask"].to(self.device)
            except StopIteration:
                if real_batch_number == 0:
                    return None
                else:
                    break
            ys_pred = self.model.eval()(xs)
            loss = self.criterion(ys_pred, ys_true)
            
            losses.append(loss.item())
            
            predictions.append(ys_pred.cpu())
            targets.append(ys_true.cpu())
        
        predictions = torch.cat(predictions, dim=0)
        targets = torch.cat(targets, dim=0)
        
        metrics = {"loss": np.mean(losses)}
        
        for metric_name, metric_fn in self.metric_functions:
            metrics[metric_name] = metric_fn(predictions, targets).item()
        
        return metrics
    
    @torch.no_grad()
    def evaluate(self, val_loader, eval_on_n_batches: int = 1) -> Dict[str, float]:
        """Calculating metrics for an epoch.
        """
        
        metrics_sum = defaultdict(float)
        num_batches = 0
        
        val_iterator = iter(val_loader)
        
        while True:
            batch_metrics = self.evaluate_batch(val_iterator, eval_on_n_batches)
            
            if batch_metrics is None:
                break
            
            for metric_name in batch_metrics:
                metrics_sum[metric_name] += batch_metrics[metric_name]
            
            num_batches += 1
        
        metrics = {}
        
        for metric_name in metrics_sum:
            metrics[metric_name] = metrics_sum[metric_name] / num_batches
        
        return metrics
    
    def fit_batch(
        self, train_iterator: Iterator, update_every_n_batches: int
    ) -> Optional[Dict[str, float]]:
        """Model training on one batch.
        """
        
        self.optimizer.zero_grad()
        
        predictions = []
        targets = []
        
        losses = []
        
        for real_batch_number in range(update_every_n_batches):
            try:
                batch = next(train_iterator)
                
                xs = batch["image"].to(self.device)
                ys_true = batch["mask"].to(self.device)
            except StopIteration:
                if real_batch_number == 0:
                    return None
                else:
                    break
            
            ys_pred = self.model.train()(xs)
            loss = self.criterion(ys_pred, ys_true)
            
            (loss / update_every_n_batches).backward()
            
            losses.append(loss.item())
            
            predictions.append(ys_pred.cpu())
            targets.append(ys_true.cpu())
        
        self.optimizer.step()
        
        predictions = torch.cat(predictions, dim=0)
        targets = torch.cat(targets, dim=0)
        
        metrics = {"loss": np.mean(losses)}
        
        for metric_name, metric_fn in self.metric_functions:
            metrics[metric_name] = metric_fn(predictions, targets).item()
        
        return metrics
    
    def fit_epoch(
        self, train_loader, update_every_n_batches: int = 1
    ) -> Dict[str, float]:
        """One epoch of model training.
        """
        
        metrics_sum = defaultdict(float)
        num_batches = 0
        
        train_iterator = iter(train_loader)
        
        while True:
            batch_metrics = self.fit_batch(train_iterator, update_every_n_batches)
            
            if batch_metrics is None:
                break
            
            for metric_name in batch_metrics:
                metrics_sum[metric_name] += batch_metrics[metric_name]
            
            num_batches += 1
        
        metrics = {}
        
        for metric_name in metrics_sum:
            metrics[metric_name] = metrics_sum[metric_name] / num_batches
        
        return metrics
    
    def fit(
        self,
        train_loader,
        num_epochs: int,
        val_loader = None,
        update_every_n_batches: int = 1,
    ) -> Dict[str, np.ndarray]:
        """Method that trains model and computes metrics for each epoch.
        """
        
        summary = defaultdict(list)
        
        def save_metrics(metrics: Dict[str, float], postfix: str = "") -> None:
            nonlocal summary, self
            
            for metric in metrics:
                metric_name, metric_value = f"{metric}{postfix}", metrics[metric]
                
                summary[metric_name].append(metric_value)
        
        for _ in tqdm(
            range(num_epochs - self.epoch_number),
            initial=self.epoch_number,
            total=num_epochs,
        ):
            self.epoch_number += 1
            
            train_metrics = self.fit_epoch(train_loader, update_every_n_batches)
            
            with torch.no_grad():
                save_metrics(train_metrics, postfix="_train")
                
                if val_loader is not None:
                    test_metrics = self.evaluate(val_loader)
                    save_metrics(test_metrics, postfix="_test")
            
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            if not self.epoch_number % 10:
                torch.save({
                    "epoch": self.epoch_number,
                    "model_state_dict": self.model.state_dict(),
                    "loss_test": test_metrics["loss"],
                    "accuracy_test": test_metrics["accuracy"],
                    "recall_test": test_metrics["recall"],
                    "exp_dice_test": test_metrics["dice"],
                    }, os.path.join("models", "unetplus_{}.pt".format(self.epoch_number)))
            
            torch.save({
                "epoch": self.epoch_number,
                "model_state_dict": self.model.state_dict(),
                "loss_test": test_metrics["loss"],
                "accuracy_test": test_metrics["accuracy"],
                "recall_test": test_metrics["recall"],
                "exp_dice_test": test_metrics["dice"],
                }, os.path.join("models", "unetplus_last.pt"))
            
            with open("log.txt", "a") as the_file:
                the_file.write("Train L {:.5f} | A {:.5f} | R {:.5f} | F1 {:.5f}".format(
                train_metrics["loss"],
                train_metrics["accuracy"],
                train_metrics["recall"],
                train_metrics["dice"])+"\n")
            
            with open('log.txt', 'a') as the_file:
                the_file.write("Test L {:.5f} | A {:.5f} | R {:.5f} | F1 {:.5f}".format(
                test_metrics["loss"],
                test_metrics["accuracy"],
                test_metrics["recall"],
                test_metrics["dice"])+"\n")
        
        summary = {metric: np.array(summary[metric]) for metric in summary}
        
        return summary

# F1 metric.
class SoftDice:
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
    
    def __call__(
        self, predictions: List[Dict[str, torch.Tensor]],
        targets: List[Dict[str, torch.Tensor]]
        ) -> torch.Tensor:
        numerator = torch.sum(2 * predictions * targets)
        denominator = torch.sum(predictions + targets)
        return numerator / (denominator + self.epsilon)

# Recall metric.
class Recall:
    def __init__(self, epsilon=1e-81):
        self.epsilon = epsilon
    
    def __call__(
        self, predictions: List[Dict[str, torch.Tensor]],
        targets: List[Dict[str, torch.Tensor]]
        ) -> torch.Tensor:
        numerator = torch.sum(predictions * targets)
        denominator = torch.sum(targets)
        
        return numerator / (denominator + self.epsilon)

# Precision metric.
class Accuracy:
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
    
    def __call__(self, predictions: list, targets: list) -> torch.Tensor:
        numerator = torch.sum(predictions * targets)
        denominator = torch.sum(predictions)
        
        return numerator / (denominator + self.epsilon)


def make_metrics():
    soft_dice = SoftDice()
    recall = Recall()
    accuracy = Accuracy()
    
    def exp_dice(pred, target):
        return soft_dice(torch.exp(pred[:, 1:]), target[:, 1:])
    
    def exp_accuracy(pred, target):
        return accuracy(torch.exp(pred[:, 1:]), target[:, 1:])
    
    def exp_recall(pred, target):
        return recall(torch.exp(pred[:, 1:]), target[:, 1:])
    
    return [("dice", exp_dice),
            ("accuracy", exp_accuracy),
            ("recall", exp_recall),
            ]

def make_criterion():
    soft_dice = SoftDice()
    
    def exp_dice(pred, target):
        return 1 - soft_dice(torch.exp(pred[:, 1:]), target[:, 1:])
    
    return exp_dice

if training:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    model = smp.UnetPlusPlus("resnet101", activation="logsoftmax", classes=2)
    model= nn.DataParallel(model)
    model.to(device)
    
    criterion = make_criterion()
    optimizer = torch.optim.Adam(model.parameters(), 0.001)

    trainer = UnetTrainer(
        model, optimizer, criterion, device, metric_functions=make_metrics()
    )

    summary = trainer.fit(train_loader, 151, val_loader=valid_loader)

    with open("data/summary.pkl", "wb") as f:
        pickle.dump(summary, f)

    print("Done!")