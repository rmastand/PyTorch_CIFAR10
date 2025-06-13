import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy
from torchvision.transforms import v2
import torch.nn.functional as F


from cifar10_models.densenet import densenet121, densenet161, densenet169
from cifar10_models.densenet1d import densenet1d
from cifar10_models.googlenet import googlenet
from cifar10_models.inception import inception_v3
from cifar10_models.mobilenetv2 import mobilenet_v2
from cifar10_models.resnet import resnet18, resnet34, resnet50, Projector
from cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from schduler import WarmupCosineLR

all_classifiers = {
    "vgg11_bn": vgg11_bn(),
    "vgg13_bn": vgg13_bn(),
    "vgg16_bn": vgg16_bn(),
    "vgg19_bn": vgg19_bn(),
    "resnet18": resnet18(),
    "resnet34": resnet34(),
    "resnet50": resnet50(),
    "densenet121": densenet121(),
    "densenet161": densenet161(),
    "densenet169": densenet169(),
    "mobilenet_v2": mobilenet_v2(),
    "googlenet": googlenet(),
    "inception_v3": inception_v3(),
     "densenet1d": densenet1d(),
}


class CIFAR10ModuleClassifier(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(vars(hparams))

        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="MULTICLASS", num_classes=10)

        self.model = all_classifiers[self.hparams.classifier]
        self.len_train_dataloader = None # set in train.py

        self.use_embedding_space = self.hparams.use_embedding_space
        if self.use_embedding_space:
            checkpoint = torch.load(self.hparams.path_to_embedding_network)
            full_state_dict = checkpoint["state_dict"]
            extractor_state_dict = {k[6:]:full_state_dict[k] for k in full_state_dict.keys() if "model" in k }
            self.embedding_network = resnet18(is_extractor=True)
            self.embedding_network.load_state_dict(extractor_state_dict)
            self.embedding_network.eval()


    def forward(self, batch):
        images, labels = batch

        if self.use_embedding_space:
            images = self.embedding_network(images)

        predictions = self.model(images)
        loss = self.criterion(predictions, labels)
        accuracy = self.accuracy(predictions, labels)
        return loss, accuracy * 100

    def training_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("loss/train", loss)
        self.log("acc/train", accuracy)
        return loss

    def validation_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("loss/val", loss)
        self.log("acc/val", accuracy)

    def test_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("acc/test", accuracy)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
        total_steps = self.hparams.max_epochs * self.len_train_dataloader
        scheduler = {
            "scheduler": WarmupCosineLR(
                optimizer, warmup_epochs=total_steps * 0.3, max_epochs=total_steps
            ),
            "interval": "step",
            "name": "learning_rate",
        }
        return [optimizer], [scheduler]

"""
"
" DEFINE AUGMENTATIONS
"
"""        
transforms = v2.Compose([
    v2.RandomResizedCrop(size=(32, 32), scale = (0.08, 1)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomApply([v2.ColorJitter( brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
    v2.RandomApply([v2.Grayscale(num_output_channels=3)], p=0.2),
    v2.RandomApply([v2.GaussianBlur(kernel_size=3)], p=0.5),
    v2.RandomSolarize(p=0.1, threshold=0.50980392156),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
 ])


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class CIFAR10ModuleExtractor(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(vars(hparams))

        self.model = resnet18(is_extractor=True) # HARDCODED from all_classifiers[self.hparams.classifier]
        self.projector = Projector(self.hparams, self.model.output_dimension)
        self.len_train_dataloader = None
        self.num_features = int(self.hparams.mlp.split("-")[-1])

    def forward(self, batch):
        images, labels = batch

        # generate augmentations
        x = transforms(images)
        y = transforms(images)

        # evaluate model on augs
        x = self.projector(self.model(x))
        y = self.projector(self.model(y))

        # VICReg loss
        repr_loss = F.mse_loss(x, y)
        
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.hparams.batch_size - 1)
        cov_y = (y.T @ y) / (self.hparams.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        loss = (
            self.hparams.sim_coeff * repr_loss
            + self.hparams.std_coeff * std_loss
            + self.hparams.cov_coeff * cov_loss
        )
        return loss, repr_loss, std_loss, cov_loss

    def training_step(self, batch, batch_nb):
        loss, repr_loss, std_loss, cov_loss = self.forward(batch)
        self.log("loss/train", loss)
        self.log("invariance_loss/train", repr_loss)
        self.log("variance_loss/train", std_loss)
        self.log("covariance_loss/train", cov_loss)
        return loss

    def validation_step(self, batch, batch_nb):
        loss, repr_loss, std_loss, cov_loss = self.forward(batch)
        self.log("loss/val", loss)
        self.log("invariance_loss/val", repr_loss)
        self.log("variance_loss/val", std_loss)
        self.log("covariance_loss/val", cov_loss)

    def test_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            momentum=0.9,
            nesterov=True,
        )
        total_steps = self.hparams.max_epochs * self.len_train_dataloader
        scheduler = {
            "scheduler": WarmupCosineLR(
                optimizer, warmup_epochs= self.len_train_dataloader*10, max_epochs=total_steps
            ),
            "interval": "step",
            "name": "learning_rate",
        }
        return [optimizer], [scheduler]
