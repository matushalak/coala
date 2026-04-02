import os

from cc.datasets.mnist import mnist
from cc.ml.heads.classifier import ClassifierHead
from cc.ml import MAE_logs, LeJEPA_logs, dataset_lightning_logs_dir
from cc.ml.heads.task_head import create_task_head_trainer

def train_mnist_classifier(batch_size, lr, epochs)-> ClassifierHead:
    train_loader, val_loader, test_loader = mnist(batch_size=batch_size,num_workers=4)
    MNIST_classifier = ClassifierHead.from_pretrained_unet(
        # checkpoint_path=os.path.join(dataset_lightning_logs_dir(MAE_logs, "mnist"), "version_14", "checkpoints", "epoch=20-step=8862.ckpt"),
        # checkpoint_path=os.path.join(dataset_lightning_logs_dir(MAE_logs, "mnist"), "version_28", "checkpoints", "epoch=20-step=8862.ckpt"),
        checkpoint_path=os.path.join(dataset_lightning_logs_dir(LeJEPA_logs, "mnist"), "version_11", "checkpoints", "epoch=50-step=21522.ckpt"),
        num_classes=10,
        latent_dim=64*4*49, # in version 9 was 64*4
        lr=lr,
        num_filters=64,
        freeze_encoder=True,
        upconv_method="upsample+conv",
    )
    trainer = create_task_head_trainer(MNIST_classifier, dataset_name="mnist", max_epochs=epochs)
    trainer.fit(MNIST_classifier, train_loader, val_loader)
    MNIST_classifier = ClassifierHead.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    trainer.test(MNIST_classifier, dataloaders=test_loader, verbose=True)
    return MNIST_classifier

if __name__ == "__main__":
    train_mnist_classifier(batch_size=128, lr=3e-3, epochs=10)
