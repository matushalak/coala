import os

from cc.datasets.mnist import mnist
from cc.ml.heads.classifier import ClassifierHead
from cc.ml import MAE_logs
from cc.ml.heads.task_head import create_task_head_trainer

def train_mnist_classifier(batch_size, lr)-> ClassifierHead:
    train_loader, val_loader, test_loader = mnist(batch_size=batch_size,num_workers=4)
    MNIST_classifier = ClassifierHead.from_pretrained_unet(
        checkpoint_path=os.path.join(MAE_logs, "version_7/checkpoints/epoch=20-step=8862.ckpt"),
        num_classes=10,
        latent_dim=32*4,
        lr=lr,
        freeze_encoder=True,
    )
    trainer = create_task_head_trainer(MNIST_classifier, max_epochs=10)
    trainer.fit(MNIST_classifier, train_loader, val_loader)
    MNIST_classifier = ClassifierHead.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    trainer.test(MNIST_classifier, dataloaders=test_loader, verbose=True)
    return MNIST_classifier

if __name__ == "__main__":
    train_mnist_classifier(batch_size=128, lr=3e-3)