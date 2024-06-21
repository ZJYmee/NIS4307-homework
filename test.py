from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from model import ViolenceClassifier
from dataset import CustomDataModule

gpu_id = [0]
batch_size = 128
log_name = "shufflenet"
 

def main():
    data_module = CustomDataModule(batch_size=batch_size)
    ckpt_root = "/your/data/root/"
    ckpt_path = ckpt_root + "shufflenet/version_0/checkpoints/shufflenet-epoch=xx-val_loss=xx.ckpt"
    model = ViolenceClassifier.load_from_checkpoint(ckpt_path)
    trainer = Trainer(accelerator='gpu', devices=gpu_id)
    trainer.test(model, data_module) 

if __name__ == '__main__':
    main()
