import pytorch_lightning as pl
import torch
import random
import numpy as np
import os
from oneformer import (
    OneFormerFinetuner,
    SegmentationDataModule,
    DATASET_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    ID2LABEL,
    LEARNING_RATE,
    LOGGER,
    DEVICES,
    CHECKPOINT_CALLBACK,
    EPOCHS,
    label_map)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(SEED)

if __name__ == "__main__":
    out = os.path.join("trained_models", "multiclass", "oneformer")
    os.makedirs(out, exist_ok=True)

    data_module = SegmentationDataModule(
        dataset_dir=DATASET_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        label_map=label_map, 
    )
   
    model = OneFormerFinetuner(
        id2label=ID2LABEL,
        lr=LEARNING_RATE,
        label_map=label_map,
    )
    trainer = pl.Trainer(
        logger=LOGGER,
        accelerator="cuda",
        devices=1 ,
        strategy="auto",
        callbacks=[CHECKPOINT_CALLBACK],
        max_epochs=EPOCHS,
        num_sanity_val_steps=0,
        precision="16-mixed",
    )

    trainer.fit(model, data_module)
    trainer.save_checkpoint(os.path.join(out, "oneformer.pth"))