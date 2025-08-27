import pytorch_lightning as pl
import torch
import random
import numpy as np
import os
from mask2former import ( Mask2FormerFinetuner, 
                        SegmentationDataModule, 
                        DATASET_DIR, 
                        BATCH_SIZE, 
                        NUM_WORKERS, 
                        ID2LABEL, 
                        LEARNING_RATE, 
                        LOGGER,  
                        DEVICES,
                        CHECKPOINT_CALLBACK, 
                        EPOCHS )

MASK_SCALE = 30
SEED = 42
random.seed(SEED)                   
np.random.seed(SEED)               
torch.manual_seed(SEED)            
torch.cuda.manual_seed(SEED)       
torch.cuda.manual_seed_all(SEED)   
torch.backends.cudnn.deterministic = True   
torch.backends.cudnn.benchmark = False      
os.environ['PYTHONHASHSEED'] = str(SEED) 

if __name__=="__main__":
    out = os.path.join("trained_models", "multiclass", "mask2former")
    data_module = SegmentationDataModule(dataset_dir=DATASET_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    model=Mask2FormerFinetuner(ID2LABEL, LEARNING_RATE)
    print("all goood")
    trainer = pl.Trainer(
        logger=LOGGER,
        accelerator='cuda',
        devices=1,
        strategy="auto",
        callbacks=[CHECKPOINT_CALLBACK],
        max_epochs=EPOCHS
    )
    print("Training starts!!")
    trainer.fit(model,data_module)
    print("saving model!")
    trainer.save_checkpoint(os.path.join(out,"mask2former.pth"))

