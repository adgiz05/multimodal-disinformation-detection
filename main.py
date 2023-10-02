# from torch_model import Model
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import architecture_builder, data_builder
import os

if __name__ == '__main__':
    pl.seed_everything(123413, workers=True)
    epochs = 30

    config = {
        'experiment' : 'TextAndComments',
        'batch_size' : 16,
        'lr' : 1e-5,
        'experiment' : '[EXTENDED]MultimodalDeberta-extended-reg-3classes',
        'batch_size' : 16,
        'freeze_clip' : True,
        'clip_not_frozen_layers' : 2,
        'freeze_comments_encoder' : True,
        'comments_encoder_not_frozen_layers' : 2,
        'body_lr' : 1e-5,
        'head_lr' : 1e-4,
        'fusion' : 'concat'
    }

    train_dataset = data_builder.MultimodalWithComments('train')
    valid_dataset = data_builder.MultimodalWithComments('valid')
    test_dataset = data_builder.MultimodalWithComments('test')

    dm = data_builder.DataModule(train_dataset, valid_dataset, test_dataset, batch_size=config['batch_size'])

    model = architecture_builder.CLIPComments(config)

    ckpt_callback = ModelCheckpoint(
        dirpath=os.path.join('./models/', config['experiment']), 
        filename=config['experiment'], 
        save_top_k=1,
        monitor='val_acc',
        mode='max'
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=10
    )

    trainer = pl.Trainer(max_epochs=epochs, callbacks=[ckpt_callback], precision=16, accelerator='gpu')
    trainer.fit(model, dm)
    trainer.test(model, dm.test_dataloader(), ckpt_path='best')