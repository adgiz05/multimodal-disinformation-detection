import torch.nn.functional as F
import torch.optim as optim
import torch
import os
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score 
import pytorch_lightning as pl
from transformers import AutoModel
import torch.nn as nn

class Model(pl.LightningModule):
    def __init__(self, config=None):
        super().__init__()
        self.save_hyperparameters(config)
        self.build_architecture()

        self.model_path = os.path.join('./models/', self.hparams.experiment)
        os.makedirs(self.model_path, exist_ok=True)

        self.epoch_preds, self.epoch_labels, self.epoch_losses = [], [], []
        self.train_accuracy, self.train_f1, self.train_mean_loss = .0, .0, .0
        self.valid_preds, self.valid_labels, self.valid_losses = [], [], []
        self.valid_accuracy, self.valid_f1, self.valid_mean_loss = .0, .0, .0
        self.best_metric = .0

        self.start_epoch = .0

        self.history = {
            'train_losses' : [],
            'valid_losses' : [],
            'train_metrics' : {
                'accuracy' : [],
                'f1' : []
            },
            'valid_metrics' : {
                'accuracy' : [],
                'f1' : []
            },
            'best_epoch' : -1
        }
    
    def training_step(self, batch, batch_idx):
        input, label = batch
        output = self(**input)
        # print(output, label)
        loss = F.cross_entropy(output, label)
        
        self.epoch_preds.append(torch.argmax(output, dim=1).detach().cpu().numpy())
        self.epoch_labels.append(label.detach().cpu().numpy())
        self.epoch_losses.append(loss.detach().cpu().numpy())

        return loss

    def validation_step(self, batch, batch_idx):
        input, label = batch
        output = self(**input)
        loss = F.cross_entropy(output, label)

        self.valid_preds.append(torch.argmax(output, dim=1).detach().cpu().numpy())
        self.valid_labels.append(label.detach().cpu().numpy())
        self.valid_losses.append(loss.detach().cpu().numpy())
        return loss

    def test_step(self, batch, batch_idx):
        input, label = batch
        output = self(**input)

        self.test_preds.append(torch.argmax(output, dim=1).detach().cpu().numpy())
        self.test_labels.append(label.detach().cpu().numpy())
    
    def on_train_epoch_start(self):
        self.start_epoch = time.time()
    
    def on_train_epoch_end(self):
        preds = np.concatenate(self.epoch_preds).tolist()
        labels = np.concatenate(self.epoch_labels).tolist()
        losses = self.epoch_losses

        self.train_accuracy = accuracy_score(preds, labels)
        self.train_f1 = f1_score(preds, labels, average='macro')
        self.train_mean_loss = np.mean(losses)

        self.history['train_losses'].append(float(self.train_mean_loss))
        self.history['train_metrics']['accuracy'].append(float(self.train_accuracy))
        self.history['train_metrics']['f1'].append(float(self.train_f1))

        self.epoch_preds.clear()
        self.epoch_labels.clear()
        self.epoch_losses.clear()

    def on_validation_epoch_end(self):
        preds = np.concatenate(self.valid_preds).tolist()
        labels = np.concatenate(self.valid_labels).tolist()
        losses = self.valid_losses

        self.valid_accuracy = accuracy_score(preds, labels)
        self.valid_f1 = f1_score(preds, labels, average='macro')
        self.valid_mean_loss = np.mean(losses)

        self.history['valid_losses'].append(float(self.valid_mean_loss))
        self.history['valid_metrics']['accuracy'].append(float(self.valid_accuracy))
        self.history['valid_metrics']['f1'].append(float(self.valid_f1))

        self.valid_preds.clear()
        self.valid_labels.clear()
        self.valid_losses.clear()

        self.log('val_loss', self.valid_mean_loss)
        self.log('val_acc', self.valid_accuracy)

        if self.valid_accuracy > self.best_metric:
            self.history['best_epoch'] = int(self.current_epoch)
            self.best_metric = self.valid_accuracy

        epoch_lasted = int(time.time() - self.start_epoch)

        log = f'Epoch {self.current_epoch}\t LASTED: {epoch_lasted} s - '
        log += f'loss: {self.train_mean_loss:.4f} - accuracy: {self.train_accuracy:.4f} - f1: {self.train_f1:.4f} - '
        log += f'valid_loss: {self.valid_mean_loss:.4f} - valid_accuracy: {self.valid_accuracy:.4f} - valid_f1: {self.valid_f1:.4f}\n'

        with open(os.path.join(self.model_path, 'logs.txt'), 'a') as f: 
            f.write(log)

    def on_test_epoch_start(self):
        self.test_preds, self.test_labels = [], []
        self.test_accuracy, self.test_f1 = .0, .0

    def on_test_epoch_end(self):
        preds = np.concatenate(self.test_preds).tolist()
        labels = np.concatenate(self.test_labels).tolist()

        self.test_accuracy = accuracy_score(preds, labels)
        self.test_f1 = f1_score(preds, labels, average='macro')

        with open(os.path.join(self.model_path, 'results.txt'), 'w') as f:
            f.write(f'Test accuracy: {self.test_accuracy}\n')
            f.write(f'Test f1: {self.test_f1}')

    #History plots for loss and each metric
    def plot_history(self):
        num_epochs = len(self.history['train_losses'])
        epochs = [int(i) for i in list(range(num_epochs))]
        num_plots = len(self.history['train_metrics']) + 1 #Plot loss and each metric
        best_epoch = self.history["best_epoch"]

        plt.figure(figsize=(5*num_plots, 5))
        plt.title(f'EXPERIMENT: {self.hparams.experiment} (BE: {best_epoch})')
        plt.subplot(1, num_plots, 1)
        plt.title('LOSS')
        plt.plot(epochs, self.history['train_losses'], label='train')
        plt.plot(epochs, self.history['valid_losses'], label='valid')
        plt.axvline(x=best_epoch, color='red', linestyle='dotted')
        plt.xlabel('Epochs')
        plt.legend()

        metric_names = self.history['train_metrics'].keys()
        train_metrics = self.history['train_metrics'].values()
        valid_metrics = self.history['valid_metrics'].values()

        for i, (m_name, train_values, valid_values) in enumerate(zip(metric_names, train_metrics, valid_metrics), 2):
            plt.subplot(1, num_plots, i)
            plt.title(m_name.upper())
            plt.plot(epochs, train_values, label='train')
            plt.plot(epochs, valid_values, label='valid')
            plt.axvline(x=best_epoch, color='red', linestyle='dotted')
            plt.xlabel('Epochs')
            plt.legend()

        plt.savefig(os.path.join(self.model_path, 'training_plot.png'))

    def on_fit_end(self):
        with open(os.path.join(self.model_path, 'history.json'), 'w') as f:
            json.dump(self.history, f)

        with open(os.path.join(self.model_path, 'config.json'), 'w') as f:
            json.dump(self.hparams, f)

        self.plot_history()

class CLIPOnlyText(Model):
    def __init__(self, config=None):
        super().__init__(config)

    def build_architecture(self):
        #Encoder
        self.clip = AutoModel.from_pretrained('openai/clip-vit-base-patch32')
        if self.hparams.freeze_encoder:
            for param in self.clip.parameters():
                param.requires_grad = False
        
        #Classification head
        self.fc1 = nn.Linear(512, 256)
        self.activation1 = nn.GELU()
        self.output = nn.Linear(256, 2)

    def forward(self, input_ids=None, attention_mask=None):
        x = self.clip.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        x = self.activation1(self.fc1(x))
        x = torch.softmax(self.output(x), dim=-1)
        return x
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, amsgrad=True, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10)

        return {
            'optimizer' : optimizer,
            'lr_scheduler' : {
                'scheduler' : scheduler,
                'monitor' : 'val_loss'
            }
        }
    
class CLIPOnlyImage(Model):
    def __init__(self, config=None):
        super().__init__(config)

    def build_architecture(self):
        #Encoder
        self.clip = AutoModel.from_pretrained('openai/clip-vit-base-patch32')
        if self.hparams.freeze_encoder:
            for param in self.clip.parameters():
                param.requires_grad = False
        
        #Classification head
        self.fc1 = nn.Linear(512, 256)
        self.activation1 = nn.GELU()
        self.output = nn.Linear(256, 2)

    def forward(self, pixel_values=None):
        x = self.clip.get_image_features(pixel_values)
        x = self.activation1(self.fc1(x))
        x = torch.softmax(self.output(x), dim=-1)
        return x
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, amsgrad=True, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10)

        return {
            'optimizer' : optimizer,
            'lr_scheduler' : {
                'scheduler' : scheduler,
                'monitor' : 'val_loss'
            }
        }
    
class MetadataClassifier(Model):
    def __init__(self, config=None):
        super().__init__(config)

    def build_architecture(self):

        self.metadata_classifier = nn.Sequential(
            nn.Linear(3, 512),
            nn.GELU(),
            # nn.LayerNorm(512),
            nn.Linear(512, 256),
            # nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(256, 2)
        )

    def forward(self, upvote_ratio=None, score=None, num_comments=None):
        x = torch.cat((upvote_ratio, score, num_comments), dim=1)
        x = self.metadata_classifier(x)
        return x
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, amsgrad=True, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10)

        return {
            'optimizer' : optimizer,
            'lr_scheduler' : {
                'scheduler' : scheduler,
                'monitor' : 'val_loss'
            }
        }
    
class CLIPComments(Model):
    def __init__(self, config=None):
        super().__init__(config)

    def build_architecture(self):
        #### BODY ####
        self.clip = AutoModel.from_pretrained('openai/clip-vit-base-patch32')
        self.comments_encoder = AutoModel.from_pretrained('microsoft/deberta-base')
        self.freeze_encoders()
        
        self.after_fusion_dim = (512 + 512) if self.hparams.fusion == 'concat' else 512

        self.clip_module_hl1_dim = 512

        self.comments_encoder_output_dim = 768
        self.comments_encoder_hl1_dim = 512

        self.multimodal_projector_hl1_dim = 512
        self.multimodal_projector_hl2_dim = 170

        self.text_projector_hl1_dim = 512
        self.text_projector_hl2_dim = 170

        self.image_projector_hl1_dim = 170

        self.head_input_dim = self.multimodal_projector_hl2_dim + self.text_projector_hl2_dim + self.image_projector_hl1_dim
        self.head_hl1_dim = 256
        self.output_dim = 3

        self.clip_module = nn.Sequential(
            nn.Linear(self.after_fusion_dim, self.clip_module_hl1_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.comments_encoder_module = nn.Sequential(
            nn.Linear(self.comments_encoder_output_dim, self.comments_encoder_hl1_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.multimodal_projector = nn.Sequential(
            nn.Linear(self.after_fusion_dim, self.multimodal_projector_hl1_dim),
            nn.LayerNorm(self.multimodal_projector_hl1_dim),
            nn.GELU(),
            # nn.Dropout(0.2),
            nn.Linear(self.multimodal_projector_hl1_dim, self.multimodal_projector_hl2_dim),
            nn.LayerNorm(self.multimodal_projector_hl2_dim),
            nn.GELU()
        )

        self.text_projector = nn.Sequential(
            nn.Linear(self.after_fusion_dim, self.text_projector_hl1_dim),
            nn.LayerNorm(self.text_projector_hl1_dim),
            nn.GELU(),
            nn.Linear(self.text_projector_hl1_dim, self.text_projector_hl2_dim),
            nn.LayerNorm(self.text_projector_hl2_dim),
            nn.GELU()
        )

        self.image_projector = nn.Sequential(
            nn.Linear(self.clip_module_hl1_dim, self.image_projector_hl1_dim),
            nn.LayerNorm(self.image_projector_hl1_dim),
            nn.GELU()
        )

        self.head = nn.Sequential(
            nn.Linear(self.head_input_dim, self.head_hl1_dim),
            nn.LayerNorm(self.head_hl1_dim),
            nn.GELU(),
            nn.Linear(self.head_hl1_dim, self.output_dim),
            # nn.Softmax(dim=-1)
        )

    def freeze_encoders(self):
        #### CLIP ####
        if self.hparams.freeze_clip:
            #Freeze the whole CLIP encoder
            for param in self.clip.parameters():
                param.requires_grad = False

        if self.hparams.clip_not_frozen_layers != 0:
            for name, parameter in self.clip.named_parameters():
                #Last norm layers for vision and text encoders
                if name.startswith("text_model.encoder.final_layer_norm") or name.startswith("vision_model.post_layernorm"):
                    parameter.requires_grad = True
                #Last k CLIP encoder layers (self-attention layers)
                elif name.startswith("text_model.encoder.layers") or name.startswith("vision_model.encoder.layers"):
                    layer_num = int(name.split(".")[3].split("-")[0])
                    if layer_num >= 12 - self.hparams.clip_not_frozen_layers:  # Freeze last 4 layers
                        parameter.requires_grad = True
                #Last projection layers for vision and text encoders
                elif name == "visual_projection.weight" or name == "text_projection.weight":
                    parameter.requires_grad = True

        #### COMMENTS ENCODER ####
        if self.hparams.freeze_comments_encoder:
            #Freeze the whole CLIP encoder
            for param in self.comments_encoder.parameters():
                param.requires_grad = False
        if self.hparams.comments_encoder_not_frozen_layers != 0:
            for name, parameter in self.comments_encoder.named_parameters():
                if name.startswith("encoder.layer"):
                    layer_num = int(name.split(".")[2])
                    if layer_num >= 12 - self.hparams.comments_encoder_not_frozen_layers:  # Freeze last 4 layers
                        parameter.requires_grad = True
                elif name.startswith("encoder.rel_embeddings"):
                    parameter.requires_grad = True

    def fusion(self, features1, features2):
        x = None
        if self.hparams.fusion == 'concat':
            x = torch.cat((features1, features2), dim=1)
        elif self.hparams.fusion == 'abs_substraction':
            x = torch.abs(features1 - features2)
        elif self.hparams.fusion == 'mul':
            x = torch.mul(features1, features2)
        elif self.hparams.fusion == 'sum':
            x = features1 + features2
        elif self.hparams.fusion == 'max':
            x, _ = torch.max(torch.stack([features1, features2]), dim=0)
        return x

    def __mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids=None, attention_mask=None, comments_input_ids=None, comments_attention_mask=None, pixel_values=None):
        # print(input_ids.shape, attention_mask.shape, comments_input_ids.shape, comments_attention_mask.shape, pixel_values.shape)
        image_features = self.clip.get_image_features(pixel_values) #512
        caption_features = self.clip.get_text_features(input_ids, attention_mask) #512
        comments_features = self.__mean_pooling(self.comments_encoder(comments_input_ids, comments_attention_mask), comments_attention_mask) #768

        #MULTIMODAL BRANCH
        multimodal_x = self.fusion(image_features, caption_features) #1024 concat else 512
        multimodal_x = self.clip_module(multimodal_x) #1024/512 -> 512

        comments_x  = self.comments_encoder_module(comments_features) #768 -> 512

        multimodal_x = self.fusion(multimodal_x, comments_x) #1024
        multimodal_x = self.multimodal_projector(multimodal_x) #1024 -> 512 -> 170

        #TEXT BRANCH
        text_x = self.fusion(caption_features, comments_x) #1024
        text_x = self.text_projector(text_x) #1024 -> 512 -> 170

        #IMAGE BRANCH
        image_x = self.image_projector(image_features) #512 -> 170

        #### HEAD ####
        x = torch.cat((multimodal_x, image_x, text_x), dim=1) #170 + 170 + 170 = 510
        x = self.head(x) #510 -> 256 -> [0, 1]
        return x
    
    def configure_optimizers(self):
        optimizer = optim.AdamW([
            {'params' : self.clip.parameters(), 'lr' : self.hparams.body_lr},
            {'params' : self.comments_encoder.parameters(), 'lr' : self.hparams.body_lr},
            {'params' : self.clip_module.parameters()},
            {'params' : self.comments_encoder_module.parameters()},
            {'params' : self.multimodal_projector.parameters()},
            {'params' : self.text_projector.parameters()},
            {'params' : self.image_projector.parameters()},
            {'params' : self.head.parameters()}
        ], lr=self.hparams.head_lr, amsgrad=True, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10)

        return {
            'optimizer' : optimizer,
            'lr_scheduler' : {
                'scheduler' : scheduler,
                'monitor' : 'val_loss'
            }
        }

class CLIPTextAndComments(Model):
    def __init__(self, config=None):
        super().__init__(config)

    def build_architecture(self):
        #### BODY ####
        self.clip = AutoModel.from_pretrained('openai/clip-vit-base-patch32')
        self.comments_encoder = AutoModel.from_pretrained('microsoft/deberta-v3-base')
        self.freeze_encoders()
        
        self.after_fusion_dim = (512 + 512) if self.hparams.fusion == 'concat' else 512

        self.comments_encoder_module = nn.Sequential(
            nn.Linear(768, 512),
            nn.GELU()
        )

        self.text_projector = nn.Sequential(
            nn.Linear(self.after_fusion_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 2)
        )

    def freeze_encoders(self):
        #### CLIP ####
        if self.hparams.freeze_clip:
            #Freeze the whole CLIP encoder
            for param in self.clip.parameters():
                param.requires_grad = False

        if self.hparams.clip_not_frozen_layers != 0:
            for name, parameter in self.clip.named_parameters():
                #Last norm layers for vision and text encoders
                if name.startswith("text_model.encoder.final_layer_norm") or name.startswith("vision_model.post_layernorm"):
                    parameter.requires_grad = True
                #Last k CLIP encoder layers (self-attention layers)
                elif name.startswith("text_model.encoder.layers") or name.startswith("vision_model.encoder.layers"):
                    layer_num = int(name.split(".")[3].split("-")[0])
                    if layer_num >= 12 - self.hparams.clip_not_frozen_layers:  # Freeze last 4 layers
                        parameter.requires_grad = True
                #Last projection layers for vision and text encoders
                elif name == "visual_projection.weight" or name == "text_projection.weight":
                    parameter.requires_grad = True

        #### COMMENTS ENCODER ####
        if self.hparams.freeze_comments_encoder:
            #Freeze the whole CLIP encoder
            for param in self.comments_encoder.parameters():
                param.requires_grad = False
        if self.hparams.comments_encoder_not_frozen_layers != 0:
            for name, parameter in self.comments_encoder.named_parameters():
                if name.startswith("encoder.layer"):
                    layer_num = int(name.split(".")[2])
                    if layer_num >= 12 - self.hparams.comments_encoder_not_frozen_layers:  # Freeze last 4 layers
                        parameter.requires_grad = True
                elif name.startswith("encoder.rel_embeddings"):
                    parameter.requires_grad = True

    def fusion(self, features1, features2):
        x = None
        if self.hparams.fusion == 'concat':
            x = torch.cat((features1, features2), dim=1)
        elif self.hparams.fusion == 'abs_substraction':
            x = torch.abs(features1 - features2)
        elif self.hparams.fusion == 'mul':
            x = torch.mul(features1, features2)
        elif self.hparams.fusion == 'sum':
            x = features1 + features2
        elif self.hparams.fusion == 'max':
            x, _ = torch.max(torch.stack([features1, features2]), dim=0)
        return x

    def __mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids=None, attention_mask=None, comments_input_ids=None, comments_attention_mask=None):
        caption_features = self.clip.get_text_features(input_ids, attention_mask) #512
        comments_features = self.__mean_pooling(self.comments_encoder(comments_input_ids, comments_attention_mask), comments_attention_mask) #768

        comments_x  = self.comments_encoder_module(comments_features) #768 -> 512

        #TEXT BRANCH
        text_x = self.fusion(caption_features, comments_x) #1024
        text_x = self.text_projector(text_x) #1024 -> 512 -> 170
        return text_x
    
    def configure_optimizers(self):
        optimizer = optim.AdamW([
            {'params' : self.clip.parameters(), 'lr' : self.hparams.body_lr},
            {'params' : self.comments_encoder.parameters(), 'lr' : self.hparams.body_lr},
            {'params' : self.comments_encoder_module.parameters()},
            {'params' : self.text_projector.parameters()},
        ], lr=self.hparams.head_lr, amsgrad=True, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10)

        return {
            'optimizer' : optimizer,
            'lr_scheduler' : {
                'scheduler' : scheduler,
                'monitor' : 'val_loss'
            }
        }    

class DebertaComments(Model):
    def __init__(self, config=None):
        super().__init__(config)
        self.freeze_encoders()

    def build_architecture(self):
        #### BODY ####
        self.comments_encoder = AutoModel.from_pretrained('microsoft/deberta-base')

        self.head = nn.Sequential(
            nn.Linear(768, 512),
            # nn.LayerNorm(self.head_hl1_dim),
            nn.GELU(),
            nn.Linear(512, 256),
            # nn.LayerNorm(self.head_hl1_dim),
            nn.GELU(),
            nn.Linear(256, 2),
            # nn.Softmax(dim=-1)
        )

    def freeze_encoders(self):
        #### COMMENTS ENCODER ####
        if self.hparams.freeze_comments_encoder:
            #Freeze the whole CLIP encoder
            for param in self.comments_encoder.parameters():
                param.requires_grad = False
        if self.hparams.comments_encoder_not_frozen_layers != 0:
            for name, parameter in self.comments_encoder.named_parameters():
                if name.startswith("encoder.layer"):
                    layer_num = int(name.split(".")[2])
                    if layer_num >= 12 - self.hparams.comments_encoder_not_frozen_layers:  # Freeze last 4 layers
                        parameter.requires_grad = True
                elif name.startswith("encoder.rel_embeddings"):
                    parameter.requires_grad = True

    def __mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, comments_input_ids=None, comments_attention_mask=None):
        comments_features = self.__mean_pooling(self.comments_encoder(comments_input_ids, comments_attention_mask), comments_attention_mask) #768
        x = self.head(comments_features) #510 -> 256 -> [0, 1]
        return x
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, amsgrad=True, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10)

        return {
            'optimizer' : optimizer,
            'lr_scheduler' : {
                'scheduler' : scheduler,
                'monitor' : 'val_loss'
            }
        }
    
class CLIP(Model):
    def __init__(self, config=None):
        super().__init__(config)

    def build_architecture(self):
        #### BODY ####
        self.body = AutoModel.from_pretrained('openai/clip-vit-base-patch32')
        self.freeze_encoder()

        #### HEAD ####
        self.hl1_dim = (512 + 512) if self.hparams.fusion == 'concat' else 512
        self.hl2_dim = 512
        self.hl3_dim = 256
        self.output_dim = 2

        self.head = nn.Sequential(
            nn.Linear(self.hl1_dim, self.hl2_dim),
            nn.LayerNorm(self.hl2_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.hl2_dim, self.hl3_dim),
            nn.GELU(),
            nn.Linear(self.hl3_dim, self.output_dim),
            nn.Softmax(dim=-1)
        )

    def freeze_encoder(self):
        if self.hparams.freeze_encoder:
            #Freeze the whole CLIP encoder
            for param in self.body.parameters():
                param.requires_grad = False

        if self.hparams.not_frozen_layers is None or self.hparams.not_frozen_layers == 0:
            for name, parameter in self.body.named_parameters():
                #Last norm layers for vision and text encoders
                if name.startswith("text_model.encoder.final_layer_norm") or name.startswith("vision_model.post_layernorm"):
                    parameter.requires_grad = True
                #Last k CLIP encoder layers (self-attention layers)
                elif name.startswith("text_model.encoder.layers") or name.startswith("vision_model.encoder.layers"):
                    layer_num = int(name.split(".")[3].split("-")[0])
                    if layer_num >= 12 - self.hparams.not_frozen_layers:  # Freeze last 4 layers
                        parameter.requires_grad = True
                #Last projection layers for vision and text encoders
                elif name == "visual_projection.weight" or name == "text_projection.weight":
                    parameter.requires_grad = True

    def fusion(self, image_features, caption_features):
        x = None
        if self.hparams.fusion == 'concat':
            x = torch.cat((image_features, caption_features), dim=1)
        elif self.hparams.fusion == 'abs_substraction':
            x = torch.abs(image_features - caption_features)
        elif self.hparams.fusion == 'mul':
            x = torch.mul(image_features, caption_features)
        elif self.hparams.fusion == 'sum':
            x = image_features + caption_features
        elif self.hparams.fusion == 'max':
            x, _ = torch.max(torch.stack([image_features, caption_features]), dim=0)
        return x

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None):
        image_features = self.body.get_image_features(pixel_values)
        caption_features = self.body.get_text_features(input_ids, attention_mask)

        x = self.fusion(image_features, caption_features)
        x = self.head(x)
        return x
    
    def configure_optimizers(self):
        optimizer = optim.AdamW([
            {'params' : self.body.parameters(), 'lr' : self.hparams.body_lr},
            {'params' : self.head.parameters(), 'lr' : self.hparams.head_lr}
        ], amsgrad=True, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)

        return {
            'optimizer' : optimizer,
            'lr_scheduler' : {
                'scheduler' : scheduler,
                'monitor' : 'val_loss'
            }
        }
    
class CLIPCommentsAndMetadata(Model):
    def __init__(self, config=None):
        super().__init__(config)

    def build_architecture(self):
        #### BODY ####
        self.clip = AutoModel.from_pretrained('openai/clip-vit-base-patch32')
        self.comments_encoder = AutoModel.from_pretrained('microsoft/deberta-base')
        self.freeze_encoders()
        
        self.after_fusion_dim = (512 + 512) if self.hparams.fusion == 'concat' else 512

        self.clip_module = nn.Sequential(
            nn.LayerNorm(self.after_fusion_dim),
            nn.Linear(self.after_fusion_dim, 512),
            nn.Dropout(0.1),
            nn.GELU()
        )

        self.comments_encoder_module = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 512),
            nn.Dropout(0.1),
            nn.GELU()
        )

        self.multimodal_projector = nn.Sequential(
            nn.LayerNorm(self.after_fusion_dim),
            nn.Linear(self.after_fusion_dim, 512),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 128),
            nn.Dropout(0.1),
            nn.GELU()
        )

        self.text_projector = nn.Sequential(
            nn.LayerNorm(self.after_fusion_dim),
            nn.Linear(self.after_fusion_dim, 512),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 128),
            nn.Dropout(0.1),
            nn.GELU()
        )

        self.image_projector = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 128),
            nn.Dropout(0.1),
            nn.GELU()
        )

        self.metadata_projector = nn.Sequential(
            nn.LayerNorm(1),
            nn.Linear(1, 64),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 128),
            nn.Dropout(0.1),
            nn.GELU()
        )

        # self.modalities_weights = nn.Parameter(torch.ones(4))

        self.head = nn.Sequential(
            nn.LayerNorm(128 * 4),
            nn.Linear(128 * 4, 256),
            nn.GELU(),
            # nn.LayerNorm(256),
            # nn.Linear(256, self.head_hl2_dim),
            # nn.GELU(),
            nn.Linear(256, 2),
            # nn.Softmax(dim=-1)
        )

    def freeze_encoders(self):
        #### CLIP ####
        if self.hparams.freeze_clip:
            #Freeze the whole CLIP encoder
            for param in self.clip.parameters():
                param.requires_grad = False

        if self.hparams.clip_not_frozen_layers != 0:
            for name, parameter in self.clip.named_parameters():
                #Last norm layers for vision and text encoders
                if name.startswith("text_model.encoder.final_layer_norm") or name.startswith("vision_model.post_layernorm"):
                    parameter.requires_grad = True
                #Last k CLIP encoder layers (self-attention layers)
                elif name.startswith("text_model.encoder.layers") or name.startswith("vision_model.encoder.layers"):
                    layer_num = int(name.split(".")[3].split("-")[0])
                    if layer_num >= 12 - self.hparams.clip_not_frozen_layers:  # Freeze last 4 layers
                        parameter.requires_grad = True
                #Last projection layers for vision and text encoders
                elif name == "visual_projection.weight" or name == "text_projection.weight":
                    parameter.requires_grad = True

        #### COMMENTS ENCODER ####
        if self.hparams.freeze_comments_encoder:
            #Freeze the whole CLIP encoder
            for param in self.comments_encoder.parameters():
                param.requires_grad = False
        if self.hparams.comments_encoder_not_frozen_layers != 0:
            for name, parameter in self.comments_encoder.named_parameters():
                if name.startswith("encoder.layer"):
                    layer_num = int(name.split(".")[2])
                    if layer_num >= 12 - self.hparams.comments_encoder_not_frozen_layers:  # Freeze last 4 layers
                        parameter.requires_grad = True
                elif name.startswith("encoder.rel_embeddings"):
                    parameter.requires_grad = True

    def fusion(self, features1, features2):
        x = None
        if self.hparams.fusion == 'concat':
            x = torch.cat((features1, features2), dim=1)
        elif self.hparams.fusion == 'abs_substraction':
            x = torch.abs(features1 - features2)
        elif self.hparams.fusion == 'mul':
            x = torch.mul(features1, features2)
        elif self.hparams.fusion == 'sum':
            x = features1 + features2
        elif self.hparams.fusion == 'max':
            x, _ = torch.max(torch.stack([features1, features2]), dim=0)
        return x

    def __mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids=None, attention_mask=None, comments_input_ids=None, comments_attention_mask=None, pixel_values=None, score=None, upvote_ratio=None, num_comments=None):
        # print(input_ids.shape, attention_mask.shape, comments_input_ids.shape, comments_attention_mask.shape, pixel_values.shape)
        image_features = self.clip.get_image_features(pixel_values) #512
        caption_features = self.clip.get_text_features(input_ids, attention_mask) #512
        comments_features = self.__mean_pooling(self.comments_encoder(comments_input_ids, comments_attention_mask), comments_attention_mask) #768
        metadata_features = torch.cat((score, upvote_ratio, num_comments), dim=1)
        # metadata_features = num_comments

        #METADATA BRANCH
        metadata_x = self.metadata_projector(metadata_features)

        #MULTIMODAL BRANCH
        multimodal_x = self.fusion(image_features, caption_features) #1024 concat else 512
        multimodal_x = self.clip_module(multimodal_x) #1024/512 -> 512

        comments_x  = self.comments_encoder_module(comments_features) #768 -> 512

        # multimodal_x = torch.cat((multimodal_x, comments_x, metadata_x), dim=1) #1024 + 128
        multimodal_x = torch.cat((multimodal_x, comments_x), dim=1) #1024 + 128
        multimodal_x = self.multimodal_projector(multimodal_x) #1024 -> 512 -> 170

        #TEXT BRANCH
        text_x = self.fusion(caption_features, comments_x) #1024
        text_x = self.text_projector(text_x) #1024 -> 512 -> 170

        #IMAGE BRANCH
        image_x = self.image_projector(image_features) #512 -> 170

        #### HEAD ####
        # multimodal_x = multimodal_x * self.modalities_weights[0]
        # image_x = image_x * self.modalities_weights[1]
        # text_x = text_x * self.modalities_weights[2]
        # metadata_x = metadata_x * self.modalities_weights[3]

        x = torch.cat((multimodal_x, image_x, text_x, multimodal_x), dim=1) #128 + 128 + 128 + 128 = 512
        x = self.head(x) #512 -> 256 -> [0, 1]
        return x
    
    def configure_optimizers(self):
        optimizer = optim.AdamW([
            {'params' : self.clip.parameters(), 'lr' : self.hparams.body_lr},
            {'params' : self.comments_encoder.parameters(), 'lr' : self.hparams.body_lr},
            {'params' : self.clip_module.parameters()},
            {'params' : self.comments_encoder_module.parameters()},
            {'params' : self.multimodal_projector.parameters()},
            {'params' : self.text_projector.parameters()},
            {'params' : self.image_projector.parameters()},
            {'params' : self.metadata_projector.parameters()},
            {'params' : self.head.parameters()},
            # {'params' : self.modalities_weights},
        ], lr=self.hparams.head_lr, amsgrad=True, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10)

        return {
            'optimizer' : optimizer,
            'lr_scheduler' : {
                'scheduler' : scheduler,
                'monitor' : 'val_loss'
            }
        }

class EfficientNet(Model):
    def __init__(self, config=None):
        super().__init__(config)

    def build_architecture(self):
        #Encoder
        self.efficient_net = AutoModel.from_pretrained('google/efficientnet-b0')
        self.output = nn.Linear(1280 * 7 * 7, 2)

    def forward(self, pixel_values=None):
        # print(pixel_values)
        batch_size = pixel_values.shape[0]
        x = self.efficient_net(pixel_values).last_hidden_state
        x = x.view(batch_size, -1)
        x = self.output(x)
        return x
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, amsgrad=True, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10)

        return {
            'optimizer' : optimizer,
            'lr_scheduler' : {
                'scheduler' : scheduler,
                'monitor' : 'val_loss'
            }
        }