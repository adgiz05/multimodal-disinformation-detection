from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
from transformers import AutoTokenizer, AutoProcessor
import torch
from torch.nn.utils.rnn import pad_sequence
import cv2
import pytorch_lightning as pl

class DataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, test_dataset, batch_size=16):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.collate_fn = self.train_dataset.collate_fn

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn, num_workers=4, pin_memory=True, prefetch_factor=8)
        # return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=4, pin_memory=True, prefetch_factor=8)
        # return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, num_workers=4, pin_memory=True, prefetch_factor=8)
        # return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)

class Multimodal(Dataset):
    def __init__(self, subset, tokenizer='openai/clip-vit-base-patch32', processor='openai/clip-vit-base-patch32', data_aug=False):
        self.images_path = f'./dataset/images/{subset}/'
        self.data_path = f'./dataset/{subset}.csv'
        self.data_aug = data_aug

        # #CLIP tokenizer and processor by default
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.processor = AutoProcessor.from_pretrained(processor)

        #RandAug if data_aug is enabled
        if self.data_aug: self.transform = transforms.RandAugment(num_ops=6, magnitude=9) 
        
        self.input_ids, self.attention_mask, self.label, self.data = self.extract_features(tokenizer)

    def extract_features(self, tokenizer):
        images = [img.split('.')[0] for img in os.listdir(self.images_path)]
        data = pd.read_csv(self.data_path)
        data = data[data.id.isin(images)]

        text = data['clean_title'].to_list()
        tokens = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        input_ids, attention_mask = tokens['input_ids'], tokens['attention_mask']

        label = torch.from_numpy(data['2_way_label'].values).float()
        return input_ids, attention_mask, label, data
    
    def __load_image(self, idx):
        id = self.data.iloc[idx, 0]
        #Open and normalize image
        img_path = os.path.join(self.images_path, f'{id}.jpg')
        image = Image.open(img_path)

        #Grayscaled images are converted to RGB
        if image.mode != 'RGB': image = image.convert('RGB')
        
        if self.data_aug: image = self.transform(image)
        #Process the images for CLIP
        pixel_values = self.processor(images=image, return_tensors='pt')['pixel_values']

        return pixel_values

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, idx):
        return {
            'pixel_values' : self.__load_image(idx),
            'input_ids' : self.input_ids[idx],
            'attention_mask' : self.attention_mask[idx],
            'label' : self.label[idx]
        }
    
    def collate_fn(self, batch):
        pixel_values = torch.stack([data['pixel_values'] for data in batch]).squeeze()
        input_ids = pad_sequence([data['input_ids'] for data in batch], batch_first=True)
        attention_mask = pad_sequence([data['attention_mask'] for data in batch], batch_first=True)
        label = torch.tensor([data['label'] for data in batch]).to(dtype=torch.long)

        return (
            {
                'pixel_values' : pixel_values,
                'input_ids' : input_ids,
                'attention_mask' : attention_mask
            },
            label
        )
    
class MultimodalWithComments(Dataset):
    def __init__(self, subset, tokenizer='openai/clip-vit-base-patch32', comments_tokenizer='microsoft/deberta-base', processor='openai/clip-vit-base-patch32'):
        self.images_path = f'./dataset/images/{subset}/'
        self.data_path = f'./dataset/{subset}.csv'

        # #CLIP tokenizer by default
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        comments_tokenizer = AutoTokenizer.from_pretrained(comments_tokenizer)
        self.processor = AutoProcessor.from_pretrained(processor)

        self.input_ids, self.attention_mask, self.comments_input_ids, self.comments_attention_mask, self.label, self.data = self.extract_features(tokenizer, comments_tokenizer)

    def extract_features(self, tokenizer, comments_tokenizer):
        images = [img.split('.')[0] for img in os.listdir(self.images_path)]
        data = pd.read_csv(self.data_path)
        data = data[data.id.isin(images)]

        text = data['clean_title'].to_list()
        tokens = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        input_ids, attention_mask = tokens['input_ids'], tokens['attention_mask']

        comments = data['comments'].to_list()
        comments_tokens = comments_tokenizer(comments, padding=True, truncation=True, return_tensors='pt')
        comments_input_ids, comments_attention_mask = comments_tokens['input_ids'], comments_tokens['attention_mask']

        label = torch.from_numpy(data['3_way_label'].values).float()
        return input_ids, attention_mask, comments_input_ids, comments_attention_mask, label, data
    
    def __load_image(self, idx):
        id = self.data.iloc[idx, 0]
        #Open and normalize image
        img_path = os.path.join(self.images_path, f'{id}.jpg')
        image = Image.open(img_path)
        #Grayscaled images are converted to RGB
        if image.mode != 'RGB': 
            image = image.convert('RGB')
        #Process the images for CLIP
        pixel_values = self.processor(images=image, return_tensors='pt')['pixel_values']

        return pixel_values

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, idx):
        return {
            'pixel_values' : self.__load_image(idx),
            'input_ids' : self.input_ids[idx],
            'attention_mask' : self.attention_mask[idx],
            'comments_input_ids' : self.comments_input_ids[idx],
            'comments_attention_mask' : self.comments_attention_mask[idx],
            'label' : self.label[idx]
        }
    
    def collate_fn(self, batch):
        pixel_values = torch.stack([data['pixel_values'] for data in batch]).squeeze()
        input_ids = pad_sequence([data['input_ids'] for data in batch], batch_first=True)
        attention_mask = pad_sequence([data['attention_mask'] for data in batch], batch_first=True)
        comments_input_ids = pad_sequence([data['comments_input_ids'] for data in batch], batch_first=True)
        comments_attention_mask = pad_sequence([data['comments_attention_mask'] for data in batch], batch_first=True)
        label = torch.tensor([data['label'] for data in batch]).to(dtype=torch.long)

        return (
            {
                'pixel_values' : pixel_values,
                'input_ids' : input_ids,
                'attention_mask' : attention_mask,
                'comments_input_ids' : comments_input_ids,
                'comments_attention_mask' : comments_attention_mask
            },
            label
        )
    
class MultimodalWithMetadata(Dataset):
    def __init__(self, subset, tokenizer='openai/clip-vit-base-patch32', processor='openai/clip-vit-base-patch32'):
        self.images_path = f'./dataset/images/{subset}/'
        self.data_path = f'./dataset/{subset}.csv'

        # #CLIP tokenizer by default
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.processor = AutoProcessor.from_pretrained(processor)

        self.input_ids, self.attention_mask, self.score, self.upvote_ratio, self.label, self.data = self.extract_features(tokenizer)

    def extract_features(self, tokenizer):
        images = [img.split('.')[0] for img in os.listdir(self.images_path)]
        data = pd.read_csv(self.data_path)
        data = data[data.id.isin(images)]

        text = data['clean_title'].to_list()
        tokens = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        input_ids, attention_mask = tokens['input_ids'], tokens['attention_mask']

        score = torch.from_numpy(data['score'].values).float()
        upvote_ratio = torch.from_numpy(data['upvote_ratio'].values).float()

        label = torch.from_numpy(data['2_way_label'].values).float()
        return input_ids, attention_mask, score, upvote_ratio, label, data
    
    def __load_image(self, idx):
        id = self.data.iloc[idx, 0]
        #Open and normalize image
        img_path = os.path.join(self.images_path, f'{id}.jpg')
        image = Image.open(img_path)

        #Grayscaled images are converted to RGB
        if image.mode != 'RGB': image = image.convert('RGB')
            
        #Process the images for CLIP
        pixel_values = self.processor(images=image, return_tensors='pt')['pixel_values']

        return pixel_values

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, idx):
        return {
            'pixel_values' : self.__load_image(idx),
            'input_ids' : self.input_ids[idx],
            'attention_mask' : self.attention_mask[idx],
            'score' : self.upvote_ratio[idx],
            'upvote_ratio' : self.upvote_ratio[idx],
            'label' : self.label[idx]
        }
    
    def collate_fn(self, batch):
        pixel_values = torch.stack([data['pixel_values'] for data in batch]).squeeze()
        input_ids = pad_sequence([data['input_ids'] for data in batch], batch_first=True)
        attention_mask = pad_sequence([data['attention_mask'] for data in batch], batch_first=True)
        score = torch.tensor([data['score'] for data in batch])
        upvote_ratio = torch.tensor([data['upvote_ratio'] for data in batch])
        label = torch.tensor([data['label'] for data in batch]).to(dtype=torch.long)

        return (
            {
                'pixel_values' : pixel_values,
                'input_ids' : input_ids,
                'attention_mask' : attention_mask,
                'score' : score,
                'upvote_ratio' : upvote_ratio
            },
            label
        )
    
class OnlyText(Dataset):
    def __init__(self, subset, tokenizer='openai/clip-vit-base-patch32'):
        self.images_path = f'./dataset/images/{subset}/'
        self.data_path = f'./dataset/{subset}.csv'

        # #CLIP tokenizer by default
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        self.input_ids, self.attention_mask, self.label = self.extract_features(tokenizer)

    def extract_features(self, tokenizer):
        images = [img.split('.')[0] for img in os.listdir(self.images_path)]
        data = pd.read_csv(self.data_path)
        data = data[data.id.isin(images)]

        text = data['clean_title'].to_list()
        tokens = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        input_ids, attention_mask = tokens['input_ids'], tokens['attention_mask']

        label = torch.from_numpy(data['2_way_label'].values).float()
        return input_ids, attention_mask, label

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, idx):
        return {
            'input_ids' : self.input_ids[idx],
            'attention_mask' : self.attention_mask[idx],
            'label' : self.label[idx]
        }
    
    def collate_fn(self, batch):
        input_ids = pad_sequence([data['input_ids'] for data in batch], batch_first=True)
        attention_mask = pad_sequence([data['attention_mask'] for data in batch], batch_first=True)
        label = torch.tensor([data['label'] for data in batch]).to(dtype=torch.long)

        return (
            {
                'input_ids' : input_ids,
                'attention_mask' : attention_mask
            },
            label
        )

class TextAndComments(Dataset):
    def __init__(self, subset, tokenizer='openai/clip-vit-base-patch32', comments_tokenizer='microsoft/deberta-base'):
        self.images_path = f'./dataset/sample_dataset/{subset}/'
        self.data_path = f'./dataset/{subset}.csv'

        # #CLIP tokenizer by default
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        comments_tokenizer = AutoTokenizer.from_pretrained(comments_tokenizer)

        self.input_ids, self.attention_mask, self.comments_input_ids, self.comments_attention_mask, self.label = self.extract_features(tokenizer, comments_tokenizer)

    def extract_features(self, tokenizer, comments_tokenizer):
        images = [img.split('.')[0] for img in os.listdir(self.images_path)]
        data = pd.read_csv(self.data_path)
        data = data[data.id.isin(images)]

        text = data['clean_title'].to_list()
        tokens = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        input_ids, attention_mask = tokens['input_ids'], tokens['attention_mask']

        comments = data['comments'].to_list()
        tokens = comments_tokenizer(comments, padding=True, truncation=True, return_tensors='pt')
        comments_input_ids, comments_attention_mask = tokens['input_ids'], tokens['attention_mask']

        label = torch.from_numpy(data['2_way_label'].values).float()
        return input_ids, attention_mask, comments_input_ids, comments_attention_mask, label

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, idx):
        return {
            'input_ids' : self.input_ids[idx],
            'attention_mask' : self.attention_mask[idx],
            'comments_input_ids' : self.comments_input_ids[idx],
            'comments_attention_mask' : self.comments_attention_mask[idx],
            'label' : self.label[idx]
        }
    
    def collate_fn(self, batch):
        input_ids = pad_sequence([data['input_ids'] for data in batch], batch_first=True)
        attention_mask = pad_sequence([data['attention_mask'] for data in batch], batch_first=True)
        comments_input_ids = pad_sequence([data['comments_input_ids'] for data in batch], batch_first=True)
        comments_attention_mask = pad_sequence([data['comments_attention_mask'] for data in batch], batch_first=True)
        label = torch.tensor([data['label'] for data in batch]).to(dtype=torch.long)

        return (
            {
                'input_ids' : input_ids,
                'attention_mask' : attention_mask,
                'comments_input_ids' : comments_input_ids,
                'comments_attention_mask' : comments_attention_mask
            },
            label
        )
    
class OnlyComments(Dataset):
    def __init__(self, subset, comments_tokenizer='microsoft/deberta-base'):
        self.images_path = f'./dataset/sample_dataset/{subset}/'
        self.data_path = f'./dataset/{subset}.csv'

        comments_tokenizer = AutoTokenizer.from_pretrained(comments_tokenizer)

        self.comments_input_ids, self.comments_attention_mask, self.label = self.extract_features(comments_tokenizer)

    def extract_features(self, comments_tokenizer):
        images = [img.split('.')[0] for img in os.listdir(self.images_path)]
        data = pd.read_csv(self.data_path)
        data = data[data.id.isin(images)]

        comments = data['comments'].to_list()
        tokens = comments_tokenizer(comments, padding=True, truncation=True, return_tensors='pt')
        comments_input_ids, comments_attention_mask = tokens['input_ids'], tokens['attention_mask']

        label = torch.from_numpy(data['2_way_label'].values).float()
        return comments_input_ids, comments_attention_mask, label

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, idx):
        return {
            'comments_input_ids' : self.comments_input_ids[idx],
            'comments_attention_mask' : self.comments_attention_mask[idx],
            'label' : self.label[idx]
        }
    
    def collate_fn(self, batch):
        comments_input_ids = pad_sequence([data['comments_input_ids'] for data in batch], batch_first=True)
        comments_attention_mask = pad_sequence([data['comments_attention_mask'] for data in batch], batch_first=True)
        label = torch.tensor([data['label'] for data in batch]).to(dtype=torch.long)

        return (
            {
                'comments_input_ids' : comments_input_ids,
                'comments_attention_mask' : comments_attention_mask
            },
            label
        )

class OnlyImage(Dataset):
    def __init__(self, subset, processor='openai/clip-vit-base-patch32'):
        self.images_path = f'./dataset/images/{subset}/'
        self.data_path = f'./dataset/{subset}.csv'

        #CLIP processor
        self.processor = AutoProcessor.from_pretrained(processor)

        self.label, self.data = self.extract_features()

    def extract_features(self):
        images = [img.split('.')[0] for img in os.listdir(self.images_path)]
        data = pd.read_csv(self.data_path)
        data = data[data.id.isin(images)]

        label = torch.from_numpy(data['2_way_label'].values).float()
        return label, data
    
    def __load_image(self, idx):
        id = self.data.iloc[idx, 0]
        #Open and normalize image
        img_path = os.path.join(self.images_path, f'{id}.jpg')
        image = Image.open(img_path)
        #Grayscaled images are converted to RGB
        if image.mode != 'RGB': 
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image = Image.fromarray(image)
            
        #Process the images for CLIP
        pixel_values = self.processor(images=image, return_tensors='pt')['pixel_values']

        return pixel_values


    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, idx):
        return {
            'pixel_values' : self.__load_image(idx),
            'label' : self.label[idx]
        }
    
    def collate_fn(self, batch):
        pixel_values = torch.stack([data['pixel_values'] for data in batch]).squeeze()
        label = torch.tensor([data['label'] for data in batch]).to(dtype=torch.long)

        return (
            {
                'pixel_values' : pixel_values,
            },
            label
        )
    
class MultimodalWithCommentsAndMetadata(Dataset):
    def __init__(self, subset, tokenizer='openai/clip-vit-base-patch32', comments_tokenizer='microsoft/deberta-base', processor='openai/clip-vit-base-patch32'):
        self.images_path = f'./dataset/images/{subset}/'
        self.data_path = f'./dataset/{subset}.csv'

        # #CLIP tokenizer by default
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        comments_tokenizer = AutoTokenizer.from_pretrained(comments_tokenizer)
        self.processor = AutoProcessor.from_pretrained(processor)

        self.input_ids, self.attention_mask, self.comments_input_ids, self.comments_attention_mask, self.score, self.upvote_ratio, self.num_comments, self.label, self.data = self.extract_features(tokenizer, comments_tokenizer)

    def extract_features(self, tokenizer, comments_tokenizer):
        images = [img.split('.')[0] for img in os.listdir(self.images_path)]
        data = pd.read_csv(self.data_path)
        data = data[data.id.isin(images)]

        text = data['clean_title'].to_list()
        tokens = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        input_ids, attention_mask = tokens['input_ids'], tokens['attention_mask']

        comments = data['comments'].to_list()
        comments_tokens = comments_tokenizer(comments, padding=True, truncation=True, return_tensors='pt')
        comments_input_ids, comments_attention_mask = comments_tokens['input_ids'], comments_tokens['attention_mask']

        score = torch.from_numpy(data['score'].values).float()
        upvote_ratio = torch.from_numpy(data['upvote_ratio'].values).float()
        num_comments = torch.from_numpy(data['num_comments'].values).float()

        label = torch.from_numpy(data['2_way_label'].values).float()
        return input_ids, attention_mask, comments_input_ids, comments_attention_mask, score, upvote_ratio, num_comments, label, data
    
    def __load_image(self, idx):
        id = self.data.iloc[idx, 0]
        #Open and normalize image
        img_path = os.path.join(self.images_path, f'{id}.jpg')
        image = Image.open(img_path)
        #Grayscaled images are converted to RGB
        if image.mode != 'RGB': image = image.convert('RGB')
            
        #Process the images for CLIP
        pixel_values = self.processor(images=image, return_tensors='pt')['pixel_values']

        return pixel_values

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, idx):
        return {
            'pixel_values' : self.__load_image(idx),
            'input_ids' : self.input_ids[idx],
            'attention_mask' : self.attention_mask[idx],
            'comments_input_ids' : self.comments_input_ids[idx],
            'comments_attention_mask' : self.comments_attention_mask[idx],
            'score' : self.score[idx],
            'upvote_ratio' : self.upvote_ratio[idx],
            'num_comments' : self.num_comments[idx],
            'label' : self.label[idx]
        }
    
    def collate_fn(self, batch):
        pixel_values = torch.stack([data['pixel_values'] for data in batch]).squeeze()
        input_ids = pad_sequence([data['input_ids'] for data in batch], batch_first=True)
        attention_mask = pad_sequence([data['attention_mask'] for data in batch], batch_first=True)
        comments_input_ids = pad_sequence([data['comments_input_ids'] for data in batch], batch_first=True)
        comments_attention_mask = pad_sequence([data['comments_attention_mask'] for data in batch], batch_first=True)
        score = torch.tensor([data['score'] for data in batch]).unsqueeze(1)
        upvote_ratio = torch.tensor([data['upvote_ratio'] for data in batch]).unsqueeze(1)
        num_comments = torch.tensor([data['num_comments'] for data in batch]).unsqueeze(1)
        label = torch.tensor([data['label'] for data in batch]).to(dtype=torch.long)

        return (
            {
                'pixel_values' : pixel_values,
                'input_ids' : input_ids,
                'attention_mask' : attention_mask,
                'comments_input_ids' : comments_input_ids,
                'comments_attention_mask' : comments_attention_mask,
                'score' : score,
                'upvote_ratio' : upvote_ratio,
                'num_comments' : num_comments
            },
            label
        )

class OnlyMetadata(Dataset):
    def __init__(self, subset):
        self.images_path = f'./dataset/images/{subset}/'
        self.data_path = f'./dataset/{subset}.csv'

        self.score, self.upvote_ratio, self.num_comments, self.label, self.data = self.extract_features()

    def extract_features(self):
        images = [img.split('.')[0] for img in os.listdir(self.images_path)]
        data = pd.read_csv(self.data_path)
        data = data[data.id.isin(images)]

        score = torch.from_numpy(data['score'].values).float()
        upvote_ratio = torch.from_numpy(data['upvote_ratio'].values).float()
        num_comments = torch.from_numpy(data['num_comments'].values).float()

        label = torch.from_numpy(data['2_way_label'].values).float()
        return score, upvote_ratio, num_comments, label, data

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, idx):
        return {
            'score' : self.score[idx],
            'upvote_ratio' : self.upvote_ratio[idx],
            'num_comments' : self.num_comments[idx],
            'label' : self.label[idx]
        }
    
    def collate_fn(self, batch):
        score = torch.tensor([data['score'] for data in batch]).unsqueeze(1)
        upvote_ratio = torch.tensor([data['upvote_ratio'] for data in batch]).unsqueeze(1)
        num_comments = torch.tensor([data['num_comments'] for data in batch]).unsqueeze(1)
        label = torch.tensor([data['label'] for data in batch]).to(dtype=torch.long)

        return (
            {
                'score' : score,
                'upvote_ratio' : upvote_ratio,
                'num_comments' : num_comments
            },
            label
        )
    
class CorruptDataset(Dataset):
    def __init__(self, images_path):
        self.images = [os.path.join(images_path, img) for img in os.listdir(images_path)]

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224), Image.BILINEAR)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        preprocessed_image = transform(image)
        preprocessed_image = torch.unsqueeze(preprocessed_image, 0)  # Add batch dimension
        return preprocessed_image
    

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        pixel_values = self.preprocess_image(path)
        return {
            'pixel_values' : pixel_values,
            'path': path
        }