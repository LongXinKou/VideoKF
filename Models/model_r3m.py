import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from r3m import load_r3m
from transformers import AutoTokenizer, AutoModel, AutoConfig
r3m_hidden_dim = {'resnet18':512, 'resnet34':512, 'resnet50':2048}

class R3M_FeatureExtractor(nn.Module):
    """docstring for FeatureExtractor."""

    def __init__(self, model_name='resnet18', device='cpu'):
        super(R3M_FeatureExtractor, self).__init__()
        self.device = device
        # freeze all parameters of pretrained model
        self.model = load_r3m(model_name)   # resnet18, resnet34, resnet50
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        # language encoder
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.language_model = AutoModel.from_pretrained('distilbert-base-uncased').to(self.device)

        self.r3m_preprocess = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor()]) # ToTensor() divides by 255
        self.hidden_dim = r3m_hidden_dim[model_name]

    def forward(self,x):
        x = torch.transpose(x, 0, 1) #(bs,n,3,h,w) --> (n,bs,3,h,w)
        video_feature = []
        for frames in x:
            image_inputs = frames   # not use r3m_preprocess
            with torch.no_grad():
                video_feature.append(self.model(image_inputs * 255.0)) # R3M expects image input to be [0-255]
        video_feature = torch.transpose(torch.stack(video_feature), 0, 1) #(n,bs,2048) --> (bs,n,2048)

        return video_feature
    
    def get_text_feature(self, langs):
        with torch.no_grad():
            encoded_input = self.tokenizer(langs, return_tensors='pt', padding=True).to(self.device)
            lang_embedding = self.language_model(**encoded_input).last_hidden_state
            lang_embedding = lang_embedding.mean(1)
        return lang_embedding  #(1,768)
    
    def get_image_features(self, img: torch.Tensor) -> torch.Tensor:
        image_inputs = img   # not use r3m_preprocess
        with torch.no_grad():
            image_feature = self.model(image_inputs * 255.0) # R3M expects image input to be [0-255]
        return image_feature 
    
    def txt_to_img_similarity(self, text, image, pre_process=False):
        if not pre_process:
            text_features = self.get_text_feature(text)
            image_features = self.get_image_features(image)
        else:
            text_features = text
            image_features = image

        # cosine_similarity = F.cosine_similarity(text_features, gen_img_features, dim=1)
        return (text_features @ image_features.T).mean()
