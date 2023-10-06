import torch
import torch.nn as nn
import numpy as np
from transformers import CLIPProcessor, CLIPModel

class CLIP_FeatureExtractor(nn.Module):
    """docstring for FeatureExtractor."""

    def __init__(self, model_name='openai/clip-vit-base-patch32', device='cpu'):
        super(CLIP_FeatureExtractor, self).__init__()
        self.device = device
        # freeze all parameters of pretrained model
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.clip_preprocess = CLIPProcessor.from_pretrained(model_name)

    def forward(self,x):
        x = torch.transpose(x, 0, 1) #(bs,n,3,h,w) --> (n,bs,3,h,w)
        video_feature = []
        for frames in x:
            image_inputs = self.clip_preprocess(images=frames, return_tensors="pt", padding=True)
            image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
            with torch.no_grad():
                video_feature.append(self.model.get_image_features(**image_inputs))
        video_feature = torch.transpose(torch.stack(video_feature), 0, 1) #(n,bs,512) --> (bs,n,512)

        return video_feature

    def get_image_feature(self, x: torch.Tensor):
        image_inputs = self.clip_preprocess(images=x, return_tensors="pt", padding=True)
        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
        with torch.no_grad():
            image_feature = self.model(**image_inputs)

        return image_feature
    
    def get_text_features(self, text: str) -> torch.Tensor:
        text_inputs = self.clip_preprocess(text=text, return_tensors="pt", padding=True)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        with torch.no_grad():
            text_features = self.model(**text_inputs)

        return text_features

    def txt_to_img_similarity(self, text, image, pre_process=False):
        if not pre_process:
            text_features    = self.get_text_features(text)
            image_features = self.get_image_features(image)
        else:
            text_features = text
            image_features = image

        # cosine_similarity = F.cosine_similarity(text_features, gen_img_features, dim=1)
        return (text_features @ image_features.T).mean()