import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel

# TODO modify model(**input) --> model.get_image_features(**inputs)
class CLIPEvaluator(object):
    def __init__(self, device, model_name='openai/clip-vit-base-patch32') -> None:
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.clip_preprocess = CLIPProcessor.from_pretrained(model_name) # Un-normalize from [-1.0, 1.0] (generator output) to [0, 1], to match CLIP input scale assumptions

    def get_text_features(self, text: str) -> torch.Tensor:
        text_inputs = self.clip_preprocess(text=text, return_tensors="pt", padding=True)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        with torch.no_grad():
            text_features = self.model(**text_inputs)

        return text_features

    def get_image_features(self, img: torch.Tensor) -> torch.Tensor:
        image_inputs = self.clip_preprocess(images=img, return_tensors="pt", padding=True)
        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
        with torch.no_grad():
            image_feature = self.model(**image_inputs)

        return image_feature   
    
    def img_to_img_similarity(self, src_images, generated_images):
        src_img_features = self.get_image_features(src_images)
        gen_img_features = self.get_image_features(generated_images)

        # cosine_similarity = F.cosine_similarity(src_img_features, gen_img_features, dim=1)
        return (src_img_features @ gen_img_features.T).mean()

    def txt_to_img_similarity(self, text, images):
        text_features    = self.get_text_features(text)
        image_features = self.get_image_features(images)

        # cosine_similarity = F.cosine_similarity(text_features, gen_img_features, dim=1)
        return (text_features @ image_features.T).mean()
