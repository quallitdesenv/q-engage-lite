from torchvision import transforms, models
import torch
from PIL import Image

class GenderClassificator:

    classes = ['male', 'female']

    def __init__(self, model):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model)
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 96)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load_model(self, model_path):
        classifier_model = models.resnet18(pretrained=False)
        classifier_model.fc = torch.nn.Linear(classifier_model.fc.in_features, len(self.classes))
        classifier_model.load_state_dict(torch.load(model_path, map_location=self.device))
        classifier_model = classifier_model.to(self.device)
        return classifier_model

    def preprocess_image(self, image):
        image = Image.fromarray(image).convert('RGB')
        return self.preprocess(image).unsqueeze(0).to(self.device)

    def predict(self, image):
        image_preprocessed = self.preprocess_image(image)
        with torch.no_grad():
            self.model.eval()
            output = self.model(image_preprocessed)
            _, predicted = torch.max(output, 1)
            return self.classes[predicted.item()] 