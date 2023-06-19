# infer on model
import torch

# import test image and transform to tensor
from PIL import Image
from torchvision import transforms

img = Image.open('data/example_data/mri_tumor.jpg').convert('RGB')
img = img.resize((224, 224))
img = transforms.ToTensor()(img)
img = img.unsqueeze(0)

# load model
model_path = "saved_models/20230618-200340_model.pt"
base_model_path = "saved_models/base_model.pt"
model = torch.load(base_model_path)
model.load_state_dict(torch.load(model_path))
model.eval()

# infer on image
with torch.no_grad():
    output = model(img)
    # print predicted class
    print("Predicted label: {}".format(torch.round(torch.sigmoid(output))))

# compare to actual label
print("Actual label: 1")

# infer on different image
img = Image.open('data/example_data/no_tumor.jpg').convert('RGB')
img = img.resize((224, 224))
img = transforms.ToTensor()(img)
img = img.unsqueeze(0)

with torch.no_grad():
    output = model(img)
    # print predicted class
    print("Predicted label: {}".format(torch.round(torch.sigmoid(output))))

# compare to actual label
print("Actual label: 0")