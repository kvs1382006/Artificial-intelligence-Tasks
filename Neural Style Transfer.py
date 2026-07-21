import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image Transformation
imsize = 512 if torch.cuda.is_available() else 128
loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor()
])

def load_image(image_path):
    image = Image.open(image_path)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def gram_matrix(input_tensor):
    a, b, c, d = input_tensor.size()
    features = input_tensor.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

def run_style_transfer(content_img_path, style_img_path, num_steps=300, style_weight=1000000, content_weight=1):
    print("[INFO] Initializing VGG19 Neural Style Transfer...")
    
    content_img = load_image(content_img_path)
    style_img = load_image(style_img_path)
    input_img = content_img.clone()
    
    # Load pre-trained VGG19
    vgg = models.vgg19(pretrained=True).features.to(device).eval()
    
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    
    run = [0]
    while run[0] <= num_steps:
        def closure():
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            
            # Simple content and style features extraction (layer representation)
            content_features = vgg(content_img)
            style_features = vgg(style_img)
            target_features = vgg(input_img)
            
            c_loss = torch.mean((target_features - content_features) ** 2)
            s_loss = torch.mean((gram_matrix(target_features) - gram_matrix(style_features)) ** 2)
            
            total_loss = content_weight * c_loss + style_weight * s_loss
            total_loss.backward()
            
            run[0] += 1
            if run[0] % 50 == 0:
                print(f"Step {run[0]}: Total Loss: {total_loss.item():.4f}")
            return total_loss
            
        optimizer.step(closure)
        
    input_img.data.clamp_(0, 1)
    return input_img

if __name__ == "__main__":
    print("=" * 60)
    print("NEURAL STYLE TRANSFER MODULE")
    print("=" * 60)
    print("System configured for processing content and style image pairs.")