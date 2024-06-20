from PIL import Image
import torchvision.models as models 
import matplotlib.pyplot as plt
import numpy as np
import torch
import streamlit as st
import torch.nn as nn 
import torch.optim as optim
import torchvision.transforms as transforms  
from torchvision.utils import save_image

#Loading the VGG model with its pre-trained weights
vgg = models.vgg19(pretrained=True).features

# Freezing all VGG parameters since we're only optimizing the target image
# Setting requires_grad to False for all parameters (weights), so that no gradients are computed for the model's weights.
for param in vgg.parameters():
    param.requires_grad_(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg.to(device)  

# IMAGE PRE-PROCESSING

def image_loader(path):
    image=Image.open(path).convert('RGB')
    #defining the image transformation steps to be performed before feeding them to the model
    loader=transforms.Compose([transforms.Resize((512,512)), transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    #The preprocessing steps involves resizing the image and then converting it to a tensor
    image=loader(image)[:3,:,:].unsqueeze(0)
    return image

def im_convert(tensor):
    image = tensor.cpu().clone().detach().numpy()  # Convert tensor to numpy array
    image = image.squeeze()  # Remove the batch dimension
    image = image.transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))  # De-normalize
    image = image.clip(0, 1)  # Clip to ensure valid pixel range [0, 1]

    return image

#FUNCTION TO GET FEATURES OF AN IMAGE
def get_features(image, model, layers=None):
    
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1', 
                  '10': 'conv3_1', 
                  '19': 'conv4_1',
                  '21': 'conv4_2',  ## content representation
                  '28': 'conv5_1'}
        
    features = {}
    x = image
    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
            
    return features

# GRAM MATRIX FOR STYLE LOSS CALCULATION
def gram_matrix(tensor):

    _, channel, height, width = tensor.size()
    
    # reshape so we're multiplying the features for each channel -> tensor.view(channel,height * width)
    
    gram = torch.mm(tensor.view(channel,height*width),tensor.view(channel,height*width).t())
    
    return gram 

# Streamlit app
st.set_page_config(page_title="Neural Style Transfer App")

st.sidebar.image("/Users/simar/Desktop/NEURAL STYLE TRANSFER.png")
st.sidebar.title("Neural Style Transfer")
choice = st.sidebar.radio("Navigation", ["Upload Content & Style", "Run Style Transfer", "Download Generated Image"])
st.sidebar.info("This app applies neural style transfer using VGG19.")

if choice == "Upload Content & Style":
    st.title("Upload Content and Style Images")
    content_file = st.file_uploader("Upload Content Image", type=['jpg', 'jpeg', 'png'])
    style_file = st.file_uploader("Upload Style Image", type=['jpg', 'jpeg', 'png'])

    if content_file and style_file:
        content_image = Image.open(content_file)
        style_image = Image.open(style_file)

        # Display uploaded images
        st.subheader("Content Image:")
        st.image(content_image, caption="Uploaded Content Image", use_column_width=True)

        st.subheader("Style Image:")
        st.image(style_image, caption="Uploaded Style Image", use_column_width=True)

        # Save images to temp files for processing
        content_image.save('content_temp.jpg')
        style_image.save('style_temp.jpg')
        
elif choice == "Run Style Transfer":
    if st.button("Generate Image"):
        # Load uploaded images
        content_path = 'content_temp.jpg'
        style_path = 'style_temp.jpg'

        #Loading the original and the style image
        original_image=image_loader(content_path).to(device)
        style_image=image_loader(style_path).to(device)

        #Creating the generated image from the original image
        generated_image=original_image.clone().requires_grad_(True).to(device)
        generated_image.size()

        # get content and style features only once before training
        content_features = get_features(original_image, vgg)
        style_features = get_features(style_image, vgg)
        target_features=get_features(generated_image,vgg)
        
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)

        # Calculating the gram matrices for each layer of our style representation
        style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
        # Let's weight each style layer differently 
        # weighting earlier layers more will result in *larger* style artifacts

        style_weights = {'conv1_1': 1.,
                 'conv2_1': 0.75,
                 'conv3_1': 0.2,
                 'conv4_1': 0.2,
                 'conv5_1': 0.2}

        content_weight = 1  # alpha
        style_weight = 1e9  # beta

        # for displaying the target image, intermittently
        show_every = 50

        
        optimizer = optim.Adam([generated_image], lr=0.003)
        epochs = 500  # decide how many iterations to update your image (5000)

        for e in range(epochs):
    
            target_features = get_features(generated_image, vgg)
            content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)

            style_loss = 0
            
            for layer in style_weights:
              
              target_feature = target_features[layer]
              # Getting target image gram matrix for the particular layer
              target_gram = gram_matrix(target_feature)
              _, d, h, w = target_feature.shape

              # Getting style image gram matrix for the particular layer
              style_gram = style_grams[layer]
              
              # the style loss for one layer, weighted appropriately
              layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)

              # add to the style loss
              style_loss += layer_style_loss / (d * h * w)
        
            # Total loss
            total_loss = content_weight * content_loss + style_weight * style_loss
    
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
    
            # display intermediate images and print the loss
            if (e+1) % show_every == 0:
              # Convert generated_image tensor to numpy array and display using Streamlit
              generated_image_numpy = im_convert(generated_image)
    
              # Display the generated image along with the total loss
              st.image(generated_image_numpy, caption=f'Generated Image - Iteration {e} - Loss: {total_loss.item():.4f}', use_column_width=True)

            
        # Display final generated image
        st.subheader("Final Generated Image")
        st.image(im_convert(generated_image), caption="Final Generated Image", use_column_width=True)

        # Save final generated image
        final_image = im_convert(generated_image)
        final_image_pil = Image.fromarray((final_image * 255).astype(np.uint8))
        final_image_pil.save('generated_image.jpg')

if choice == "Download Generated Image":
    download_button_placeholder = st.empty()
    if download_button_placeholder.button("Download Generated Image"):
        with open('generated_image.jpg', 'rb') as f:
            download_button_placeholder.download_button(label='Download Generated Image', data=f, file_name='generated_image.jpg')
