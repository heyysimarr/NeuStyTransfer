# NeuStyTransfer

This project focuses on implementing a Neural Style Transfer (NST) model from scratch, drawing inspiration from Leon Gatys' seminal work on artistic style transfer. Neural Style Transfer is a technique that combines the content of one image with the style of another, resulting in visually appealing synthesised images that blend content and artistic characteristics.
In essence, NST harnesses the power of deep learning to separate and recombine the content and style of arbitrary images, enabling the creation of new artworks that mimic the artistic styles of renowned paintings, photographs, or other visual media. The process involves using convolutional neural networks to extract high-level features from both the content image (e.g., a photograph) and the style image (e.g., a painting), and then optimising these features to minimise the perceptual di erence between the generated image and the style image, while preserving the content of the content image.
<img width="618" alt="Screenshot 2024-06-20 at 5 49 38 PM" src="https://github.com/heyysimarr/NeuStyTransfer/assets/154510691/56576119-62bd-405d-81e2-0e0aa91f1fe3">

IMPLEMENTATION AND CODE SUMMARY :-
Github consist of .ipynb and streamlit.py file :
● NeuStyTransfer.ipynbconsistsofthemaincodewhichwas
written on a kaggle notebook. Main highlights of code :
● ModelandLibraries:Importingnecessarylibraries including PyTorch, torchvision and PIL. Loads VGG-19 model pre-trained on ImageNet.
● ImagePreprocessing:Definesfunctionsforimage loading and transformation using torchvision transforms, converting images to tensors and resizing them.
● FeatureExtraction:UtilisesVGG-19toextractfeatures at specified layers ('conv1_1', 'conv2_1', etc.) for content and style images.
● GramMatrixCalculation:ComputesGrammatrixto capture style representation of feature maps, essential for style loss calculation.
● StyleTransferAlgorithm:Initialisesthegenerated image from the content image, computes content loss by comparing feature maps, and style loss using Gram matrices weighted across di erent layers.
● Optimization:UsesAdamoptimizertominimisethe total loss combining content and style losses, iterating over epochs to refine the generated image.
● streamlit.pycontainsthesamecodeasascripttouseweb interface powered by Streamlit, enabling real-time style transfer and visualisation of the transformations

