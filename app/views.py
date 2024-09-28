import pickle
import os
import tensorflow as tf
from django.conf import settings
from django.shortcuts import render
from django.http import HttpResponse
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import os
import tensorflow as tf
import pickle
from django.conf import settings
def download_and_save_model(request):
    # Load the pre-trained ResNet50 model
    model = tf.keras.applications.ResNet50(weights='imagenet')

    # Save the model using pickle
    model_path = os.path.join(settings.MEDIA_ROOT, 'issac.pkl')
    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file)
    return HttpResponse("Model downloaded and saved successfully!")



def classify_image(request):
    if request.method == 'GET':
        return render(request, 'classify.html')

    if request.method == 'POST' and request.FILES['image']:
        # Load the pre-trained ResNet50 model if not already loaded
        model_path = os.path.join(settings.MEDIA_ROOT, 'issac.pkl')
        if not os.path.exists(model_path):
            download_and_save_model()

        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)

        # Retrieve the uploaded image
        uploaded_image = request.FILES['image']

        # Save the uploaded image temporarily
        image_path = os.path.join(settings.MEDIA_ROOT, uploaded_image.name)
        with open(image_path, 'wb+') as destination:
            for chunk in uploaded_image.chunks():
                destination.write(chunk)

        # Load and preprocess the image
        img = Image.open(image_path)
        img = img.resize((224, 224))  # Resize image to match model input size
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis

        # Perform inference
        predictions = model.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=3)[0]  # Get top 3 predictions
        classified_results = [(label, round(float(score), 2)) for (_, label, score) in decoded_predictions]

        # Delete the temporarily saved image
        os.remove(image_path)

        # Return the result
        return render(request, 'result.html', {'results': classified_results})

    return HttpResponse("No image uploaded!")



#------------------------------------
# second part











def explore(request):
# Load the dataset
    import rasterio
    import matplotlib.pyplot as plt

    # Path to the downloaded GFC dataset file
    gfc_file = 'dataset.tif'

    # Load the GFC dataset using rasterio
    with rasterio.open(gfc_file) as src:
        gfc_data = src.read(1)  # Read the first band (forest cover change)

    # Visualize the GFC dataset
    plt.figure(figsize=(10, 10))
    plt.imshow(gfc_data, cmap='viridis')  # Change colormap as needed
    plt.colorbar(label='Forest Cover Change')
    plt.title('Global Forest Change Dataset')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()
    return render(request, 'explore.html')






