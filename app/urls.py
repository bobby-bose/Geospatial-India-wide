from django.urls import path
from .views import download_and_save_model, classify_image, explore
urlpatterns = [
    path('download_model/', download_and_save_model, name='download_model'),
    path('classify_image/', classify_image, name='classify_image'),
    path('explore/', explore, name='explore'),
]