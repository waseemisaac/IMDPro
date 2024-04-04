from django.urls import path
from . import views
app_name = 'IDMPro' 

urlpatterns = [
    path('', views.view_index,name='index'),  # Map the root URL to the index view
    path('results/', views.model_results, name='model_results'),  # Map '/results/' to the model_results view
]