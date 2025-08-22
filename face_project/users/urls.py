# users/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('register/', views.register_user, name='register'),
    path('recognize/', views.recognize_user, name='recognize'),
       path("recognize_user/", views.recognize_user, name="recognize_user"),
]


