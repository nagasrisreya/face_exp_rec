from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('register/', views.register_user, name='register'),
    
    path('recognize/', views.recognize_user, name='recognize'),
    path('recognize_user/', views.recognize_user, name='recognize_user'),
    path('start-confidence-test/', views.start_confidence_test, name='start_confidence_test'),
    path('process-frame/', views.process_frame, name='process_frame'),
    path('emotion-stats/', views.emotion_stats, name='emotion_stats'),
    # NEW counting endpoints
    path('start-counting/', views.start_counting, name='start_counting'),
    path('stop-counting/', views.stop_counting, name='stop_counting'),
    path('counting-status/', views.counting_status, name='counting_status'),
]



