
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
  
# importing views from views..py
from . import views
  
urlpatterns = [
     path('/', views.home, name="home"),
     path('/survey', views.ask_info, name="ask_info"),
     path('/result', views.show_result, name="show_result"),
     path('/no_result', views.no_result, name="no_result"),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)