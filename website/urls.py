
from django.urls import path

from django.conf.urls.static import static

from . import views
urlpatterns = [
    path('', views.login_user, name='login_user'),
    path('logout/', views.logout_user, name='logout_user'),
    path('register/', views.register_user, name='register_user'),
    path('Dashboard/', views.Dashboard, name='Dashboard'),
    path('machine/', views.machine, name='machine'),
   
]

