
from django.urls import path

from django.conf.urls.static import static

from . import views
urlpatterns = [
    path('', views.login_user, name='login_user'),
    path('logout/', views.logout_user, name='logout_user'),
    path('register/', views.register_user, name='register_user'),
    path('Dashboard/', views.Dashboard, name='Dashboard'),
    path('machine/', views.machine, name='machine'),
    path('energy_metrics/', views.energy_metrics_view, name='energy_metrics'),
    path('report/', views.report_view, name='report'),
    path('random_forest_prediction/',views.random_forest_prediction,name='random_forest_prediction')
   
]

