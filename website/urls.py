
from django.urls import path
from .views import machine_graph
from django.conf.urls.static import static

from . import views
urlpatterns = [
    path('', views.login_user, name='login_user'),
    path('logout/', views.logout_user, name='logout_user'),
    path('register/', views.register_user, name='register_user'),
    path('Dashboard/', views.Dashboard, name='Dashboard'),
    path('report/', views.report, name='report'),
    path('download_csv/', views.download_csv, name='download_csv'),
    path('energy_slab_rates/', views.energy_slab_rates, name='energy_slab_rates'),
    path('edit_energy_slab_rate/<int:slab_rate_id>/', views.edit_energy_slab_rate, name='edit_energy_slab_rate'),
    path('delete_energy_slab_rate/<int:slab_rate_id>/', views.delete_energy_slab_rate, name='delete_energy_slab_rate'),
    path('machine_graph/<str:machine_name>/', views.machine_graph, name='machine_graph'),
   path('all_machine_graphs/', views.all_machine_graphs, name='all_machine_graphs'),
]

