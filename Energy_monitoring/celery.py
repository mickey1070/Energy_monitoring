# celery.py
from __future__ import absolute_import, unicode_literals
import os
from celery import Celery
from celery.schedules import crontab  # Import crontab module

# set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Energy_monitoring.settings')

# create a Celery instance and configure it using the settings from Django
celery_app = Celery('Energy_monitoring')

# Load task modules from all registered Django app configs.
celery_app.config_from_object('django.conf:settings', namespace='CELERY')

# Auto-discover tasks in all installed apps
celery_app.autodiscover_tasks()

# Your other configurations go here

# Define your periodic tasks
celery_app.conf.beat_schedule = {
    'transfer-monthly-costs': {
        'task': 'website.tasks.transfer_monthly_costs',
        'schedule': crontab(day_of_month='1'),  # Run on the 1st day of the month
    },
}
