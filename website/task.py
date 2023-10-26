# costs/tasks.py
from celery import shared_task
from .models import EnergyCost
from django.db.models import Sum
from datetime import date, timedelta
from celery.schedules import crontab

@shared_task
def transfer_monthly_costs():
    # Get the first and last day of the current month
    today = date.today()
    first_day_of_month = today.replace(day=1)
    last_day_of_month = first_day_of_month.replace(month=today.month + 1) - timedelta(days=1)

    # Get the total monthly costs for the current month
    total_monthly_cost = EnergyCost.objects.filter(
        date__gte=first_day_of_month,
        date__lte=last_day_of_month
    ).aggregate(Sum('cost_month'))['cost_month__sum'] or 0

    # Create or update the overall monthly cost record
    overall_monthly_cost, created = EnergyCost.objects.update_or_create(
        date=last_day_of_month,
        defaults={'cost_month': total_monthly_cost}
    )
