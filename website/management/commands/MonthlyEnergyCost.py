# myapp/management/commands/calculate_monthly_energy_cost.py
from django.core.management.base import BaseCommand
from datetime import date
from website.models import Energy, MonthlyEnergyCost

class Command(BaseCommand):
    help = 'Calculate and store monthly energy cost'

    def handle(self, *args, **kwargs):
        # Calculate the current month's energy cost
        today = date.today()
        start_of_month = today.replace(day=1)
        end_of_month = start_of_month.replace(day=1, month=today.month + 1) - timedelta(days=1)
        data_month = Energy.objects.filter(
            timestamp__range=(start_of_month, end_of_month)
        )
        total_consumption_month = sum(entry.value for entry in data_month)
        total_cost_month = calculate_cost(total_consumption_month, slab_rates)

        # Insert or update the monthly cost record
        monthly_cost, created = MonthlyEnergyCost.objects.get_or_create(
            month=start_of_month,
            defaults={'total_cost': total_cost_month}
        )

        if not created:
            monthly_cost.total_cost = total_cost_month
            monthly_cost.save()
