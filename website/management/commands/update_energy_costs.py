# yourapp/management/commands/update_energy_costs.py

from django.core.management.base import BaseCommand
from website.views import calculate_yearly_energy_cost

class Command(BaseCommand):
    help = 'Update yearly energy costs and monthly cost data'

    def handle(self, *args, **kwargs):
        calculate_yearly_energy_cost()
        self.stdout.write(self.style.SUCCESS('Successfully updated energy costs and monthly data.'))
