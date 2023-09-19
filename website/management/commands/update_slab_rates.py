from django.core.management.base import BaseCommand
from website.models import EnergySlabRate  # Replace 'yourapp' with the name of your app

class Command(BaseCommand):
    help = 'Update energy slab rates'

    def handle(self, *args, **options):
        slab_rates = {
            'slab1': {
                'start': 0,
                'end': 100,
                'rate': 0.10  # $0.10 per unit
            },
            'slab2': {
                'start': 101,
                'end': 200,
                'rate': 0.15  # $0.15 per unit
            },
            'slab3': {
                'start': 201,
                'end': None,
                'rate': 0.20  # $0.20 per unit for anything over 200 units
            }
        }

        for slab_name, slab_data in slab_rates.items():
            start_usage = slab_data['start']
            end_usage = slab_data['end']
            rate_per_unit = slab_data['rate']

            # Check if the slab rate already exists based on some criteria
            # For example, you can check if a slab with the same start and end usage exists
            existing_slab = EnergySlabRate.objects.filter(start_usage=start_usage, end_usage=end_usage).first()

            if existing_slab:
                # If the slab already exists, update its rate
                existing_slab.rate_per_unit = rate_per_unit
                existing_slab.save()
            else:
                # If the slab doesn't exist, create a new one
                EnergySlabRate.objects.create(
                    start_usage=start_usage,
                    end_usage=end_usage,
                    rate_per_unit=rate_per_unit
                )

            self.stdout.write(self.style.SUCCESS(f'Slab rate "{slab_name}" updated'))

