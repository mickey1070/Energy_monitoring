# Generated by Django 4.2.5 on 2023-09-08 11:58

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('website', '0003_remove_energy_kwh_energy_timestamp_energy_value_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='energy',
            name='timestamp',
            field=models.DateField(),
        ),
    ]
