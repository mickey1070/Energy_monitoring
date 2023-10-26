# Generated by Django 4.2.5 on 2023-09-19 07:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('website', '0007_alter_energy_timestamp'),
    ]

    operations = [
        migrations.CreateModel(
            name='MonthlyEnergyCost',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('month', models.DateField(unique=True)),
                ('total_cost', models.DecimalField(decimal_places=2, max_digits=10)),
            ],
        ),
    ]