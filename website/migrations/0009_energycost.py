# Generated by Django 4.2.5 on 2023-10-02 07:00

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('website', '0008_monthlyenergycost'),
    ]

    operations = [
        migrations.CreateModel(
            name='EnergyCost',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('date', models.DateField()),
                ('cost_day', models.FloatField()),
                ('cost_month', models.FloatField()),
                ('cost_year', models.FloatField()),
                ('cost_previous_day', models.FloatField()),
                ('cost_previous_month', models.FloatField()),
                ('cost_previous_year', models.FloatField()),
            ],
        ),
    ]