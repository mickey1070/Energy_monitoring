# Generated by Django 4.2.5 on 2023-09-08 09:21

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('website', '0002_energy'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='energy',
            name='KwH',
        ),
        migrations.AddField(
            model_name='energy',
            name='timestamp',
            field=models.DateTimeField(auto_now_add=True, default=django.utils.timezone.now),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='energy',
            name='value',
            field=models.FloatField(default=4),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='energy',
            name='status',
            field=models.CharField(max_length=20),
        ),
    ]
