from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from .forms import SignUpForm
from .models import energy
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from django.shortcuts import render
from datetime import date, datetime, timedelta
from django.db.models import Q
import csv
from django.http import HttpResponse
from decimal import Decimal
from django.http import HttpResponse
from io import BytesIO
import base64
import os
from django.conf import settings
from django.utils import timezone
from django.db.models.functions import TruncMonth
from django.utils.dateparse import parse_date

# Create your views here.

def login_user(request):
	# Check to see if logging in
	if request.method == 'POST':
		username = request.POST['username']
		password = request.POST['password']
		# Authenticate
		user = authenticate(request, username=username, password=password)
		if user is not None:
			login(request, user)
			messages.success(request, "You Have Been Logged In!")
			return redirect('Dashboard')
		else:
			messages.success(request, "There Was An Error Logging In, Please Try Again...")
			return redirect('login_user')
	else:
		return render(request, 'login_user.html', {})
	
def logout_user(request):
	logout(request)
	messages.success(request, "You Have Been Logged Out...")
	return redirect('login_user')

def register_user(request):
	if request.method == 'POST':
		form = SignUpForm(request.POST)
		if form.is_valid():
			form.save()
			# Authenticate and login
			username = form.cleaned_data['username']
			password = form.cleaned_data['password1']
			user = authenticate(username=username, password=password)
			login(request, user)
			messages.success(request, "You Have Successfully Registered! Welcome!")
			return redirect('Dashboard')
	else:
		form = SignUpForm()
		return render(request, 'register.html', {'form':form})

	return render(request, 'register.html', {'form':form})




#using the below code if you want status vs cost
""" cost_per_unit = 0.10  # Replace with your cost per unit

def Dashboard(request):
    # Fetch data from the MySQL database
    data = energy.objects.all()

    # Calculate the total consumption for each status
    status_consumption = {}
    for entry in data:
        if entry.status in status_consumption:
            status_consumption[entry.status] += entry.value
        else:
            status_consumption[entry.status] = entry.value

    # Calculate the cost for each status
    status_cost = {status: consumption * cost_per_unit for status, consumption in status_consumption.items()}

    # Prepare data for plotting
    statuses = list(status_cost.keys())
    costs = list(status_cost.values())

    # Create a bar graph
    plt.figure(figsize=(10, 6))
    plt.bar(statuses, costs)
    plt.xlabel('Status')
    plt.ylabel('Cost')
    plt.title('Status vs. Cost')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot as an image
    plt.savefig('static/Graphs/status_vs_consumption.png')  # Media directory

    return render(request, 'Dashboard.html') """



def Dashboard(request):
    today = date.today()
    previous_day = today - timedelta(days=1)
    first_day_current_year = today.replace(month=1, day=1)

    # Calculate today's consumption
    first_entry_day = energy.objects.filter(
        timestamp__year=today.year,
        timestamp__month=today.month,
        timestamp__day=today.day
    ).order_by('timestamp').first()

    last_entry_day = energy.objects.filter(
        timestamp__year=today.year,
        timestamp__month=today.month,
        timestamp__day=today.day
    ).order_by('-timestamp').first()

    if first_entry_day and last_entry_day:
        today_consumption = last_entry_day.value - first_entry_day.value
    else:
        today_consumption = 0

    # Fetch the first entry for the month
    start_of_month = datetime(today.year, today.month, 1)
    first_entry_month = energy.objects.filter(
        timestamp__year=today.year,
        timestamp__month=today.month
    ).order_by('timestamp').first()

    # Get the last entry from the 'value' column for the month
    last_entry_month = energy.objects.latest('timestamp')

    # Calculate monthly consumption
    if first_entry_month and last_entry_month:
        monthly_consumption = last_entry_month.value - first_entry_month.value
    else:
        monthly_consumption = 0

    # Debugging: Print first and last entries, and monthly consumption
    print("Today's Consumption:", today_consumption)
    print("First Entry (Month):", first_entry_month)
    print("Last Entry (Month):", last_entry_month)
    print("Monthly Consumption:", monthly_consumption)

    # Calculate total consumption for the day, month, and year
    total_consumption_day = today_consumption if today_consumption else 0

    start_of_year = datetime(today.year, 1, 1)
    end_of_year = datetime(today.year, 12, 31)
    data_year = energy.objects.filter(
        Q(timestamp__gte=start_of_year) &
        Q(timestamp__lte=end_of_year)
    )

    # Get the first entry of the year
    first_entry_year = energy.objects.filter(
        timestamp__year=start_of_year.year
    ).order_by('timestamp').first()

    # Get the last entry of the year
    last_entry_year = energy.objects.filter(
        timestamp__year=end_of_year.year
    ).order_by('-timestamp').first()

    # Calculate yearly consumption
    if first_entry_year and last_entry_year:
        total_consumption_year = last_entry_year.value - first_entry_year.value
    else:
        total_consumption_year = 0

    # Fetch previous day, month, and year data
    data_previous_day = energy.objects.filter(
        timestamp__year=previous_day.year,
        timestamp__month=previous_day.month,
        timestamp__day=previous_day.day
    )

    first_day_previous_month = today.replace(day=1) - timedelta(days=1)
    start_of_previous_month = first_day_previous_month.replace(day=1)
    data_previous_month = energy.objects.filter(
        timestamp__year=start_of_previous_month.year,
        timestamp__month=start_of_previous_month.month
    )

    data_previous_year = energy.objects.filter(
        timestamp__year=first_day_current_year.year - 1
    )

    total_consumption_previous_day = float(sum(entry.value for entry in data_previous_day))
    total_consumption_previous_month = float(sum(entry.value for entry in data_previous_month))
    total_consumption_previous_year = float(sum(entry.value for entry in data_previous_year))

    # Get the last entry from the 'value' column
    last_entry = energy.objects.latest('timestamp').value

    context = {
        'total_consumption_day': total_consumption_day,
        'total_consumption_month': monthly_consumption,
        'total_consumption_year': total_consumption_year,
        'total_consumption_previous_day': total_consumption_previous_day,
        'total_consumption_previous_month': total_consumption_previous_month,
        'total_consumption_previous_year': total_consumption_previous_year,
        'last_entry': last_entry,
    }

    return render(request, 'Dashboard.html', context)

