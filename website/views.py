from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from .forms import SignUpForm
from .models import energy,Main
import matplotlib
matplotlib.use('Agg')
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
import os,io
from django.conf import settings
from django.utils import timezone
from django.db.models.functions import TruncMonth
from django.utils.dateparse import parse_date
import urllib, base64
from django.http import HttpResponse
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from django.utils import timezone
import json
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


def machine(request):
    if request.method == 'POST':
        selected_machine = request.POST.get('machine')
        selected_option = request.POST.get('options')

        # Fetch all entries for the selected machine
        machine_entries = energy.objects.filter(machine=selected_machine)

        # Initialize variables to store data for comparison
        current_data = []
        previous_data = []

        if selected_option == 'DvsD':
            # Calculate current day and previous day
            today = datetime.now().date()
            previous_day = today - timedelta(days=1)

            # Filter entries for the current day and previous day
            current_day_entries = machine_entries.filter(timestamp__date=today)
            previous_day_entries = machine_entries.filter(timestamp__date=previous_day)

            # Get values for the current day and previous day
            current_data = [entry.value for entry in current_day_entries]
            previous_data = [entry.value for entry in previous_day_entries]

            # Plot the comparison bar graph
            plt.bar(['Current Day', 'Previous Day'], [sum(current_data), sum(previous_data)])
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.title(f'{selected_machine} - DvsD Comparison')
            
            # Convert plot to bytes
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            buffer.seek(0)
            # Encode plot bytes as base64 string
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()

        elif selected_option == 'MvsM':
            # Calculate current month and previous month
            today = datetime.now().date()
            previous_month = today.replace(day=1) - timedelta(days=1)
            current_month_start = today.replace(day=1)

            # Filter entries for the current month and previous month
            current_month_entries = machine_entries.filter(
                timestamp__year=today.year,
                timestamp__month=today.month
            )
            previous_month_entries = machine_entries.filter(
                timestamp__year=previous_month.year,
                timestamp__month=previous_month.month
            )

            # Get values for the current month and previous month
            current_data = [entry.value for entry in current_month_entries]
            previous_data = [entry.value for entry in previous_month_entries]

            # Plot the comparison bar graph
            plt.bar(['Current Month', 'Previous Month'], [sum(current_data), sum(previous_data)])
            
            plt.title(f'{selected_machine} - MvsM Comparison')
            
            # Convert plot to bytes
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            buffer.seek(0)
            # Encode plot bytes as base64 string
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()

        elif selected_option == 'YvsY':
            # Calculate current year and previous year
            today = datetime.now().date()
            previous_year = today.replace(year=today.year - 1)

            # Filter entries for the current year and previous year
            current_year_entries = machine_entries.filter(timestamp__year=today.year)
            previous_year_entries = machine_entries.filter(timestamp__year=previous_year.year)

            # Get values for the current year and previous year
            current_data = [entry.value for entry in current_year_entries]
            previous_data = [entry.value for entry in previous_year_entries]

            # Plot the comparison bar graph
            plt.bar(['Current Year', 'Previous Year'], [sum(current_data), sum(previous_data)])
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.title(f'{selected_machine} - YvsY Comparison')
            
            # Convert plot to bytes
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            buffer.seek(0)
            # Encode plot bytes as base64 string
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()

        else:
            return render(request, 'machine.html', {'error': 'Invalid option selected'})

        # Construct HTML snippet to display the graph
        html_plot = f'<img src="data:image/png;base64,{plot_data}" />'
        context = {'graph': html_plot, 'machines': energy.objects.values_list('machine', flat=True).distinct(),
                   'options': ['DvsD', 'MvsM', 'YvsY']}
        return render(request, 'machine.html', context)

    else:
        machines = energy.objects.values_list('machine', flat=True).distinct()
        options = ['DvsD', 'MvsM', 'YvsY']
        return render(request, 'machine.html', {'machines': machines, 'options': options})
    
def calculate_energy_metrics():
    # Get the timestamp for 24 hours ago in UTC timezone
    start_time = timezone.now() - timedelta(hours=24)
    
    # Filter data from the last 24 hours
    data = Main.objects.filter(timestamp__gte=start_time)
    
    if not data.exists():
        print("No data available for the last 24 hours.")
        return None, None, None, None

    # Extract timestamps and energy values
    timestamps = np.array([(entry.timestamp - start_time).total_seconds() / 3600 for entry in data])
    energy_values = np.array([entry.value for entry in data])

    # Reshape timestamps for regression
    X = timestamps.reshape(-1, 1)

    # Fit Linear Regression model
    model = LinearRegression()
    model.fit(X, energy_values)

    # Use the model to predict standby energy at 24 hours
    standby_energy = max(0, model.predict([[24]])[0])  # Ensure standby energy is non-negative
    
    # Calculate peak current consumption, current consumption, and today's consumption
    peak_current = max(energy_values)
    current_consumption = energy_values[-1]
    today_consumption = sum(energy_values)

    return standby_energy, peak_current, current_consumption, today_consumption

def plot_last_24_hours(data):
    # Get the timestamp for 24 hours ago in UTC timezone
    start_time = timezone.now() - timedelta(hours=24)
    
    # Extract timestamps and energy values
    timestamps = [(entry.timestamp - start_time).total_seconds() / 3600 for entry in data]
    energy_values = [entry.value for entry in data]
    
    # Create new arrays for timestamps and energy values at 30-minute intervals
    timestamps_30mins = []
    energy_values_30mins = []
    last_hour = None
    
    for timestamp, energy_value in zip(timestamps, energy_values):
        hour = int(timestamp)
        minute = int((timestamp - hour) * 60)
        if last_hour is None or hour > last_hour:
            timestamps_30mins.append(timestamp)
            energy_values_30mins.append(energy_value)
            last_hour = hour
        elif minute % 30 == 0:
            timestamps_30mins.append(timestamp)
            energy_values_30mins.append(energy_value)
    
    # Create the plot
    plt.figure(figsize=(20, 4))
    plt.plot(timestamps_30mins, energy_values_30mins, marker='o', linestyle='-', alpha=0.5)  # Set alpha for transparency
    plt.xlabel('Time (hours ago)')
    plt.ylabel('Energy Value')
    plt.title('Energy Readings in the Last 24 Hours')
    plt.grid(True)
    
    # Convert plot to bytes with transparent background
    buffer = BytesIO()
    plt.savefig(buffer, format='png', transparent=True)  # Set transparent=True
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()
    
    return plot_data



def energy_metrics_view(request):
    # Get the timestamp for 24 hours ago in UTC timezone
    start_time = timezone.now() - timedelta(hours=24)
    
    # Filter data from the last 24 hours
    data = Main.objects.filter(timestamp__gte=start_time)
    
    if not data.exists():
        print("No data available for the last 24 hours.")
        return None, None, None, None

    # Calculate standby energy, peak current consumption, current consumption, and today's consumption
    standby_energy, peak_current, current_consumption, today_consumption = calculate_energy_metrics()
    
    # Plot the last 24 hours of energy readings
    plot_data = plot_last_24_hours(data)
    
    # Cast values to integers
    standby_energy = int(standby_energy)
    peak_current = int(peak_current)
    current_consumption = int(current_consumption)
    today_consumption = int(today_consumption)
    
    return render(request, 'energy_metrics.html', {
        'standby_energy': standby_energy,
        'peak_current': peak_current,
        'current_consumption': current_consumption,
        'today_consumption': today_consumption,
        'plot_data': plot_data
    })