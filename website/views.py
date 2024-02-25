from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from .forms import SignUpForm
from .models import energy,Main
import matplotlib
import io
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from django.shortcuts import render
from datetime import date, datetime, timedelta
from django.db.models import Q
from io import BytesIO
import base64
from django.conf import settings
from django.utils import timezone
from django.utils.dateparse import parse_date
from django.http import HttpResponse
from sklearn.linear_model import LinearRegression
import numpy as np
from django.utils import timezone
import calendar
from django.db.models import Sum
from dateutil.relativedelta import relativedelta

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



def plot_yearly_data(data):
    # Get the timestamp for 12 months ago from today
    twelve_months_ago = datetime.now() - timedelta(days=365)

    # Filter data for the last 12 months
    data_last_12_months = data.filter(timestamp__gte=twelve_months_ago)

    # Extract timestamps and energy values
    timestamps = [entry.timestamp for entry in data_last_12_months]
    energy_values = [entry.value for entry in data_last_12_months]

    # Convert timestamps to months
    months = [timestamp.strftime('%b') for timestamp in timestamps]

    # Group energy values by month
    monthly_energy = {}
    for month, value in zip(months, energy_values):
        if month in monthly_energy:
            monthly_energy[month] += value
        else:
            monthly_energy[month] = value

    # Create sorted list of months
    sorted_months = sorted(monthly_energy.keys(), key=lambda x: list(calendar.month_abbr).index(x))

    # Plot the data
    plt.figure(figsize=(9, 5))
    plt.bar(sorted_months, [monthly_energy[month] for month in sorted_months], color='skyblue')

    plt.xlabel('Month')
    plt.ylabel('Total Energy Consumption (kWh)')
    plt.title('Total Energy Consumption for Last 12 Months')
    plt.xticks(rotation=45)
    plt.grid(False)
    plt.tight_layout()

    # Convert plot to bytes
    buffer = BytesIO()
    plt.savefig(buffer, format='png', transparent=True)
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()

    return plot_data

def Dashboard(request):
    today = datetime.now()

    # Calculate today's consumption in kWh
    start_of_day = today.replace(hour=0, minute=0, second=0, microsecond=0)
    today_consumption_wh = Main.objects.filter(timestamp__gte=start_of_day).aggregate(Sum('value'))['value__sum']
    today_consumption_wh = today_consumption_wh or 0
    today_consumption_kwh = today_consumption_wh / 1000  # Convert Wh to kWh

    # Calculate monthly consumption in kWh
    start_of_month = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    monthly_consumption_wh = Main.objects.filter(timestamp__gte=start_of_month).aggregate(Sum('value'))['value__sum']
    monthly_consumption_wh = monthly_consumption_wh or 0
    monthly_consumption_kwh = monthly_consumption_wh / 1000  # Convert Wh to kWh

    # Calculate yearly consumption in kWh
    start_of_year = today.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    yearly_consumption_wh = Main.objects.filter(timestamp__gte=start_of_year).aggregate(Sum('value'))['value__sum']
    yearly_consumption_wh = yearly_consumption_wh or 0
    yearly_consumption_kwh = yearly_consumption_wh / 1000  # Convert Wh to kWh

    # Fetch previous day, month, and year data
    previous_day = today - timedelta(days=1)
    previous_month = today - relativedelta(months=1)
    previous_year = today - relativedelta(years=1)

    total_consumption_previous_day_wh = Main.objects.filter(
        timestamp__gte=start_of_day - timedelta(days=1),
        timestamp__lt=start_of_day
    ).aggregate(Sum('value'))['value__sum']
    total_consumption_previous_day_wh = total_consumption_previous_day_wh or 0
    total_consumption_previous_day_kwh = total_consumption_previous_day_wh / 1000  # Convert Wh to kWh

    total_consumption_previous_month_wh = Main.objects.filter(
        timestamp__gte=start_of_month - relativedelta(months=1),
        timestamp__lt=start_of_month
    ).aggregate(Sum('value'))['value__sum']
    total_consumption_previous_month_wh = total_consumption_previous_month_wh or 0
    total_consumption_previous_month_kwh = total_consumption_previous_month_wh / 1000  # Convert Wh to kWh

    total_consumption_previous_year_wh = Main.objects.filter(
        timestamp__gte=start_of_year - relativedelta(years=1),
        timestamp__lt=start_of_year
    ).aggregate(Sum('value'))['value__sum']
    total_consumption_previous_year_wh = total_consumption_previous_year_wh or 0
    total_consumption_previous_year_kwh = total_consumption_previous_year_wh / 1000  # Convert Wh to kWh

    # Get the last entry from the 'value' column
    last_entry_wh = Main.objects.latest('timestamp').value
    last_entry_kwh = last_entry_wh / 1000  # Convert Wh to kWh

    # Plot the yearly data
    data_year = Main.objects.filter(timestamp__year=today.year)
    plot_data_year = plot_yearly_data(data_year)

    context = {
        'total_consumption_day': today_consumption_kwh,
        'total_consumption_month': monthly_consumption_kwh,
        'total_consumption_year': yearly_consumption_kwh,
        'total_consumption_previous_day': total_consumption_previous_day_kwh,
        'total_consumption_previous_month': total_consumption_previous_month_kwh,
        'total_consumption_previous_year': total_consumption_previous_year_kwh,
        'last_entry': last_entry_kwh,
        'plot_data_year': plot_data_year,
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
    # Get the latest entry from the Main model
    latest_entry = Main.objects.latest('timestamp')
    
    # Extract latest energy value
    current_consumption = latest_entry.value

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
    
    # Calculate peak current consumption and today's consumption
    peak_current = max(energy_values)
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

def plot_last_7_days(data):
    # Initialize lists to store energy consumption for day and night for each of the last 7 days
    days = []
    day_consumption = []
    night_consumption = []

    # Get the timestamp for 7 days ago
    start_time = timezone.now() - timedelta(days=7)

    # Iterate over the last 7 days
    for i in range(7):
        # Get the start and end timestamp for the current day
        day_start = start_time + timedelta(days=i)
        day_end = day_start + timedelta(days=1)

        # Filter data for the current day
        day_data = data.filter(timestamp__gte=day_start, timestamp__lt=day_end)

        # Initialize variables to store energy consumption during day and night
        day_energy = 0
        night_energy = 0

        # Iterate over data for the current day
        for entry in day_data:
            # Assuming night time is from 6:00 PM to 6:00 AM
            if entry.timestamp.hour < 6 or entry.timestamp.hour >= 18:
                night_energy += entry.value
            else:
                day_energy += entry.value

        # Append energy consumption for the current day to respective lists
        days.append(day_start.strftime('%Y-%m-%d'))
        day_consumption.append(day_energy)
        night_consumption.append(night_energy)

    # Convert lists to numpy arrays for plotting
    days = np.array(days)
    day_consumption = np.array(day_consumption)
    night_consumption = np.array(night_consumption)

    # Create stacked bar graph
    plt.figure(figsize=(8, 3.3))
    plt.bar(days, day_consumption, label='Day', color='lightblue')
    plt.bar(days, night_consumption, bottom=day_consumption, label='Night', color='darkblue')

    plt.xlabel('Date')
    plt.ylabel('Energy Consumption')
    # plt.title('Energy Consumption During Day and Night (Last 7 Days)')
    # plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Convert plot to bytes with transparent background
    buffer = BytesIO()
    plt.savefig(buffer, format='png', transparent=True)
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

    # Plot the last 7 days of energy consumption
    plot_data_7_days = plot_last_7_days(Main.objects.all())  # Assuming all data is used for last 7 days
    
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
        'plot_data': plot_data,
        'plot_data_7_days': plot_data_7_days,
    })