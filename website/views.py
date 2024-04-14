from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from .forms import SignUpForm
from .models import Main
import matplotlib
import io
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
from django.shortcuts import render
from datetime import date, datetime, timedelta
from io import BytesIO
import base64
from django.conf import settings
from django.utils import timezone
from django.http import HttpResponse
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from django.utils import timezone
import calendar
from django.db.models import Sum
from dateutil.relativedelta import relativedelta
import csv
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
from django.db.models import Sum
from statsmodels.tsa.arima.model import ARIMA
from django.http import JsonResponse




def login_user(request):
	
	if request.method == 'POST':
		username = request.POST['username']
		password = request.POST['password']
		
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





""" cost_per_unit = 0.10  

def Dashboard(request):
    
    data = energy.objects.all()

    
    status_consumption = {}
    for entry in data:
        if entry.status in status_consumption:
            status_consumption[entry.status] += entry.value
        else:
            status_consumption[entry.status] = entry.value

    
    status_cost = {status: consumption * cost_per_unit for status, consumption in status_consumption.items()}

    
    statuses = list(status_cost.keys())
    costs = list(status_cost.values())

    
    plt.figure(figsize=(10, 6))
    plt.bar(statuses, costs)
    plt.xlabel('Status')
    plt.ylabel('Cost')
    plt.title('Status vs. Cost')
    plt.xticks(rotation=45)
    plt.tight_layout()

    
    plt.savefig('static/Graphs/status_vs_consumption.png')  

    return render(request, 'Dashboard.html') """



def plot_yearly_data(data):
    current_date = datetime.now().date()
    twelve_months_ago = current_date - timedelta(days=365)
    current_day = current_date.day

    # Filter data for the last 12 months up to the current day
    data_last_12_months = data.filter(timestamp__gte=twelve_months_ago, timestamp__date__lte=current_date)

    # Extract timestamps and energy values
    timestamps = [entry.timestamp for entry in data_last_12_months]
    energy_values = [entry.value for entry in data_last_12_months]

    # Get month names
    months = [timestamp.strftime('%b') for timestamp in timestamps]

    # Aggregate energy consumption for each month
    monthly_energy = {}
    for month, value in zip(months, energy_values):
        if month in monthly_energy:
            monthly_energy[month] += value
        else:
            monthly_energy[month] = value

    # Sort months
    sorted_months = sorted(monthly_energy.keys(), key=lambda x: list(calendar.month_abbr).index(x))

    # Plot data
    plt.figure(figsize=(9, 5))
    plt.bar(sorted_months, [monthly_energy[month] for month in sorted_months], color='skyblue')

    plt.xlabel('Month')
    plt.ylabel('Total Energy Consumption (kWh)')
    plt.title('Total Energy Consumption for Last 12 Months (Up to Current Day)')
    plt.grid(False)
    plt.tight_layout()

    # Convert plot to base64 encoded image
    buffer = BytesIO()
    plt.savefig(buffer, format='png', transparent=True)
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()

    return plot_data

def predict_energy_consumption(request):
    # Retrieve historical data
    historical_data = Main.objects.all().values('timestamp', 'value')
    
    # Convert QuerySet to DataFrame
    df = pd.DataFrame(list(historical_data))

    # Convert timestamp to datetime index
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # Resample to lower frequency (e.g., hourly)
    df_resampled = df.resample('H').mean().ffill()  # Resample to hourly data, forward fill missing values

    # Fit ARIMA model
    model = ARIMA(df_resampled['value'], order=(5,1,0))  # Example order, you may need to tune this
    model_fit = model.fit()

    # Make forecast for next month
    forecast = model_fit.forecast(steps=24*30)  # Adjust steps for the number of hours in the month

    # Print the forecasted values
    print("Forecasted values for the next month:")
    print(forecast)

    # Calculate total forecasted consumption
    total_forecast = sum(forecast)

    return total_forecast

def predict_yearly_consumption(request):
    # Retrieve historical data
    historical_data = Main.objects.all().values('timestamp', 'value')
    
    # Convert QuerySet to DataFrame
    df = pd.DataFrame(list(historical_data))

    # Convert timestamp to datetime index
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # Resample to lower frequency (e.g., hourly)
    df_resampled = df.resample('H').mean().ffill()  # Resample to hourly data, forward fill missing values

    # Fit ARIMA model
    model = ARIMA(df_resampled['value'], order=(5,1,0))  # Example order, you may need to tune this
    model_fit = model.fit()

    # Make forecast for the entire year
    forecast = model_fit.forecast(steps=24*365)  # Adjust steps for the number of hours in the year

    # Print the forecasted values
    print("Forecasted values for the entire year:")
    print(forecast)

    # Calculate total forecasted yearly consumption
    total_forecast_yearly = sum(forecast)

    return total_forecast_yearly

def detect_anomalies(actual_values, forecast_values):
    anomalies = []

    # Set a threshold for anomaly detection (e.g., 10% deviation)
    threshold = 0.1

    # Compare actual values with forecasted values
    if isinstance(actual_values, (int, float)) and isinstance(forecast_values, (int, float)):
        if abs(actual_values - forecast_values) > threshold * actual_values:
            anomalies.append((actual_values, forecast_values))
    else:
        for actual, forecast in zip(actual_values, forecast_values):
            if abs(actual - forecast) > threshold * actual:
                anomalies.append((actual, forecast))
    
    return anomalies
def get_energy_recommendations(anomalies, forecast_monthly, forecast_yearly):
    recommendations = []

    # Example recommendations based on anomalies
    if anomalies:
        recommendations.append("There are anomalies detected in the energy consumption. Investigate further.")

    # Example recommendations based on forecast
    if isinstance(forecast_monthly, list):
        monthly_forecast = forecast_monthly[0] if forecast_monthly else 0  # Get the first element or 0 if the list is empty
    else:
        monthly_forecast = forecast_monthly
    
    if monthly_forecast > forecast_yearly:
        recommendations.append("The forecasted monthly consumption is higher than the forecasted yearly consumption. Consider implementing energy-saving measures.")

    return recommendations




def Dashboard(request):
    today = datetime.now()

    # Get start of the day, month, and year
    start_of_day = today.replace(hour=0, minute=0, second=0, microsecond=0)
    start_of_month = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    start_of_year = today.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    print("Start of the month date:", start_of_month.date())
    print("Start of the day time:", start_of_day.time())


    # Get total consumption for today, month, and year in kWh
    today_consumption_wh = Main.objects.filter(timestamp__gte=start_of_day).aggregate(Sum('value'))['value__sum'] or 0
    today_consumption_kwh = today_consumption_wh / 1000

    monthly_consumption_wh = Main.objects.filter(timestamp__gte=start_of_month).aggregate(Sum('value'))['value__sum'] or 0
    monthly_consumption_kwh = monthly_consumption_wh / 1000

    yearly_consumption_wh = Main.objects.filter(timestamp__gte=start_of_year).aggregate(Sum('value'))['value__sum'] or 0
    yearly_consumption_kwh = yearly_consumption_wh / 1000

    # Get total consumption for previous day, month, and year in kWh
    total_consumption_previous_day_wh = Main.objects.filter(
        timestamp__gte=start_of_day - timedelta(days=1),
        timestamp__lt=start_of_day
    ).aggregate(Sum('value'))['value__sum'] or 0
    total_consumption_previous_day_kwh = total_consumption_previous_day_wh / 1000

    total_consumption_previous_month_wh = Main.objects.filter(
        timestamp__gte=start_of_month,
        timestamp__lte=today
    ).aggregate(Sum('value'))['value__sum'] or 0
    total_consumption_previous_month_kwh = total_consumption_previous_month_wh / 1000

    total_consumption_previous_year_wh = Main.objects.filter(
        timestamp__gte=start_of_year,
        timestamp__lte=today
    ).aggregate(Sum('value'))['value__sum'] or 0
    total_consumption_previous_year_kwh = total_consumption_previous_year_wh / 1000

    # Get the last entry value in kWh
    last_entry_wh = Main.objects.latest('timestamp').value
    last_entry_kwh = last_entry_wh / 1000

    # Get yearly data for plotting
    data_year = Main.objects.filter(timestamp__year=today.year)
    # Assuming plot_yearly_data is implemented elsewhere
    plot_data_year = plot_yearly_data(data_year)

    data_yearly = Main.objects.filter(timestamp__gte=start_of_year)
    data_monthly = Main.objects.filter(timestamp__gte=start_of_month)
    forecast_monthly_value=predict_energy_consumption(request)
    forecast_monthly = predict_energy_consumption(request)
    forecast_yearly = predict_yearly_consumption(request)

    # Detect anomalies
    actual_data = Main.objects.filter(timestamp__gte=start_of_day).values_list('value', flat=True)
    forecast_monthly = [forecast_monthly] * len(actual_data)  # Repeat forecast value for each actual value
    anomalies = detect_anomalies(actual_data, forecast_monthly)

    # Get energy efficiency recommendations
    recommendations = get_energy_recommendations(anomalies, forecast_monthly, forecast_yearly)

    context = {
        'total_consumption_day': today_consumption_kwh,
        'total_consumption_month': monthly_consumption_kwh,
        'total_consumption_year': yearly_consumption_kwh,
        'total_consumption_previous_day': total_consumption_previous_day_kwh,
        'total_consumption_previous_month': total_consumption_previous_month_kwh,
        'total_consumption_previous_year': total_consumption_previous_year_kwh,
        'last_entry': last_entry_kwh,
        'plot_data_year': plot_data_year,
        'forecast_monthly': forecast_monthly_value,
        'forecast_yearly': forecast_yearly,
        'anomalies': anomalies,
        'recommendations': recommendations,
    }

    return render(request, 'Dashboard.html', context)

def machine(request):
    if request.method == 'POST':
        selected_option = request.POST.get('options')

        
        main_entries = Main.objects.all()

        
        current_data = []
        previous_data = []

        if selected_option == 'DvsD':
            
            today = datetime.now().date()
            previous_day = today - timedelta(days=1)

            
            current_day_entries = main_entries.filter(timestamp__date=today)
            previous_day_entries = main_entries.filter(timestamp__date=previous_day)

            
            current_data = [entry.value for entry in current_day_entries]
            previous_data = [entry.value for entry in previous_day_entries]

            
            plt.bar(['Current Day', 'Previous Day'], [sum(current_data), sum(previous_data)])
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.title('DvsD Comparison')
            
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            buffer.seek(0)
            
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()

        elif selected_option == 'MvsM':
            
            today = datetime.now().date()
            previous_month = today.replace(day=1) - timedelta(days=1)
            current_month_start = today.replace(day=1)

            
            current_month_entries = main_entries.filter(
                timestamp__year=today.year,
                timestamp__month=today.month
            )
            previous_month_entries = main_entries.filter(
                timestamp__year=previous_month.year,
                timestamp__month=previous_month.month
            )

            
            current_data = [entry.value for entry in current_month_entries]
            previous_data = [entry.value for entry in previous_month_entries]

            
            plt.bar(['Current Month', 'Previous Month'], [sum(current_data), sum(previous_data)])
            
            plt.title('MvsM Comparison')
            
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            buffer.seek(0)
            
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()

        elif selected_option == 'YvsY':
            
            today = datetime.now().date()
            previous_year = today.replace(year=today.year - 1)

            
            current_year_entries = main_entries.filter(timestamp__year=today.year)
            previous_year_entries = main_entries.filter(timestamp__year=previous_year.year)

            
            current_data = [entry.value for entry in current_year_entries]
            previous_data = [entry.value for entry in previous_year_entries]

            
            plt.bar(['Current Year', 'Previous Year'], [sum(current_data), sum(previous_data)])
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.title('YvsY Comparison')
            
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            buffer.seek(0)
            
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()

        else:
            return render(request, 'machine.html', {'error': 'Invalid option selected'})

        
        html_plot = f'<img src="data:image/png;base64,{plot_data}" />'
        context = {'graph': html_plot, 'options': ['DvsD', 'MvsM', 'YvsY']}
        return render(request, 'machine.html', context)

    else:
        options = ['DvsD', 'MvsM', 'YvsY']
        return render(request, 'machine.html', {'options': options})
    
def calculate_energy_metrics():
    # Get the latest entry
    latest_entry = Main.objects.latest('timestamp')
    
    # Get current consumption
    current_consumption = latest_entry.value
    
    # Get start time for the last 24 hours
    start_time = timezone.now() - timedelta(hours=24)
    
    # Filter data for the last 24 hours
    data = Main.objects.filter(timestamp__gte=start_time)
    
    if not data.exists():
        print("No data available for the last 24 hours.")
        return None, None, None, None
    
    # Extract timestamps and energy values
    timestamps = np.array([(entry.timestamp - start_time).total_seconds() / 3600 for entry in data])
    energy_values = np.array([entry.value for entry in data])
    
    # Train a Linear Regression model to predict standby energy
    model = LinearRegression()
    model.fit(timestamps.reshape(-1, 1), energy_values)
    
    # Predict standby energy for the day (24 hours)
    standby_energy = max(0, model.predict([[24]])[0])
    
    # Find the peak current consumption for the whole day
    peak_current = max(energy_values)
    
    # Fetch current value using the current time
    current_value = fetch_current_value()
    
    # Calculate total consumption for today alone in kWh
    today_consumption = sum(energy_values) / 1000  # Assuming energy values are in Wh
    
    return standby_energy, peak_current, current_value, today_consumption

def fetch_current_value():
    # Get the current time
    current_time = timezone.now()
    
    # Get the start time for the last 24 hours
    start_time = current_time - timedelta(hours=24)
    
    # Filter data for the last 24 hours
    data = Main.objects.filter(timestamp__gte=start_time)
    
    if not data.exists():
        print("No data available for the last 24 hours.")
        return None
    
    # Find the entry with the closest timestamp to the current time
    closest_entry = min(data, key=lambda entry: abs(entry.timestamp - current_time))
    
    return closest_entry.value if closest_entry else None


def plot_last_24_hours(data):
    end_time = timezone.now()
    start_time = end_time - timedelta(hours=24)
    
    # Filter data for the last 24 hours
    filtered_data = [entry for entry in data if start_time <= entry.timestamp <= end_time]
    
    hourly_energy_consumption = {}  # Dictionary to store hourly energy consumption
    
    for entry in filtered_data:
        hour = entry.timestamp.replace(minute=0, second=0, microsecond=0)  # Round down to the nearest hour
        if hour not in hourly_energy_consumption:
            hourly_energy_consumption[hour] = 0
        hourly_energy_consumption[hour] += entry.value
    
    timestamps = []
    energy_values = []
    
    for hour in sorted(hourly_energy_consumption.keys()):
        timestamps.append((hour - start_time).total_seconds() / 3600)
        energy_values.append(hourly_energy_consumption[hour])
    
    plt.figure(figsize=(20, 4))
    plt.plot(timestamps, energy_values, marker='o', linestyle='-', alpha=0.5)  
    plt.xlabel('Time (hours ago)')
    plt.ylabel('Total Energy Consumed in Hour')
    plt.grid(True)
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', transparent=True)  
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()
    
    return plot_data

def plot_last_7_days(data):
    
    days = []
    day_consumption = []
    night_consumption = []

    
    start_time = timezone.now() - timedelta(days=7)

    
    for i in range(7):
        
        day_start = start_time + timedelta(days=i)
        day_end = day_start + timedelta(days=1)

        
        day_data = data.filter(timestamp__gte=day_start, timestamp__lt=day_end)

        
        day_energy = 0
        night_energy = 0

        
        for entry in day_data:
            
            if entry.timestamp.hour < 6 or entry.timestamp.hour >= 18:
                night_energy += entry.value
            else:
                day_energy += entry.value

        
        days.append(day_start.strftime('%Y-%m-%d'))
        day_consumption.append(day_energy)
        night_consumption.append(night_energy)

    
    days = np.array(days)
    day_consumption = np.array(day_consumption)
    night_consumption = np.array(night_consumption)

    
    plt.figure(figsize=(8, 3.3))
    plt.bar(days, day_consumption, label='Day', color='lightblue')
    plt.bar(days, night_consumption, bottom=day_consumption, label='Night', color='darkblue')

    plt.xlabel('Date')
    plt.ylabel('Energy Consumption')
    
    
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', transparent=True)
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()

    return plot_data



def energy_metrics_view(request):
    
    start_time = timezone.now() - timedelta(hours=24)
    
    
    data = Main.objects.filter(timestamp__gte=start_time)
    
    if not data.exists():
        print("No data available for the last 24 hours.")
        return None, None, None, None

    
    standby_energy, peak_current, current_consumption, today_consumption = calculate_energy_metrics()
    
    
    plot_data = plot_last_24_hours(data)

    
    plot_data_7_days = plot_last_7_days(Main.objects.all())  
    
    
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


def report_view(request):
    if request.method == 'POST':
        start_date = request.POST.get('start_date')
        end_date = request.POST.get('end_date')

        
        data = Main.objects.filter(timestamp__range=[start_date, end_date])

        
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="main_data.csv"'

        
        writer = csv.writer(response)
        writer.writerow(['Timestamp', 'Value'])
        for entry in data:
            writer.writerow([entry.timestamp, entry.value])

        return response

    return render(request, 'report.html')