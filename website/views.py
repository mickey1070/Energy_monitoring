from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from .forms import SignUpForm,DateRangeForm,EnergySlabRateForm
from .models import energy,EnergySlabRate,MonthlyEnergyCost,EnergyCost
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

def calculate_cost(consumption, slab_rates):
    cost = Decimal('0.0')  # Initialize cost as a Decimal

    for slab_rate in slab_rates:
        start = Decimal(str(slab_rate.start_usage))  # Ensure start is Decimal
        end = slab_rate.end_usage
        rate = Decimal(str(slab_rate.rate_per_unit))  # Ensure rate is Decimal

        if end is None or end == float('inf'):
            cost += (Decimal(str(consumption)) - start) * rate
            break
        elif consumption <= end:
            cost += (Decimal(str(consumption)) - start + Decimal('1.0')) * rate
            break
        else:
            cost += (Decimal(str(end)) - start + Decimal('1.0')) * rate

    return float(cost)

def calculate_yearly_energy_cost(data_year, slab_rates):
    total_consumption_year = sum(entry.value for entry in data_year)
    total_cost_year = calculate_cost(total_consumption_year, slab_rates)

    # Predicting the total cost for the year
    average_monthly_cost = total_cost_year / len(data_year)
    remaining_months = 12 - len(data_year)
    predicted_total_cost_year = total_cost_year + (average_monthly_cost * remaining_months)

    return total_cost_year, predicted_total_cost_year



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

def report(request):
    if request.method == 'POST':
        form = DateRangeForm(request.POST)
        if form.is_valid():
            start_date = form.cleaned_data['start_date']
            end_date = form.cleaned_data['end_date']
            
            # Query the database for data within the specified date range
            data_within_range = energy.objects.filter(
                timestamp__range=(start_date, end_date)
            )
            
            context = {
                'data_within_range': data_within_range,
                'form': form,
            }
            return render(request, 'report.html', context)
    else:
        form = DateRangeForm()
    
    context = {'form': form}
    return render(request, 'report.html', context)

def download_csv(request):
    if request.method == 'POST':
        # Your data retrieval logic here (similar to date_range_query view)
        start_date = ...
        end_date = ...
        data_within_range = energy.objects.filter(
            timestamp__range=(start_date, end_date)
        )

        # Create a CSV response
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="data.csv"'

        # Create a CSV writer
        writer = csv.writer(response)

        # Write the header row
        writer.writerow(['Date', 'Value'])

        # Write data rows
        for entry in data_within_range:
            writer.writerow([entry.timestamp, entry.value])

        return response
	
def energy_slab_rates(request):
    slab_rates = EnergySlabRate.objects.all()
    form = EnergySlabRateForm()

    if request.method == 'POST':
        form = EnergySlabRateForm(request.POST)
        if form.is_valid():
            if not form.cleaned_data['end_usage']:
                form.cleaned_data['end_usage'] = float('inf')  # Set to positive infinity or another suitable value
            form.save()
            return redirect('energy_slab_rates')

    # Set positive_infinity in the context
    positive_infinity = float('inf')

    context = {
        'slab_rates': slab_rates,
        'form': form,
        'positive_infinity': positive_infinity,
    }

    return render(request, 'energy_slab_rates.html', context)

def edit_energy_slab_rate(request, slab_rate_id):
    slab_rate = EnergySlabRate.objects.get(id=slab_rate_id)

    if request.method == 'POST':
        form = EnergySlabRateForm(request.POST, instance=slab_rate)
        if form.is_valid():
			
            form.save()
            return redirect('energy_slab_rates')
    else:
        form = EnergySlabRateForm(instance=slab_rate)

    context = {
        'form': form,
    }

    return render(request, 'edit_energy_slab_rate.html', context)

def delete_energy_slab_rate(request, slab_rate_id):
    slab_rate = EnergySlabRate.objects.get(id=slab_rate_id)
    slab_rate.delete()
    return redirect('energy_slab_rates')


def generate_line_chart(data, title):
    # Assume data is a Pandas DataFrame with 'timestamp' and 'value' columns
    plt.figure(figsize=(10, 6))
    plt.plot(data['timestamp'], data['value'], marker='o')
    plt.title(title)
    plt.xlabel('Timestamp')
    plt.ylabel('Consumption (kW)')
    plt.grid(True)

    # Save the plot to a BytesIO object
    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    plt.close()

    # Move the BytesIO pointer to the beginning of the stream
    image_stream.seek(0)

    # Encode the image data as a base64 string
    encoded_image = base64.b64encode(image_stream.read()).decode('utf-8')

    return encoded_image


def machine_graph(request, machine_name):
    # Get the current date
    today = date.today()

    # Calculate the date for the first day of the current year
    start_of_year = date(today.year, 1, 1)

    # Fetch data from the MySQL database for the current year and the specified machine
    data_current_year = energy.objects.filter(
        machine=machine_name,
        timestamp__year=today.year
    ).order_by('timestamp')

    # Fetch data from the MySQL database for the previous year and the specified machine
    data_previous_year = energy.objects.filter(
        machine=machine_name,
        timestamp__year=today.year - 1
    ).order_by('timestamp')

    # Prepare data for plotting
    timestamps_current_year = [entry.timestamp for entry in data_current_year]
    consumption_values_current_year = [entry.value for entry in data_current_year]

    timestamps_previous_year = [entry.timestamp for entry in data_previous_year]
    consumption_values_previous_year = [entry.value for entry in data_previous_year]

    # Create a line graph
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps_current_year, consumption_values_current_year, label=f'Current Year Consumption')
    plt.plot(timestamps_previous_year, consumption_values_previous_year, label=f'Previous Year Consumption', linestyle='dashed')

    plt.xlabel('Timestamp')
    plt.ylabel('Energy Consumption')
    plt.title(f'Machine {machine_name} Consumption Comparison (Current Year vs. Previous Year)')
    plt.legend()
    plt.tight_layout()

    # Save the plot to a file in the static folder
    graph_filename = f'static/Graphs/machine_graphs/machine_{machine_name}_consumption_comparison.png'
    plt.savefig(graph_filename, transparent=True)
    plt.close()

    # Render the HTML template with the filename of the saved graph
    return render(request, 'machine_graph.html', {'graph_filename': graph_filename})

def all_machine_graphs(request):
    # Get the list of all machines
    machines = energy.objects.values('machine').distinct()

    # Create a list to store the filenames for each machine
    machine_graphs = []

    for machine in machines:
        machine_name = machine['machine']

        # Fetch data for the current date
        current_data_day = energy.objects.filter(
            machine=machine_name,
            timestamp__year=date.today().year,
            timestamp__month=date.today().month,
            timestamp__day=date.today().day
        ).order_by('timestamp')

        # Fetch data for the previous day
        previous_date_day = date.today() - timedelta(days=1)
        previous_data_day = energy.objects.filter(
            machine=machine_name,
            timestamp__year=previous_date_day.year,
            timestamp__month=previous_date_day.month,
            timestamp__day=previous_date_day.day
        ).order_by('timestamp')

        # Fetch data for the current month
        start_of_month = date.today().replace(day=1)
        end_of_month = (start_of_month + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        current_data_month = energy.objects.filter(
            machine=machine_name,
            timestamp__range=(start_of_month, end_of_month)
        ).order_by('timestamp')

        # Fetch data for the previous month
        first_day_previous_month = date.today().replace(day=1) - timedelta(days=1)
        start_of_previous_month = first_day_previous_month.replace(day=1)
        end_of_previous_month = (start_of_previous_month + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        previous_data_month = energy.objects.filter(
            machine=machine_name,
            timestamp__range=(start_of_previous_month, end_of_previous_month)
        ).order_by('timestamp')

        # Fetch data for the current year
        current_data_year = energy.objects.filter(
            machine=machine_name,
            timestamp__year=date.today().year
        ).order_by('timestamp')

        # Fetch data for the previous year
        previous_data_year = energy.objects.filter(
            machine=machine_name,
            timestamp__year=date.today().year - 1
        ).order_by('timestamp')

        # Prepare data for plotting
        def prepare_data(data):
            timestamps = [entry.timestamp for entry in data]
            consumption_values = [entry.value for entry in data]
            return timestamps, consumption_values

        timestamps_current_day, consumption_values_current_day = prepare_data(current_data_day)
        timestamps_previous_day, consumption_values_previous_day = prepare_data(previous_data_day)

        timestamps_current_month, consumption_values_current_month = prepare_data(current_data_month)
        timestamps_previous_month, consumption_values_previous_month = prepare_data(previous_data_month)

        timestamps_current_year, consumption_values_current_year = prepare_data(current_data_year)
        timestamps_previous_year, consumption_values_previous_year = prepare_data(previous_data_year)

        # Create line graphs for each comparison
        plt.figure(figsize=(15, 10))

        # Day vs Previous Day
        plt.subplot(3, 1, 1)
        plt.plot(timestamps_current_day, consumption_values_current_day, label=f'Current Day Consumption')
        plt.plot(timestamps_previous_day, consumption_values_previous_day, label=f'Previous Day Consumption', linestyle='dashed')
        plt.xlabel('Timestamp')
        plt.ylabel('Energy Consumption')
        plt.title(f'Machine {machine_name} Day vs. Previous Day Consumption Comparison')
        plt.legend()
        plt.tight_layout()

        # Month vs Previous Month
        plt.subplot(3, 1, 2)
        plt.plot(timestamps_current_month, consumption_values_current_month, label=f'Current Month Consumption')
        plt.plot(timestamps_previous_month, consumption_values_previous_month, label=f'Previous Month Consumption', linestyle='dashed')
        plt.xlabel('Timestamp')
        plt.ylabel('Energy Consumption')
        plt.title(f'Machine {machine_name} Month vs. Previous Month Consumption Comparison')
        plt.legend()
        plt.tight_layout()

        # Year vs Previous Year
        plt.subplot(3, 1, 3)
        plt.plot(timestamps_current_year, consumption_values_current_year, label=f'Current Year Consumption')
        plt.plot(timestamps_previous_year, consumption_values_previous_year, label=f'Previous Year Consumption', linestyle='dashed')
        plt.xlabel('Timestamp')
        plt.ylabel('Energy Consumption')
        plt.title(f'Machine {machine_name} Year vs. Previous Year Consumption Comparison')
        plt.legend()
        plt.tight_layout()

        # Save the plots to files in the static folder
        graph_filename_day = f'static/Graphs/machine_graphs/machine_{machine_name}_day_comparison.png'
        plt.savefig(graph_filename_day, transparent=True)
        plt.close()

        graph_filename_month = f'static/Graphs/machine_graphs/machine_{machine_name}_month_comparison.png'
        plt.savefig(graph_filename_month, transparent=True)
        plt.close()

        graph_filename_year = f'static/Graphs/machine_graphs/machine_{machine_name}_yearly_comparison.png'
        plt.savefig(graph_filename_year, transparent=True)
        plt.close()

        # Append the filenames and machine name to the list
        machine_graphs.append({
            'machine_name': machine_name,
            'graph_filename_day': graph_filename_day,
            'graph_filename_month': graph_filename_month,
            'graph_filename_year': graph_filename_year,
        })

    # Render the HTML template with the list of filenames and machine names
    return render(request, 'all_machine_graphs.html', {'machine_graphs': machine_graphs})