from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from .forms import SignUpForm,DateRangeForm,EnergySlabRateForm
from .models import energy,EnergySlabRate
import pandas as pd
import matplotlib.pyplot as plt
from django.shortcuts import render
from datetime import date, datetime, timedelta
from django.db.models import Q
import csv
from django.http import HttpResponse
from decimal import Decimal

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

def calculate_cost(consumption):
    cost = Decimal('0.0')  # Initialize cost as a Decimal
    slab_rates = EnergySlabRate.objects.all().order_by('start_usage')

    for slab_rate in slab_rates:
        start = Decimal(str(slab_rate.start_usage))  # Ensure start is Decimal
        end = slab_rate.end_usage
        rate = Decimal(str(slab_rate.rate_per_unit))  # Ensure rate is Decimal

        if end is None:
            cost += (Decimal(str(consumption)) - start) * rate
            break
        elif consumption <= end:
            cost += (Decimal(str(consumption)) - start + Decimal('1.0')) * rate
            break
        else:
            cost += (end - start + Decimal('1.0')) * rate

    return float(cost)

def Dashboard(request):
	today = date.today()
	previous_day = today - timedelta(days=1)
	# Calculate the date for the first day of the current year
	first_day_current_year = today.replace(month=1, day=1)
	data_day = energy.objects.filter(
    timestamp__year=today.year,
    timestamp__month=today.month,
    timestamp__day=today.day
	)
	start_of_month = datetime(today.year, today.month, 1)
	end_of_month = start_of_month.replace(day=1, month=today.month + 1) - timedelta(days=1)
	data_month = energy.objects.filter(
    Q(timestamp__gte=start_of_month) &
    Q(timestamp__lte=end_of_month)
	)
	# Fetch data from the MySQL database for one year (starting from January 1st of the current year)
	start_of_year = datetime(today.year, 1, 1)
	end_of_year = datetime(today.year, 12, 31)
	data_year = energy.objects.filter(
    Q(timestamp__gte=start_of_year) &
    Q(timestamp__lte=end_of_year)
	)
	 # Calculate previous day's total cost
	data_previous_day = energy.objects.filter(
	    timestamp__year=previous_day.year,
	    timestamp__month=previous_day.month,
	    timestamp__day=previous_day.day
	)
	
	 # Calculate the date for the first day of the previous month
	first_day_previous_month = today.replace(day=1) - timedelta(days=1)
	start_of_previous_month = first_day_previous_month.replace(day=1)

	# Fetch data from the MySQL database for the previous month
	data_previous_month = energy.objects.filter(
	    timestamp__year=start_of_previous_month.year,
	    timestamp__month=start_of_previous_month.month
	)

	# Calculate previous year's total cost
	data_previous_year = energy.objects.filter(
	    timestamp__year=first_day_current_year.year - 1
	)
	total_consumption_previous_year = sum(entry.value for entry in data_previous_year)
	total_cost_previous_year = calculate_cost(total_consumption_previous_year)



	# Calculate total consumption and total cost for one day
	total_consumption_day = sum(entry.value for entry in data_day)
	total_cost_day = calculate_cost(total_consumption_day)	
	# Calculate total consumption and total cost for one month
	total_consumption_month = sum(entry.value for entry in data_month)
	total_cost_month = calculate_cost(total_consumption_month)	
	# Calculate total consumption and total cost for one year
	total_consumption_year = sum(entry.value for entry in data_year)
	total_cost_year = calculate_cost(total_consumption_year)
	#predicting the total cost for the year
	average_monthly_cost = total_cost_year / len(data_year)
	# Estimate the remaining months in the year (assuming 12 months in a year)
	remaining_months = 12 - len(data_year)
	predicted_total_cost_year = total_cost_year + (average_monthly_cost * remaining_months)

	total_consumption_previous_day = sum(entry.value for entry in data_previous_day)
	total_cost_previous_day = calculate_cost(total_consumption_previous_day)

	# Calculate total consumption and total cost for the previous month
	total_consumption_previous_month = sum(entry.value for entry in data_previous_month)
	total_cost_previous_month = calculate_cost(total_consumption_previous_month)

	# Calculate total consumption and total cost for the previous year
	total_consumption_previous_year = sum(entry.value for entry in data_previous_year)
	total_cost_previous_year = calculate_cost(total_consumption_previous_year)

	# Compare today's cost with previous day's cost and output a symbol
	if total_cost_day > total_cost_previous_day:
		cost_comparison_symbol = 'red.png'
	elif total_cost_day < total_cost_previous_day:
		cost_comparison_symbol = 'green.png'
	else:
		cost_comparison_symbol = 'equal.png'

	 # Determine the cost comparison symbol for year vs. previous year
	if total_cost_year > total_cost_previous_year:
		cost_comparison_symbol_year = 'red.png'
	elif total_cost_year < total_cost_previous_year:
		cost_comparison_symbol_year = 'green.png'
	else:
		cost_comparison_symbol_year = 'equals.png'

# Determine the cost comparison symbol for month vs. previous month
	if total_cost_month > total_cost_previous_month:
		cost_comparison_symbol_month = 'red.png'
	elif total_cost_month < total_cost_previous_month:
		cost_comparison_symbol_month = 'green.png'
	else:
		cost_comparison_symbol_month = 'equals.png'


	data = energy.objects.all()
    # Create a Pandas DataFrame from the data
	df = pd.DataFrame(data.values())	
	# Group data by 'status' and calculate the sum of 'value' for each status
	grouped_data = df.groupby('status')['value'].sum().reset_index()	
	# Prepare data for plotting
	statuses = grouped_data['status']
	consumption_sum = grouped_data['value']	
	# Create a bar graph
	plt.figure(figsize=(8, 6))
	plt.bar(statuses, consumption_sum)
	plt.xlabel('Status')
	plt.ylabel('Sum of Consumption (kW)')
	#plt.title('Status vs. Sum of Consumption')
	plt.xticks(rotation=45)
	plt.tight_layout()	
	
	# Save the plot as an image
	plt.savefig('static/Graphs/status_vs_consumption.png',transparent=True	)  # Media directory	

	context = {
        'total_consumption_day': total_consumption_day,
        'total_cost_day': total_cost_day,
        'total_cost_month': total_cost_month,	
        'total_cost_year': total_cost_year,
		'average_cost_year': average_monthly_cost * 12,  # Average cost for a full year
    	'predicted_total_cost_year': predicted_total_cost_year,
		'average_monthly_cost':average_monthly_cost,
		'cost_comparison_symbol':cost_comparison_symbol,
		'cost_comparison_symbol_year': cost_comparison_symbol_year,
		'cost_comparison_symbol_month': cost_comparison_symbol_month,
    }

	return render(request, 'Dashboard.html',context)


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

    context = {
        'slab_rates': slab_rates,
        'form': form,
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