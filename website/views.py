from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from .forms import SignUpForm, AddRecordForm
from .models import Record,energy
import pandas as pd
import matplotlib.pyplot as plt
from django.shortcuts import render
from datetime import date, datetime, timedelta
from django.db.models import Q


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
			return redirect('dashboard')
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
			return redirect('dashboard')
	else:
		form = SignUpForm()
		return render(request, 'register.html', {'form':form})

	return render(request, 'register.html', {'form':form})




#using the below code if you want status vs cost
""" cost_per_unit = 0.10  # Replace with your cost per unit

def dashboard(request):
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

    return render(request, 'dashboard.html') """

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
    cost = 0
    for slab, values in slab_rates.items():
        start = values['start']
        end = values['end']
        rate = values['rate']

        if end is None:
            cost += (consumption - start) * rate
            break
        elif consumption <= end:
            cost += (consumption - start + 1) * rate
            break
        else:
            cost += (end - start + 1) * rate

    return cost


def dashboard(request):
	today = date.today()
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
    }

	return render(request, 'dashboard.html',context)