from django.db import models


class Record(models.Model):
	created_at = models.DateTimeField(auto_now_add=True)
	first_name = models.CharField(max_length=50)
	last_name =  models.CharField(max_length=50)
	email =  models.CharField(max_length=100)
	phone = models.CharField(max_length=15)
	address =  models.CharField(max_length=100)
	city =  models.CharField(max_length=50)
	state =  models.CharField(max_length=50)
	zipcode =  models.CharField(max_length=20)

	def __str__(self):
		return(f"{self.first_name} {self.last_name}")
	
class energy(models.Model):
	timestamp = models.DateTimeField()
	status = models.CharField(max_length=20)
	value = models.FloatField()
	machine=models.CharField(max_length=20,default='machine1')

	def __str__(self):
	  return f"{self.status} at {self.timestamp}"
	

class Generation(models.Model):
    timestamp = models.DateTimeField()
    value = models.FloatField()
    device = models.CharField(max_length=20)

    def __str__(self):
        return f"{self.device} - {self.timestamp}"

class Main(models.Model):
    timestamp = models.DateTimeField()
    value = models.FloatField()

    def __str__(self):
        return f"Main - Timestamp: {self.timestamp}, Value: {self.value}"	

class EnergyCost(models.Model):
    date = models.DateField()
    cost_day = models.FloatField()
    cost_month = models.FloatField()
    cost_year = models.FloatField()
    cost_previous_day = models.FloatField()
    cost_previous_month = models.FloatField()
    cost_previous_year = models.FloatField()	