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
	
class EnergySlabRate(models.Model):
    start_usage = models.DecimalField(max_digits=10, decimal_places=2)
    end_usage = models.DecimalField(max_digits=10, decimal_places=2,default=float('inf'))
    rate_per_unit = models.DecimalField(max_digits=5, decimal_places=2)

    def __str__(self):
        return f"Slab Rate: {self.start_usage} - {self.end_usage} @ {self.rate_per_unit} per unit"
	