from django.db import models
from django.contrib.auth.models import User
# Create your models here.


class Country(models.Model):
    name = models.CharField(max_length=255)

    def __str__(self):
        return str(self.name)

class ForexPrediction(models.Model):
    country = models.ForeignKey(Country,on_delete=models.CASCADE)
    year = models.IntegerField()
    result = models.CharField(max_length=255,blank=True,null=True)

    def __str__(self):
        return str(self.result)



