from django.contrib import admin
from .models import *
from svm_predict import *
# Register your models here.
from django.utils.safestring import mark_safe


class ForexPredictionAdmin(admin.ModelAdmin):

    search_fields = ("result",)
    list_display = ("id",'country',"year","result")


    def save_model(self, request, obj, form, change):
    	print("===============HERE ==================")
        obj.result = getPrediction(obj.country.name,obj.year)
        super(ForexPredictionAdmin, self).save_model(request, obj, form, change)



#admin.site.site_header = 'FOREX PREDICTION1'
#admin.site.register(ForexPrediction,ForexPredictionAdmin)
#admin.site.register(Country)