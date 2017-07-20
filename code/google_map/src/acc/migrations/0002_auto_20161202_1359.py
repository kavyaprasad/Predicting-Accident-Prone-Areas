# -*- coding: utf-8 -*-
# Generated by Django 1.9 on 2016-12-02 13:59
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('acc', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='pointofinterest',
            name='Accident_Index',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name='pointofinterest',
            name='Accident_Severity',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name='pointofinterest',
            name='Carriageway_Hazards',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name='pointofinterest',
            name='Date',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name='pointofinterest',
            name='Day_of_Week',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name='pointofinterest',
            name='Did_Police_Officer_Attend_Scene_of_Accident',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name='pointofinterest',
            name='Junction_Control',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name='pointofinterest',
            name='Junction_Detail',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name='pointofinterest',
            name='LSOA_of_Accident_Location',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name='pointofinterest',
            name='Light_Conditions',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name='pointofinterest',
            name='Local_Authority_District',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name='pointofinterest',
            name='Local_Authority_Highway',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name='pointofinterest',
            name='Location_Easting_OSGR',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name='pointofinterest',
            name='Location_Northing_OSGR',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name='pointofinterest',
            name='Number_of_Casualties',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name='pointofinterest',
            name='Number_of_Vehicles',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name='pointofinterest',
            name='Pedestrian_Crossing_Human_Control',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name='pointofinterest',
            name='Pedestrian_Crossing_Physical_Facilities',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name='pointofinterest',
            name='Police_Force',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name='pointofinterest',
            name='Road_Surface_Conditions',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name='pointofinterest',
            name='Road_Type',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name='pointofinterest',
            name='Special_Conditions_at_Site',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name='pointofinterest',
            name='Speed_limit',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name='pointofinterest',
            name='Time',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name='pointofinterest',
            name='Urban_or_Rural_Area',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name='pointofinterest',
            name='Weather_Conditions',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name='pointofinterest',
            name='first_Road_Class',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name='pointofinterest',
            name='first_Road_Number',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name='pointofinterest',
            name='second_Road_Class',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AlterField(
            model_name='pointofinterest',
            name='second_Road_Number',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
    ]
