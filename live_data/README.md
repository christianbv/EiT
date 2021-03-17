# Oversikt
Christian: Litt lengere ned finner du den orginale README.md filen for lastX (starter med Usage and Status). Her er en kort oppsumering av hva som er i zip filen du har fått:

**aq-norway-lastx.py**	Dette er den nye Lambda funksjonen (og er egentlig siste versjon av koden så sjekk at det som er i det frittstående getlastXHours.py er lik)

**getLastXHours.py**	Første versjon av koden (som var et frittstående python program)

**lasX.py**				"Bibliotek" med funskjoner som brukes i aq-norway-lastx.py (ble laget et "Lambda lag" med innholdet) og i getLastXHours.py. Her må du legge inn din egen CLIENT_ID (for MET tror jeg, ser at den er brukt både for MET og NEA men tror at for NEA så er det en copy/paste feil)

**README.py**			Denne filen :-)

**Preliminary Project report.docx**	En kort rapport med info som kan være nyttig


# Usage and status
**Note that the latest version is now in the ../lambda/src/aq-norway-lastx.py**

Code that will retrieve last X hours of data from all needed NEA, PRA and MET stations using their respective API's (which stations to retrieve data from is embedded in the inputted *column_list_file*). The result of running this code is a file containing input data to the prediction model in the exact same format as the training data of the model. 

```
python getLastXHours.py --column_list_file train_data_columns.pkl --hours 48
```
The train_data_columns.pkl file is a metadata file that must be created as part of the training code like this:

```
# df holds the training data before feature engineering and normalisation.
# The output file will be named according to scheme: NEA.stationId_lastX.txt
# where the stationId is found in the input .pkl file and X is the number of hours

import pickle
with open('NEA.Elgeseter_train_data_columns.pkl', 'wb') as f:
    pickle.dump(df.columns, f)
```

This metadata file lists all column names in the raw training data (before feature engineering and normalisation). Based on this file it is possible to collect lastX data from all NEA, PRA and MET stations used in the training. After collection the data must be flattened and stuffed into a dataframe with the exact same columns used for training.

All code needed is now present but it feels a bit "fragile". The end product of this code is lastX input in the format expected by the prediction model.

# Background
For RNN LSTM prediction models that we have created (eventually for all NEA stations in Norway) it is necessary to feed the model with the last X hours of data that exactly matches the features the model was trained on. We therefore need to be able to generate such data. This is in fact more complicated than it sounds for several reasons. Here are some of the reasons:

1. For each NEA station the relevant stations used from PRA and MET will be different when the combined dataframe is generated (this dataframe is the basis for the training data), thus the MET and PRA API calls will need to have different parameters (reflecting the stations dataset) when collecting lastX data
2. Stations (NEA, PRA and MET) come and go (and the total dataset generated for a NEA station contains a varying number of MET and PRA stations)
3. Feature engineering can generate syntetic features (and may vary between models). This is code, and how to save it for reuse is TBD.
4. Models can be trained to give predictions for different lenghts (e.g. 48 hours, a week etc.)

## Metadata (what do we have that can guide the generation of last X)
When we generate the basis for the training we combine data from NEA, PRA, MET (and Kystverket). We have specified a scheme for naming columns in the combined dataframe and we also generate meta data files when the combination is done.
The naming schema for columns is:
***“PRO.” + “StationID.” + “MeasurementType” + “:observations_level_value”(optional)***
where PRO identifies provider: e.g: NEA, MET or PRA
The metadata file generated has the follwoing syntax:
```json
{
  ‘NEA_station_id’: {
	‘MET.SN68125:0.sum_precipitation_amount_PT1H’: {
		‘Station_type’: ‘MET’,
		‘Original_element_name’:  ‘what you find in the frost API’,
		‘Distance_to_nea_station_km’: 13.7,
		‘Station_id’: ‘1287412874’,
		‘Rank’: 1,
	},
	‘MET.SN12345:0.sum_precipitation_amount_PT1H’: {
		‘Station_type’: ‘MET’,
		‘Original_element_name’:  ‘what you find in the frost API’,
		‘Distance_to_nea_station_km’: 15.7,
		‘Station_id’: ‘11231241412874’,
		‘Rank’: 2,
	},
	‘PRA.23875fhn10.2.up_to_5_6’ {
		‘Station_type’: ‘PRA’,
		‘Original_element_name’: ‘up_to_5_6’,
		‘Distance’: 5.2,
		‘Station_ID’: ‘23875fhn10’,
		‘Rank’: 2,
	}
    }
}
```
Note that MET and PRA names in the meta data file follows the naming schema for columns.

**This meta data is however not enough since we still do not know the exact layout of the training data. We will therefore have to create additional metadata reflecting the training data format (before feature engineering). It will be sufficient to save the column names of the training data dataframe since this will give us all stations for all providers that have been used in the training. This information can then later be used both to retrieve the relevant data for all stations included and to sort the columns correctly (and exactly in the same order as for the training data (before feature engineering)).**

# How to generate last X dataframe that matches training data

## Retrieve data
As a first step to generate the last X data it is necessary to retrieve (collect) the raw data (in JSON format) using the NEA, MET and PRA API´s for all the needed stations. This will be similar to what we did when collecting the entire AQ dataset but now only for the relevant stations. **Note that current models does not use the Kystverket data**. In order to do this we need to identify the NEA station and all relevant MET and PRA stations since this is necessary input for the API calls. **The list of relevant stations can be retrieved from the training data column names list (see above in section Metadata).** We will not vary the element list for MET, we will just use the "maximum" element list: 
```
"air_temperature", "surface_air_pressure", "wind_speed", "wind_from_direction", "relative_humidity", "specific_humidity" "road_water_film_thickness", "sum(duration_of_precipitation PT1H)", "sum(precipitation_amount PT1H)", "cloud_area_fraction", "surface_snow_thickness", "sea_surface_temperature", "volume_fraction_of_water_in_soil"
```

## Flatten data
When the raw data from the relevant stations have been collected it is necessary to flatten this data. This will also be similar to what we have done in general for the AQ data set.

## Combine data into last X
After the retrieval and flattening of the NEA station data and the relevant MET and PRA station data we need to combine the flattened data in the exact same way as it was done for the training data. This basically means to create a dataframe with the exact same columns (i.e. and named exacly equal) as the training data. We can use the training data column names list to do this. Iterate over the training data column list and get the associated columns from the flattened dataframes (for the provider in question) in the same order as it was in the training data. 

## Impute, feature engineering and normalisation
In order to make lastX data exactly match the training data it is also necessary to do the same imputation, feature engineering and normalisation that was done on the training data. This part is probably a bit tricky to do in a generic way since the code to do this is specific to each NEA station (since the underlying set of MET and PRA stations used will vary and hence the column names the code works on will vary). Should we try to solve it in a generic way by saving the code used as metadata for each station?

Whichever way this is done (generic or manually) it should end up with a lastX set of values in the exact same format as was used when training.

## Transform the created dataframe to prediction input format
When the final dataframe is ready it needs to be transformed into the format expected by the prediction model.
