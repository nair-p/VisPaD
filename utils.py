'''
Author: Pratheeksha Nair
'''
import pandas as pd
import numpy as np
import ast
import json
import plotly.graph_objects as go
from sklearn.linear_model import ElasticNet, LinearRegression


def get_feature_weights(X, Y):
	'''
	This functions returns the features that get more weightage according to the selected points. 
	i.e, if certain points were visually selected due to lying close to each other in a low-dim space, 
	we want to identify which features in the higher-dimensions may be responsible for that. 
	For this, we employ a Logistic Regression classifier to choose which features it gives importance to
	while learning which are selected and which are not
	'''
	reg = LinearRegression().fit(X, Y)
	feature_weights = reg.coef_
	feature_weight_inds = np.argsort(feature_weights)[::-1][:4] # top 4 out of 9 important features 
	# print(feature_weight_inds)
	return feature_weight_inds
	


def get_summary(df):
	'''
	This function returns the summary of the micro-cluster. It returns:
	 - Number of ads
	 - Number of phones numbers
	 - Number of Img URLs
	 - Number of Names
	 - Span of Days
	 - Number of Locations
	'''
	summary_stats = {}
	df['post_date'] = pd.to_datetime(df.post_date)
	df = df.sort_values(by='post_date')
	# df['name'] = df['name'].apply(lambda x: ast.literal_eval(x))
	summary_stats['num_ads'] = [len(df)]
	summary_stats['num_phone'] = [df.phone_num.count(), df.phone_num.nunique()]
	summary_stats['img_urls'] = [df.img_urls.count(), df.img_urls.nunique()]
	summary_stats['num_names'] = [df.name.count(), df.name.nunique()]
	summary_stats['num_loc'] = [df.location.count(), df.location.nunique()]
	summary_stats['num_days'] = [df.post_date.count(), df.post_date.nunique()]

	return summary_stats


def plotlyfromjson(fpath):
	"""Render a plotly figure from a json file"""
	with open(fpath, 'r', encoding='utf-8') as f:
		v = json.loads(f.read())

	fig = go.Figure(data=v['data'], layout=v['layout'])
	return fig


def get_location_time_info(df, geoloc_info):
	'''
	count ranges:
	0-99 : Category 1
	100-499: Category 2
	500-999: Category 3
	1000-5000: Cateogry 4
	> 5000: Category 5

	'''

	ads_per_location_time_df = pd.DataFrame(columns=['date','lat','lon','count','location','meta_label'])
	lats = []
	lons = []
	counts = []
	plot_counts = []
	locations = []
	plot_txt = []
	dates = []
	meta_ids = []

	for i, cluster in df.groupby("Meta label"):

		for id, grp in cluster.groupby("year-month"):
		    cities = grp['cleaned_loc'].unique()
		    for city in cities:
		        dates.append(grp['year-month'].values[0])
		        geo = geoloc_info.loc[city]
		        if city == 'Victoria':
		            lats.append(geo['lat'][0])
		            lons.append(geo['lng'][0])
		        else:
		            lats.append(geo['lat'])
		            lons.append(geo['lng'])
		        cnt = grp[grp.cleaned_loc==city].ad_id.count()
		        counts.append(cnt)
		        if cnt <= 99:
		        	plot_counts.append(5000)
		        elif cnt <= 499:
		        	plot_counts.append(10000)
		        elif cnt <= 999:
		        	plot_counts.append(20000)
		        elif cnt <= 4999:
		        	plot_counts.append(50000)
		        else:
		        	plot_counts.append(90000)
		        locations.append(city)
		        plot_txt.append(city)
		
		        meta_ids.append(i)
	                
	ads_per_location_time_df['lat'] = lats
	ads_per_location_time_df['lon'] = lons
	ads_per_location_time_df['count'] = counts
	ads_per_location_time_df['location'] = locations
	ads_per_location_time_df['plot_text'] = plot_txt
	ads_per_location_time_df['date'] = dates

	ads_per_location_time_df['plot_counts'] = plot_counts
	ads_per_location_time_df['Meta Cluster ID'] = meta_ids
	ads_per_location_time_df.sort_values(by='date',inplace=True)

	return ads_per_location_time_df