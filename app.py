'''
Author: Pratheeksha Nair
'''
import dash
from dash import dcc
from dash import html, callback_context
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
from sklearn.cluster import KMeans
import hdbscan
import pickle as pkl
import sys
import visdcc
import dash_daq as daq
import numpy as np
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

from process_data import get_summary, plotlyfromjson, get_location_time_info, get_feature_weights



# list of clustering methods available
available_indicators = [
				'ICA (2 components)',
				'TSNE',
				'UMAP',
				]


dimension_cols = ["Cluster Size", "Phone Count", "Loc Count", "Phone Entropy", "Loc Radius",\
			"Person Name Count",\
			"Valid URLs", "Invalid URLs", "Ads/week"]

hover_cols = ['Cluster Size Val','Phone Count Val', 'Loc Count Val', 'Loc Radius Val', \
			 'Loc Radius Val','Person Name Count Val','Valid URLs Val', 'Invalid URLs Val',\
			 'Ads/week Val', 'Num URLs Val', 'cluster_id']

# read in the data files for plotting
# full_df = pd.read_csv("data/annoncexxx_name_infoshield.zip",index_col=False) # file name containing the cluster characteristics. 'plot_df.csv' from analyze_clusters.ipynb
full_df = pd.read_csv("data/annoncexxx_filtered_infoshield.zip",index_col=False) # file name containing the cluster characteristics. 'plot_df.csv' from analyze_clusters.ipynb
plot_df = pd.read_csv("data/filtered_df.csv",index_col=False)
plot_df.set_index('cluster_id', drop=False, inplace=True)


micro_to_meta = full_df[['LSH label', 'Meta label']].set_index('LSH label').to_dict()['Meta label']


plot_df['Meta Cluster ID'] = plot_df['cluster_id'].apply(lambda x: micro_to_meta[x])

# DO THE SAME AS ABOVE FOR WEAK LABELS AND TRUE LABELS

# full_df = full_df[full_df['LSH label'] != -1]
top_clusters = full_df.groupby('LSH label').size().sort_values()[-10:].index.values
largest_clusters = full_df[full_df['LSH label'].isin(top_clusters)]
# largest_clusters = full_df[full_df['LSH label'].isin(range(50))]


app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])


data_contents = html.Div([
	html.Div([
		html.H3(children='Ads posted over time'),
		dcc.Graph(
			id='ads_over_time',
			hoverData={'points': [{'customdata': 'Japan'}]}
		)
	], style={'width': '32%', 'float':'left','display': 'inline-block', 'padding': '0 20'}),
	html.Div([
		dbc.DropdownMenu(
		    label="Meta data-type", 
		    size='lg',
		    children=[
		        dbc.DropdownMenuItem("Cluster Size", id='Cluster Size'),
		        dbc.DropdownMenuItem("Phone Number", id='Phone Number'),
		        dbc.DropdownMenuItem("Img URL", id='Img URL'),
		        dbc.DropdownMenuItem("Name", id='Name'),
		    ], style={'float':'right','margin-top':'5px'}
		),
		html.H3(children='Meta-data over time'),
		dcc.Graph(id='metadata_time'),
	], style={'display': 'inline-block', 'width': '35%','float':'center'}),
	html.Div([
		html.H3(children='Geographical Spread of ads'),
		dcc.Graph(id='geo_plot')], style={'width': '30%', 'float':'right'}),
	html.Div([
		html.H3(children='Ad descriptions'),
		dcc.Textarea(id='text_box', readOnly=True,
			style={'height':350,'width':'100%'})],style={'width':'100%', 'float':'left'})
])

mini_dists = [0, 0.01, 0.05, 0.1, 0.5, 1]
char_contents = html.Div([
	html.Div(children=[
			html.H3(children='Feature embeddings'),

			dcc.Dropdown( # dropdown menu for choosing clustering method/ UMAP/ TSNE
				id='analysis-type',
				options=[{'label': ind, 'value': i} for i, ind in enumerate(available_indicators)],
				value=2
			),
			html.Br(),
			html.Div(
			    [
			        dbc.Button("Meta-Cluster Labels", id='b1', size='lg', color="primary", disabled=False, className="me-1"),
			        dbc.Button("Weak Labels", id='b2', size='lg', color="secondary", disabled=True, className="me-1"),
			        dbc.Button("True Labels", id='b3', size='lg', color="dark", disabled=True, className="me-1"),
			        dbc.Tooltip("Clusters with shared meta data",placement='bottom', target='b1', style={'fontSize':20}),
			        dbc.Tooltip("Weak labels inferred from the data",placement='bottom', target='b2', style={'fontSize':20}),
			        dbc.Tooltip("True M.O labels of clusters",placement='bottom', target='b3', style={'fontSize':20})
			    ], style={'float':'right'}),
			
			html.Br(), 
			html.Br(),

			dcc.Graph( # scatter plot showing the micro-clusters using method chosen above
			id='micro-cluster-scatter',
			responsive=True,style={'height':'40vh'}
			),

			html.Div(id='slider-comp', children=[ # Create slides to hide/show for UMAP
        	dcc.Slider(
		    id='slider',
            min=0,  max=5,
            value=0, step=1,
            # marks={i: '{}'.format(10 ** i) for i in mini_dists},
            marks={str(i): {'style':{'fontSize':17},'label':str(mini_dists[i])} for i in range(len(mini_dists))},
            updatemode='drag'
            # tooltip={"placement": "bottom", "always_visible": True}
            ),
        	html.Div(id='slider-name',children=['Minimum distance'],style={'textAlign':'center', 'fontSize':20})
            ], style= {'display': 'none'} # <-- This is the line that will be changed by the dropdown callback
		    ),

			dcc.Graph( # only for UMAP and TSNE
			id='enlarged-graph',
			responsive=True, style={'display':'none'}
			)
			], style={'width': '44%', 'display': 'inline-block', 'margin-top': '25px','margin-left':'-10px'}),

	html.Div(children=[ # pair-wise scatter plot 
			html.H3(children='Cluster Characterization'),
			# html.Div(children="InfoShield Clusters"),
			dcc.Dropdown( # dropdown menu for choosing clustering method/ UMAP/ TSNE
				id='feature-type', 
				options=[{'label': i, 'value': i} for i in dimension_cols],
				value=None,
				multi=True
			),
			dbc.Row(children=[
				dbc.Col(dcc.RadioItems(
				id='scale-type',
				options=[{'label': i, 'value': i} for i in ['Log', 'Linear']],
				value='Log',
				labelStyle={'display': 'inline-block', 'marginTop': '6px'},
				inputStyle={'margin-right':'5px', 'margin-left':'5px', 'margin-top':'10px'}
				)),

				dbc.Col([
				dbc.Button(children="Show Individual", id='hist', size='lg', color="primary", active=True, disabled=False, \
					className="me-1", style={'display':'inline-block',\
				'margin-top':'15px', 'margin-left':'270px'}),\
				# dbc.Tooltip("Show/hide relevant features as decided by an ML classifier",placement='bottom',autohide=False, \
			    # target='feats'),
			    dbc.Tooltip("Show/hide individual feature distribution",placement='bottom',autohide=False, \
			    target='hist', style={'fontSize':20}),
				html.Div([html.Div(id='toggle-switch',children=['Show all points']),
			    daq.ToggleSwitch(
			        id='my-toggle-switch',
			        value=False
			    ), dbc.Tooltip("Show/hide unselected points",placement='bottom',autohide=False, \
			    target='my-toggle-switch')], \
			    style={'float':'right','display':'inline-block'})])
				]
			),
			dcc.Graph(
				id='hist-plot', responsive=True, style={'display':'inline-block'}),
			dcc.Graph(
				id='main-plot', responsive=False
			, style={'display':'inline-block'}),
			],style={'width': '55%', 'float': 'right', 'margin-top': '25px'}),

	], style={"height":"100vh"})

# layout of the app
app.layout = dbc.Container(
	[
		# dcc.Store(id="store"),
		html.Div([
		html.H2("VisPaD: Tool for Visualization and Pattern Discovery"),

		dbc.Row(children=
				[
				dbc.Col(html.Div(id="meta_clusters_n")),
				dbc.Col(html.Div(id="micro_clusters_n")),
                dbc.Col(html.Div(id="ads_n")),
                dbc.Col(html.Div(id="phone_n")),
                dbc.Col(html.Div(id="img_url_n")),
                dbc.Col(html.Div(id="location_n")),
                dbc.Col(html.Div(id="name_n"))
				], align='center', style={'margin-top':'10px', 'fontSize':25}
			),
		dbc.Row(children=
				[
				dbc.Col(html.Div(id="meta_clusters")),
				dbc.Col(html.Div(id="micro_clusters")),
                dbc.Col(html.Div(id="ads")),
                dbc.Col(html.Div(id="phone")),
                dbc.Col(html.Div(id="img_url")),
                dbc.Col(html.Div(id="location")),
                dbc.Col(html.Div(id="name")),
				], align='center'
				, style={'margin-top':'-2px', 'fontSize':20}
			)], style={'textAlign': 'center','backgroundColor': 'blue', \
			'color':'white','height':'120px'}
		),		
		# html.Div([ # top row header
		html.Br(),
		# ]),
		dbc.Tabs(
			[
				dbc.Tab(data_contents, label="Inspect Clusters", label_style={'fontSize':20,}, tab_id="data", style={'fontSize':20}),
				dbc.Tab(char_contents, label="Analysis", label_style={'fontSize':20}, tab_id="scatter", style={'fontSize':20})
			],
			id="tabs",
			active_tab="data"
		),
		visdcc.Run_js(id = 'javascriptLog', run = ""),
		# html.Div(id="tab-content", className="p-4"),
	], fluid=True, style={'width':'100%'})


'''
=============================SUMMARY OF CHOSEN CLUSTERS======================================
'''
@app.callback(
Output('meta_clusters_n', 'children'),
Output('micro_clusters_n', 'children'),
Output('ads_n', 'children'),
Output('phone_n', 'children'),
Output('img_url_n', 'children'),
Output('location_n', 'children'),
Output('name_n', 'children'),
Input('tabs','active_tab')) # input: currently active tab
def update_summary_heads(active_tab):
	return "Meta-clusters", "Micro-clusters", "Ads", "Phone Numbers", "Img URLs", "Locations", "Names"



@app.callback(
Output('meta_clusters', 'children'),
Output('micro_clusters', 'children'),
Output('ads', 'children'),
Output('phone', 'children'),
Output('img_url', 'children'),
Output('location', 'children'),
Output('name', 'children'),
Input('enlarged-graph', 'selectedData'), # input: selected points from the enlarged scatter-plot (TSNE/UMAP)
Input('main-plot', 'selectedData'), # input: selected points from the pair-plots
Input('micro-cluster-scatter', 'selectedData')) # input: selected points from the feature embedding plot
def update_summary(selectedData, selected_from_pair_plots, selected_from_ica):

	if selectedData:
		selected_points = [] # the selection info is in JSON format and we need to extract the index of points selected
		for item in selectedData['points']:
			selected_points.append(item['customdata'][-1])

		selected_df = full_df[full_df['LSH label'].isin(selected_points)]
	
	elif selected_from_pair_plots:  
		selected_points = [] # the selection info is in JSON format and we need to extract the index of points selected
		for item in selected_from_pair_plots['points']:
			selected_points.append(item['customdata'][-1])

		selected_df = full_df[full_df['LSH label'].isin(selected_points)]
	
	elif selected_from_ica:
		selected_points = []
		for item in selected_from_ica['points']:
			selected_points.append(item['customdata'][-1])
		selected_df = full_df[full_df['LSH label'].isin(selected_points)]
	else:
		selected_df = largest_clusters


	num_meta = selected_df['Meta label'].nunique()
	num_clusters = selected_df['LSH label'].nunique()
	num_ads = len(selected_df)
	num_names = selected_df.name.count()
	num_unique_names = selected_df.phone_num.nunique()
	num_imgs = selected_df.img_urls.count()
	num_unique_imgs = selected_df.img_urls.nunique()
	num_phones = selected_df.phone_num.count()
	num_unique_phones = selected_df.phone_num.nunique()
	num_locs = selected_df.cleaned_loc.count()
	num_unique_locs = selected_df.cleaned_loc.nunique()


	return str(num_meta), str(num_clusters), str(num_ads), str(num_phones)+" ("+str(num_unique_phones)+")", \
	str(num_imgs)+" ("+str(num_unique_imgs)+")", str(num_locs)+" ("+str(num_unique_locs)+")", \
	str(num_names)+" ("+str(num_unique_names)+")"



'''
=============================ADS POSTED OVER TIME======================================
'''
@app.callback(
Output('ads_over_time', 'figure'), # output: ads poster over time
Input('tabs','active_tab'), # input: currently active tab
Input('enlarged-graph', 'selectedData'), # input: selected points from the enlarged scatter-plot (TSNE/UMAP)
Input('main-plot', 'selectedData'), # input: selected points from the pair-plots
Input('micro-cluster-scatter', 'selectedData'))
def update_ads_over_time(active_tab, selectedData, selected_from_pair_plots, selected_from_ica):
	if active_tab != 'data':
		return px.scatter([])
	
	if selectedData:
		selected_points = [] # the selection info is in JSON format and we need to extract the index of points selected
		for item in selectedData['points']:
			selected_points.append(item['customdata'][-1])

		selected_df = full_df[full_df['LSH label'].isin(selected_points)]
		# selected_df = full_df.loc[selected_points]
	
	elif selected_from_pair_plots:  
		selected_points = [] # the selection info is in JSON format and we need to extract the index of points selected
		for item in selected_from_pair_plots['points']:
			selected_points.append(item['customdata'][-1])

		selected_df = full_df[full_df['LSH label'].isin(selected_points)]
		# selected_df = full_df.loc[selected_points]
	
	elif selected_from_ica:
		selected_points = []
		for item in selected_from_ica['points']:
			selected_points.append(item['customdata'][-1])
		selected_df = full_df[full_df['LSH label'].isin(selected_points)]
		# selected_df = full_df.loc[selected_points]

	else:
		selected_df = largest_clusters

	top_clusters = selected_df.groupby('LSH label').size().sort_values(ascending=False)[:10].index.values

	selected_df = selected_df[selected_df['LSH label'].isin(top_clusters)]
	
	# we are trying to plot the number of ads posted in a particular date for each selected micro-cluster
	ads_timeline = pd.DataFrame(columns=['cluster','start_date','end_date', 'Count', 'Meta Cluster ID'])
	clusters = []
	starts = []
	ends = []
	counts = []
	meta_ids = []


	for meta_id, micro_cluster in selected_df.groupby('Meta label'):
		for grp in micro_cluster.groupby('LSH label'):
			clusters.append(str(grp[0]))
			micro_df = grp[1]
			micro_df['post_date'] = pd.to_datetime(micro_df['post_date'], infer_datetime_format=True)
			micro_df = micro_df.sort_values(by='post_date')

			starts.append(micro_df.iloc[0].post_date)
			ends.append(micro_df.iloc[-1].post_date)
			counts.append(len(micro_df))
			meta_ids.append(meta_id)

	ads_timeline['cluster'] = clusters
	ads_timeline['start_date'] = starts
	ads_timeline['end_date'] = ends
	ads_timeline['Count'] = counts
	ads_timeline['Meta Cluster ID'] = meta_ids

	ads_timeline = ads_timeline.sort_values(by='Count')[-10:]

	ads_timeline = ads_timeline.sort_values(by='start_date')
	# ads_timeline['start_date'] = ads_timeline['start_date'].apply(lambda x: x.strftime("%Y-%m-%d"))
	# ads_timeline['end_date'] = ads_timeline['end_date'].apply(lambda x: x.strftime("%Y-%m-%d"))
	ads_timeline['Meta Cluster ID'] = 'M'+ ads_timeline['Meta Cluster ID'].astype(str)
	ads_timeline['cluster'] = 'C' + ads_timeline['cluster'].astype(str)

	fig = px.timeline(ads_timeline, x_start='start_date', x_end='end_date', y='cluster', color='Meta Cluster ID', \
		color_discrete_sequence=px.colors.qualitative.Vivid, hover_data=ads_timeline.columns)
	fig.update_yaxes(autorange='reversed')

	fig.update_layout(
		font_size=20,
		xaxis = dict(
			tickmode='array',
			ticktext=ads_timeline['start_date'],
			title='Post Date'
		),
		yaxis = dict(
			tickmode='array',
			ticktext=ads_timeline['cluster'],
			title='Cluster ID'
		),
		)
	return fig


'''
=============================META DATA OVER TIME======================================
'''
@app.callback(
Output('metadata_time', 'figure'), # output: ads poster over time
Input('tabs','active_tab'), # input: currently active tab
Input('enlarged-graph', 'selectedData'), # input: selected points from the enlarged scatter-plot (TSNE/UMAP)
Input('main-plot', 'selectedData'), # input: selected points from the pair-plots
Input('micro-cluster-scatter', 'selectedData'),
Input('Cluster Size', 'n_clicks'),
Input('Phone Number', 'n_clicks'),
Input('Img URL', 'n_clicks'),
Input('Name', 'n_clicks')) 
def update_meta_data(active_tab, selectedData, selected_from_pair_plots, selected_from_ica, \
	clus_size, phone_num, img_url, name):
	if active_tab != 'data':
		return px.scatter([])
	else:
		if selectedData:
			selected_points = [] # the selection info is in JSON format and we need to extract the index of points selected
			for item in selectedData['points']:
				selected_points.append(item['customdata'][-1])

			# selected_df = full_df.loc[selected_points]
			selected_df = full_df[full_df['LSH label'].isin(selected_points)]
		
		elif selected_from_pair_plots:  
			selected_points = [] # the selection info is in JSON format and we need to extract the index of points selected
			for item in selected_from_pair_plots['points']:
				selected_points.append(item['customdata'][-1])

			# selected_df = full_df.loc[selected_points]
			selected_df = full_df[full_df['LSH label'].isin(selected_points)]			
		
		elif selected_from_ica:
			selected_points = []
			for item in selected_from_ica['points']:
				selected_points.append(item['customdata'][-1])
			# selected_df = full_df.loc[selected_points]
			selected_df = full_df[full_df['LSH label'].isin(selected_points)]

		else:
			selected_df = largest_clusters


		top_clusters = selected_df.groupby('LSH label').size().sort_values(ascending=False)[:10].index.values

		selected_df = selected_df[selected_df['LSH label'].isin(top_clusters)]


		selected_df['post_date'] = pd.to_datetime(selected_df['post_date'], infer_datetime_format=True)
		selected_df = selected_df.sort_values(by='post_date')


		dates = []
		counts = []
		meta_data_types = []
		micro_cluster_ids = []
		meta_cluster_ids = []

		for meta_id, meta_df in selected_df.groupby('Meta label'):
			for micro_id, micro_df in meta_df.groupby('LSH label'):
				for grp in micro_df.groupby('post_date'):

					# for phone numbers
					dates.append(grp[0].strftime('%Y-%m-%d'))
					counts.append(grp[1].phone_num.nunique())
					micro_cluster_ids.append(str(micro_id))
					meta_cluster_ids.append(str(meta_id))
					meta_data_types.append("Phone Number")

					# for img urls
					dates.append(grp[0].strftime('%Y-%m-%d'))
					counts.append(grp[1].img_urls.nunique())
					micro_cluster_ids.append(str(micro_id))
					meta_cluster_ids.append(str(meta_id))
					meta_data_types.append("Img URL")

					# for names
					dates.append(grp[0].strftime('%Y-%m-%d'))
					counts.append(grp[1].name.nunique())
					micro_cluster_ids.append(str(micro_id))
					meta_cluster_ids.append(str(meta_id))
					meta_data_types.append("Name")

					# for cluster sizes
					dates.append(grp[0].strftime('%Y-%m-%d'))
					counts.append(len(micro_df))
					micro_cluster_ids.append(str(micro_id))
					meta_cluster_ids.append(str(meta_id))
					meta_data_types.append("Cluster Size")

		meta_data = pd.DataFrame({'Post Date':dates, 'Count':counts, 'Type':meta_data_types, \
		 'Micro Cluster ID': micro_cluster_ids, 'Meta Cluster ID': meta_cluster_ids})

		# based on button selection
		ctx = dash.callback_context

		if not ctx.triggered:
			chosen_type = "Cluster Size"

		else:
			chosen_type = ctx.triggered[0]["prop_id"].split(".")[0]
			
		if chosen_type == 'tabs':
			chosen_type = 'Cluster Size'
		plot_data = meta_data[meta_data.Type==chosen_type]

		plot_data.drop_duplicates(inplace=True)
		plot_data = plot_data.sort_values(by='Count')[::-1]
		plot_data = plot_data.sort_values(by='Post Date')
		plot_data['Micro Cluster ID'] = 'C'+plot_data['Micro Cluster ID'].astype(str)
		plot_data['Meta Cluster ID'] = 'M'+plot_data['Meta Cluster ID'].astype(str)


		try:
			fig = px.scatter(plot_data, x='Post Date', y='Micro Cluster ID', \
				color='Meta Cluster ID', size='Count', color_discrete_sequence=px.colors.qualitative.Vivid)
		except Exception:
			fig = px.scatter(plot_data, x='Post Date', y='Micro Cluster ID', \
				color='Meta Cluster ID', size='Count', color_discrete_sequence=px.colors.qualitative.Vivid)
		
		fig.update_layout(
			font_size=20,
			yaxis = dict(
				tickmode='array',
				ticktext=plot_data['Micro Cluster ID'],
				title='Cluster ID'
		), title_text=chosen_type)

		return fig



'''
=============================GEO DATA======================================
'''
@app.callback(
Output('geo_plot', 'figure'), # output: ads poster over time
Input('tabs','active_tab'), # input: currently active tab
Input('enlarged-graph', 'selectedData'), # input: selected points from the enlarged scatter-plot (TSNE/UMAP)
Input('main-plot', 'selectedData'), # input: selected points from the pair-plots
Input('micro-cluster-scatter', 'selectedData'))
def update_meta_data(active_tab, selectedData, selected_from_pair_plots, selected_from_ica):
	if active_tab != 'data':
		return px.scatter([])
	else:
		cities_df = pd.read_csv("data/geoloc_info.csv",index_col='city_ascii')
		if selectedData:
			selected_points = [] # the selection info is in JSON format and we need to extract the index of points selected
			for item in selectedData['points']:
				selected_points.append(item['customdata'][-1])

			# selected_df = full_df.loc[selected_points]
			selected_df = full_df[full_df['LSH label'].isin(selected_points)]
		
		elif selected_from_pair_plots:  
			selected_points = [] # the selection info is in JSON format and we need to extract the index of points selected
			for item in selected_from_pair_plots['points']:
				selected_points.append(item['customdata'][-1])

			# selected_df = full_df.loc[selected_points]
			selected_df = full_df[full_df['LSH label'].isin(selected_points)]
		
		elif selected_from_ica:
			selected_points = []
			for item in selected_from_ica['points']:
				selected_points.append(item['customdata'][-1])

			# selected_df = full_df.loc[selected_points]
			selected_df = full_df[full_df['LSH label'].isin(selected_points)]

		else:
			selected_df = largest_clusters

		top_clusters = selected_df.groupby('LSH label').size().sort_values()[-10:].index.values
		selected_df = selected_df[selected_df['LSH label'].isin(top_clusters)]
		
		selected_df = pd.merge(selected_df, cities_df, left_on='cleaned_loc', right_on='city_ascii', left_index=True, \
			how='left', sort=False)

		selected_df['post_date'] = pd.to_datetime(selected_df.post_date, format='%Y-%m-%d')
		selected_df['year-month'] = selected_df['post_date'].apply(lambda x: str(x.year) + ' ' + str(x.month))
		

		ads_per_location_time_df = get_location_time_info(selected_df, cities_df)
		ads_per_location_time_df['Meta Cluster ID'] = 'M'+ads_per_location_time_df['Meta Cluster ID'].astype('str')
		ads_per_location_time_df.sort_values(by='date',inplace=True)
		# print(ads_per_location_time_df[ads_per_location_time_df['Meta Cluster ID'].isin(['0','1','2','5'])])
		# ads_per_location_time_df = ads_per_location_time_df[ads_per_location_time_df['Meta Cluster ID'].isin(['0','1','2','5'])]

		try:
			fig = px.scatter_geo(ads_per_location_time_df, lat='lat',lon='lon', 
                     hover_name="plot_text", size='plot_counts', hover_data={'date':False,'plot_counts':False, \
                     'count':True},
                     animation_frame="date", scope='north america', \
                     color_discrete_sequence=px.colors.qualitative.Vivid, color='Meta Cluster ID')
		except Exception:
			fig = px.scatter_geo(ads_per_location_time_df, lat='lat',lon='lon', 
                     hover_name="plot_text", size='plot_counts', hover_data={'date':False,'plot_counts':False, \
                     'count':True},
                     animation_frame="date", scope='north america', color='Meta Cluster ID', \
                     color_discrete_sequence=px.colors.qualitative.Vivid)

		fig.update_layout(font_size=20)
		return fig


'''
=============================MICRO-CLUSTER AD DESCRIPTIONS======================================
'''
@app.callback(
Output('text_box', 'value'), # output: text box for ad descriptions
Output('javascriptLog','run'), # output: scroll bar for text box
Input('tabs','active_tab'), # input: currently active tab
Input('enlarged-graph','selectedData'), # input: selected points from the enlarged scatter-plot (TSNE/UMAP)
Input('main-plot', 'selectedData')) # input: selected points from the pair-plots
def update_ad_text(active_tab, selectedData, selected_from_pair_plots):
	# print the ad text in the data tab
	ad_text = "var textarea = document.getElementById('text_box'); textarea.scrollTop = textarea.scrollHeight;"
	if active_tab != 'data':
		txts = ""

	if selectedData:
		selected_points = [] # the selection info is in JSON format and we need to extract the index of points selected
		for item in selectedData['points']:
			selected_points.append(item['customdata'][-1])

	elif selected_from_pair_plots:
		selected_points = []
		for item in selected_from_pair_plots['points']:
			selected_points.append(item['customdata'][-1])

	else:
		selected_points = largest_clusters['LSH label'].unique()

	sel_data = full_df[full_df['LSH label'].isin(selected_points)]
	ordered_clusters = sel_data.groupby('LSH label').size().sort_values(ascending=False)


	txts = ""
	# for grp in sel_data.groupby('LSH label'):
	for row in ordered_clusters.index:
		txts += "\n\nCluster : C" + str(row) + "\n"
		grp = sel_data[sel_data['LSH label']==row]
		txts += ("\n".join(d for d in grp.description.values))

	return txts, ad_text


'''
=============================MICRO-CLUSTER FEATURE EMBEDDING======================================
'''
# function for showing the clicked point from pair-plots on the scatter plot
@app.callback( 
Output('micro-cluster-scatter', 'figure'),
Input('analysis-type', 'value'),
Input('main-plot','clickData'),
Input('main-plot','selectedData'),
Input('slider','value'),
Input('b1', 'n_clicks'), # input: if meta cluster label is clicked
Input('b2', 'n_clicks'), # input: if weak label is clicked
Input('b3', 'n_clicks')) # input: if true label is clicked
def update_graph(selected_clustering, clickData, selectedData, mini_dist_ind, b1_click, b2_click, b3_click):
	perp_vals = [5, 10, 20, 30, 40, 50]

	show_color_bar = True
	if b1_click and b1_click%2==1: # if button has been clicked on
		# show meta-cluster labels
		color_label = 'Meta Cluster ID'
		color_dict = micro_to_meta.copy()
	elif b2_click and b2_click%2==1:
		# show weak labels
		color_label = 'Weak Label'
	elif b3_click and b3_click%2==1:
		# show true labels
		color_label = 'MO Label'
	else:
		color_label = 'Color'
		show_color_bar = False
		color_dict = dict(zip(list(micro_to_meta.keys()), ['blue']*len(micro_to_meta)))

	# the ICA, TSNE and UMAP are precomputed and saved to disk for plotting
	if selected_clustering == 0: # ICA 2 comp
		df = pd.read_csv("data/is_ica.zip",index_col=False) #CHANGE THE DATA FILES ACCORDINGLY
		df.set_index('cluster_id',drop=False,inplace=True)
		df[color_label] = 'M'+df['cluster_id'].apply(lambda x: color_dict[x]).astype('str')
		if color_label != 'Color':
			fig = px.scatter(df, x='x',y='y', hover_data=hover_cols, height=1600, \
			color=color_label, color_discrete_sequence=px.colors.qualitative.Vivid)
		else:
			fig = px.scatter(df, x='x',y='y', hover_data=hover_cols, height=1600, \
			color=color_label)
		fig.update_traces(marker=dict(size=3), showlegend=show_color_bar)
		fig.update_layout(dragmode='lasso', font_size=20)

	elif selected_clustering == 1: # TSNE
		# fig = plotlyfromjson("data/tsne_plot.json")
		# fig.update_traces(marker=dict(size=3))
		tsne_res = pkl.load(open("data/all_tsne_res.pkl",'rb')) # Since we are looking at multiple parameter values
		# # df = pd.read_csv("data/is_tsne.zip",index_col=False)
		perp_vals = list(tsne_res.keys())
		titles = []
		for p in perp_vals:
			titles.append(str(p))
		fig = make_subplots(
					rows=1, cols=len(perp_vals),\
					subplot_titles=tuple(titles), \
					horizontal_spacing=0.01, vertical_spacing=0.01, \
					shared_xaxes=True, shared_yaxes=True, \
					# title_text='Perplexity Values'
			)

		template_str = ""
		for i, col in enumerate(hover_cols):
			if i != len(hover_cols)-1:
				template_str += (col+":%{customdata["+str(i)+"]}<br>")
			else:
				template_str += (col+":%{customdata["+str(i)+"]}")

		for i, p in enumerate(perp_vals):
			dd = tsne_res[p]
			fig.add_scatter(x=list(dd.x), y=list(dd.y), \
								customdata=dd[hover_cols], \
								hovertemplate=template_str,\
								# marker_color=dd[color_label], \
								mode='markers', marker={'opacity':0.3, 'color':'blue', 'size':3}, \
								row=1, col=i+1)
				
			if i == 0:
				fig.layout.annotations[i].update(text=str(p))

		fig.update_layout(height=350, width=650, dragmode="lasso", showlegend=False,\
		font_size=20, title_text='Perplexity Values')
		# fig.update_layout(height=1600, width=1000, showlegend=False)

	elif selected_clustering == 2: # UMAP
		umap_res = pkl.load(open("data/umap_res.pkl",'rb')) # with multiple parameter values 
		# df = pd.read_csv("data/is_umap.zip",index_col=False)	
		nbr_sizes = [10, 50, 100, 200, 500, 1000]
		mini_dists = [0, 0.01, 0.05, 0.1, 0.5, 1]

		titles = []
		for d in nbr_sizes:
			titles.append(str(d))

		template_str = ""
		for i, col in enumerate(hover_cols):
			if i != len(hover_cols)-1:
				template_str += (col+":%{customdata["+str(i)+"]}<br>")
			else:
				template_str += (col+":%{customdata["+str(i)+"]}")

		
		fig = make_subplots(rows=1, cols=len(nbr_sizes), \
				subplot_titles=tuple(titles[:6]), \
				# horizontal_spacing=0.02, vertical_spacing=0.05, \
				shared_xaxes=True, shared_yaxes=True, \
				)

		# mini_dist_ind = 2
		for i in range(len(nbr_sizes)):
			dd = umap_res[i][mini_dist_ind]
			fig.add_scatter(x=list(dd.x), y=list(dd.y), \
								customdata=dd[hover_cols], \
								hovertemplate=template_str,\
								# marker_color=dd[color_label],\
								mode='markers', marker={'opacity':0.3, 'color':'blue', 'size':3}, \
								row=1, col=i+1)

		fig.update_layout(title_text="Nbrhood Size",height=350, width=700, showlegend=False, font_size=20)
		

	if clickData:
		if selected_clustering == 0: # if a certain point has been clicked on (only if not TSNE and UMAP)
			cluster_id = clickData['points'][0]['customdata'][-1] # retrieve info of clicked point
			fig.add_traces(
				px.scatter(df[df.cluster_id==cluster_id], \
							  x='x',y='y', hover_data=hover_cols).update_traces(marker_color="red",marker_size=4,marker_symbol='star').data
				)
	
		else:
			cluster_id = clickData['points'][0]['customdata'][-1]
			if selected_clustering == 1: # TSNE
				tsne_res = pkl.load(open("data/all_tsne_res.pkl",'rb'))
				for i, p in enumerate(perp_vals):
					dd = tsne_res[p]
					dd.set_index('cluster_id', drop=False, inplace=True)
					hover_trace = dict(type='scatter', \
						x=list(dd[dd.cluster_id==cluster_id].x), y=list(dd[dd.cluster_id==cluster_id].y), \
						customdata=dd[hover_cols], \
						hovertemplate=template_str,\
						mode='markers', marker={'symbol':'star', 'color': 'red','size':10})
					fig.append_trace(hover_trace, 1, i+1)

			if selected_clustering == 2:
				umap_res = pkl.load(open("data/umap_res.pkl",'rb'))
				nbr_sizes = [10, 50, 100, 200, 500, 1000]
				# mini_dists = [0, 0.01, 0.05, 0.1, 0.5, 1]

				template_str = ""
				for i, col in enumerate(hover_cols):
					if i != len(hover_cols)-1:
						template_str += (col+":%{customdata["+str(i)+"]}<br>")
					else:
						template_str += (col+":%{customdata["+str(i)+"]}")

				for i in range(len(nbr_sizes)):
					dd = umap_res[i][mini_dist_ind]
					dd.set_index('cluster_id', drop=False, inplace=True)
					fig.add_scatter(x=list(dd[dd.cluster_id==cluster_id].x), y=list(dd[dd.cluster_id==cluster_id].y),\
										customdata=dd[hover_cols], \
										hovertemplate=template_str,\
										mode='markers', marker={'symbol':'star', 'color': 'red','size':10}, \
										row=1, col=i+1)

				fig.update_layout(height=350, width=700, showlegend=False, font_size=20)
	

	if selectedData:
		selected_points = [] # the selection info is in JSON format and we need to extract the index of points selected
		for item in selectedData['points']:
			selected_points.append(item['customdata'][-1])
		if selected_clustering == 0: # if a certain point has been clicked on (only if not TSNE and UMAP)
			fig.add_traces(
				px.scatter(df.loc[selected_points], \
								  x='x',y='y', hover_data=hover_cols).update_traces(marker_color="black",marker_size=4,marker_symbol='star').data
			)
	
		else:
			if selected_clustering == 1: # TSNE
				tsne_res = pkl.load(open("data/all_tsne_res.pkl",'rb'))
				for i, p in enumerate(perp_vals):
					dd = tsne_res[p]
					dd.set_index('cluster_id', drop=False, inplace=True)
					hover_trace = dict(type='scatter', \
						x=list(dd.loc[selected_points].x), y=list(dd.loc[selected_points].y), \
						customdata=dd.loc[selected_points][hover_cols], \
						hovertemplate=template_str,\
						mode='markers', marker={'color': 'black','size':3})
					fig.append_trace(hover_trace, 1, i+1)

			if selected_clustering == 2:
				nbr_sizes = [10, 50, 100, 200, 500, 1000]
				umap_res = pkl.load(open("data/umap_res.pkl",'rb'))

				template_str = ""
				for i, col in enumerate(hover_cols):
					if i != len(hover_cols)-1:
						template_str += (col+":%{customdata["+str(i)+"]}<br>")
					else:
						template_str += (col+":%{customdata["+str(i)+"]}")

				for i in range(len(nbr_sizes)):
					dd = umap_res[i][mini_dist_ind]
					dd.set_index('cluster_id', drop=False, inplace=True)
					fig.add_scatter(x=list(dd.loc[selected_points].x), y=list(dd.loc[selected_points].y), \
										customdata=dd[hover_cols], \
										hovertemplate=template_str,\
										mode='markers', marker={'color': 'black','size':3}, \
										row=1, col=i+1)

				fig.update_layout(height=350, width=700, showlegend=False, font_size=20)

	return fig
	


'''
=============================DISPLAYING SLIDER======================================
'''
@app.callback(
Output('slider-comp','style'), # output: visibility of slider
Input('analysis-type','value')) # input: type of vis. Only make visible if UMAP
def display_slider(selected_clustering):
	if selected_clustering == 2:
		return {'display':'block'}
	else:
		return {'display': 'None'}


'''
=============================ENLARGING THE CLICKED SUB PLOT======================================
'''
@app.callback(
Output('enlarged-graph','style'), # output: the display style of the enlarged graph. Basically to unhide it
Output('enlarged-graph','figure'), # output: the enlarged graph
Input('analysis-type','value'), # input: the type of vis - TSNE/UMAP
Input('micro-cluster-scatter','clickData'), # input: selected point from pair-plots for highlighting in red
Input('main-plot','selectedData'), # input: selected data points for highlighting in pair-plots
Input('slider','value'), # input: minimum distance value for UMAP
Input('b1', 'n_clicks'), # input: if meta cluster label is clicked
Input('b2', 'n_clicks'), # input: if weak label is clicked
Input('b3', 'n_clicks')) # input: if true label is clicked
def enlarge_subplot(selected_clustering, clickData, selectedData, mini_dist_ind, b1_click, b2_click, b3_click):

	tsne_res = pkl.load(open("data/all_tsne_res.pkl",'rb'))
	umap_res = pkl.load(open("data/umap_res.pkl",'rb'))

	show_color_bar = True
	if b1_click and b1_click%2==1: # if button has been clicked on
		# show meta-cluster labels
		color_label = 'Meta Cluster ID'
		color_dict = micro_to_meta.copy()
	elif b2_click and b2_click%2==1:
		# show weak labels
		color_label = 'Inferred Label'
	elif b3_click and b3_click%2==1:
		# show true labels
		color_label = 'MO Label'
	else:
		color_label = 'Color'
		show_color_bar = False
		color_dict = dict(zip(list(micro_to_meta.keys()), ['blue']*len(micro_to_meta)))

	if selected_clustering == 0 or not clickData: # not TSNE/UMAP
		dd = tsne_res[10]
		dd[color_label] = 'M'+dd['cluster_id'].apply(lambda x: color_dict[x]).astype(str)
		try:
			fig = px.scatter(dd, x='x',y='y', hover_data=hover_cols, color=color_label)
		except Exception:
			if color_label == 'Color':
				fig = px.scatter(dd, x='x',y='y', hover_data=hover_cols, color=color_label)
			else:
				fig = px.scatter(dd, x='x',y='y', hover_data=hover_cols, color=color_label, color_discrete_sequence=px.colors.qualitative.Vivid)
		fig.update_traces(showlegend=show_color_bar)
		fig.update_layout(font_size=20)
		return {'display':'none'}, fig

	if clickData: # clicking has happened
		if selected_clustering == 1:
			perp_vals = [5, 10, 20, 30, 40, 50]
			perp_index = clickData['points'][0]['curveNumber']
			if perp_index >= len(perp_vals):
				perp_index = perp_index % len(perp_vals) 
			perplexity = perp_vals[perp_index]
			dd = tsne_res[perplexity]
			dd[color_label] = 'M'+dd['cluster_id'].apply(lambda x: color_dict[x]).astype(str)
			title = 'Perplexity:'+str(perplexity)
			
		else:
			nbr_sizes = [10, 50, 100, 200, 500, 1000]
			nbr_size_ind = clickData['points'][0]['curveNumber']
			
			if nbr_size_ind >= len(nbr_sizes):
				nbr_size_ind = nbr_size_ind % 6

			nbr_size = nbr_sizes[nbr_size_ind]
			dd = umap_res[nbr_size_ind][mini_dist_ind]
			dd[color_label] = 'M'+dd['cluster_id'].apply(lambda x: color_dict[x]).astype(str)
			title = 'Nbrhood Size:'+str(nbr_size)
			
		if color_label != 'Color':
			fig = px.scatter(dd, x='x',y='y', hover_data=hover_cols, color=color_label, \
			color_discrete_sequence=px.colors.qualitative.Vivid)
		else:
			fig = px.scatter(dd, x='x',y='y', hover_data=hover_cols, color=color_label)
		fig.update_layout(width=1000, title_text=title, dragmode='lasso', font_size=20)
		fig.update_traces(marker=dict(size=3), showlegend=show_color_bar)

		if selectedData: # some points in main plot have been selected
			selected_points = [] # the selection info is in JSON format and we need to extract the index of points selected
			for item in selectedData['points']:
				selected_points.append(item['customdata'][-1])
			dd.set_index('cluster_id', drop=False, inplace=True)
			fig.add_traces(
				px.scatter(dd.loc[selected_points], \
								  x='x',y='y', hover_data=hover_cols).update_traces(marker_color="cyan",marker_size=4,marker_symbol='star').data
			)
			fig.update_layout(font_size=20)
		return {'display': 'inline-block','margin-left':'40px'}, fig

	

'''
=========================CLUSTER CHARACTERIZATION PLOT==================================== 
'''

# function for highlighting in red the chosen cluster over all pair-plots
@app.callback( 
Output('main-plot','figure'), # output: 'main-plot' figure. Corresponding to app.layout
Input('main-plot','clickData'), # input: 'main-plot' with clicked data point. 
Input('micro-cluster-scatter','selectedData'), # input: 'micro-cluster-scatter' with points selected using lasso-tool
Input('feature-type', 'value'), # input: 'feature-type' selects which feature columns to show in the pair-plot
Input('scale-type', 'value'), # input: log or linear scale
Input('enlarged-graph','selectedData'),
Input('my-toggle-switch','value')) # input: selected data points from enlarged graph
def highlight_same_clusters(clickData, selected_clusters, selected_feats, scale, enlarged_selected, toggle_value):

	# if no features are currently selected, then display all
	if not selected_feats:
		# log or linear scale as decided by the radio button
		if scale == 'Log':
			cols = dimension_cols
			labels = dimension_cols
		else:
			cols = list(set(hover_cols)-{'cluster_id'})
			labels={col:col[:-4] for col in cols}
		selected_feats = cols
	else:
		if scale == 'Linear':
			selected_feats = [feat+" Val" for feat in selected_feats]
			labels = selected_feats
		else:
			labels = selected_feats

	if selected_clusters: # if some selection of points has been made
		selected_points = [] # the selection info is in JSON format and we need to extract the index of points selected
		for item in selected_clusters['points']:
			selected_points.append(item['customdata'][-1])

		if toggle_value: # we want to show all the points but grey out the non-selected ones
			to_plot = plot_df.copy()
		else:
			to_plot = plot_df.loc[selected_points]
			classifier_df = plot_df.copy()
			classifier_df['Y'] = np.zeros(len(plot_df))
			classifier_df['Y'].loc[selected_points] = 1
			X = classifier_df[dimension_cols].to_numpy()
			Y = classifier_df['Y'].values
			important_features = np.array(dimension_cols)[get_feature_weights(X, Y)]

			if len(selected_feats) == len(dimension_cols):
				selected_feats = important_features
			if scale == 'Linear':
				selected_feats = [feat+" Val" for feat in selected_feats]
			labels = selected_feats


	elif enlarged_selected: # if some selections of points has been made from the enlarged plot
		selected_points = []
		for item in enlarged_selected['points']:
			selected_points.append(item['customdata'][-1])

		if toggle_value: # we want to show all the points but grey out the non-selected ones
			to_plot = plot_df.copy()
		else:
			to_plot = plot_df.loc[selected_points]
			# print(full_df.columns)
			classifier_df = plot_df.copy()
			classifier_df['Y'] = np.zeros(len(plot_df))
			classifier_df['Y'].loc[selected_points] = 1
			X = classifier_df[dimension_cols].to_numpy()
			Y = classifier_df['Y'].values
			important_features = np.array(dimension_cols)[get_feature_weights(X, Y)]

			if len(selected_feats) == len(dimension_cols):
				selected_feats = important_features
			if scale == 'Linear':
				selected_feats = [feat+" Val" for feat in selected_feats]
			labels = selected_feats


	else: # if no selection has been made
		to_plot = plot_df.copy()


	if clickData: # if a certain point has been clicked on (we want to track that point across pair-plots)
		cluster_id = clickData['points'][0]['customdata'][-1] # extract info from JSON format

		# pair-plots
		fig = px.scatter_matrix(to_plot, dimensions=selected_feats,opacity=0.6,\
					   hover_data=hover_cols, labels=labels)
		fig.update_traces(marker=dict(size=3))
		if len(selected_feats) > 4:
			fig.layout.font.size = 10
			fig.update_layout(height=2600,width=2600, dragmode='lasso')
		else:
			fig.update_layout(height=500,width=1000, dragmode='lasso', font_size=20)
		# highlight the current clicked point across all pair-plots in red
		fig.add_traces(
		px.scatter_matrix(to_plot[to_plot.cluster_id==cluster_id], \
					  dimensions=selected_feats, labels=labels).update_traces(marker_color="cyan").data
		)
		fig.update_traces(marker=dict(size=3))
		if len(selected_feats) > 4:
			fig.layout.font.size = 10
			fig.update_layout(height=1000,width=1000, dragmode='lasso')
		else:
			fig.update_layout(height=500,width=1000, dragmode='lasso', font_size=20)
	else: 
		try:
			fig = px.scatter_matrix(to_plot, dimensions=selected_feats,opacity=0.6,\
					   hover_data=hover_cols, labels=labels)
		except:
			fig = px.scatter_matrix(to_plot, dimensions=selected_feats,opacity=0.6,\
					   hover_data=hover_cols, labels=labels)
		fig.update_traces(marker=dict(size=3, color='blue'))
		if len(selected_feats) > 4:
			fig.update_layout({ax:{"tickmode":"array","tickvals":[]} for ax in fig.to_dict()["layout"] if "axis" in ax})			
			fig.update_layout(height=1000,width=1000, dragmode='lasso')
			print(fig.layout)
		else:
			fig.update_layout(height=500,width=1000, dragmode='lasso', font_size=17)
		
		if toggle_value and (enlarged_selected or selected_clusters):
			fig.add_traces(
				px.scatter_matrix(to_plot[~to_plot.index.isin(selected_points)], \
			dimensions=selected_feats, labels=labels, opacity=0.1).update_traces(marker=dict(size=3,color='grey')).data
			)
		
	return fig


'''
=============================DISPLAYING HISTOGRAM======================================
'''
@app.callback(
Output('main-plot','style'), # output: visibility of pair-plot
Output('hist-plot', 'style'), # output: visibility of histogram
Output('hist-plot','figure'), # output: histogram plot figure
Output('hist','children'), # output: title on the histogram button
Input('feature-type', 'value'), # input: 'feature-type' selects which feature columns to show in the pair-plot
Input('main-plot', 'selectedData'), # input: selected points from the pair-plots
Input('main-plot', 'figure'), # input: the pair-plot figure so that we can extract the current columsn displayed
Input('enlarged-graph', 'selectedData'), # input: selected points from the feature embedding plot
Input('micro-cluster-scatter', 'selectedData'), # input: selected points from ICA
Input('scale-type', 'value'), # input: log or linear scale
Input('hist','n_clicks')) # input: type of vis. Only make visible if UMAP
def display_histogram(feature_values, selected_from_pair_plots, pair_plot, selectedData, selected_from_ica, scale, histogram_button):

	if not histogram_button:
		try: # this is a hacky way of avoiding the Invalid Value error that randomly happens in Dash. Found from plotly community page
			return {'display':'inline-block'}, {'display': 'None'}, px.scatter([]), "Show Individual"
		except Exception:
			return {'display':'inline-block'}, {'display': 'None'}, px.scatter([]), "Show Individual"
		
	if histogram_button%2==1:
		if not feature_values:
			feature_values = [item['label'] for item in pair_plot['data'][0]['dimensions']]
		else:
			if scale == 'Linear':
				feature_values = [feat+" Val" for feat in feature_values]

		to_plot = plot_df[feature_values]
		if selectedData:
			selected_points = []
			for item in selectedData['points']:
				selected_points.append(item['customdata'][-1])
		elif selected_from_pair_plots:
			selected_points = []
			for item in selected_from_pair_plots['points']:
				selected_points.append(item['customdata'][-1])
		elif selected_from_ica:
			selected_points = []
			for item in selected_from_ica['points']:
				selected_points.append(item['customdata'][-1])

		else:
			selected_points = to_plot.index.values

		to_plot = to_plot.loc[selected_points]

		fig = px.box(to_plot)
		fig.update_layout(font_size=20)
		
		return {'display':'None'}, {'display':'block'}, fig, "Show Pair-Plots"
	else:
		return {'display':'inline-block'}, {'display': 'None'}, px.scatter([]), "Show Individual"
		


if __name__ == '__main__':
	app.run_server(debug=True, port=8801)