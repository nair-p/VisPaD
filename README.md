# VisPaD
An interactive visual system designed for domain experts to annotate suspicious clusters of escort ads

## Creating a virtual environment

It is highly recommended to use a virtual environment for running this app as it helps dealing with dependencies of packages.

If you don't have virtualenv installed, run
```
pip install virtualenv
```

After installing virtualenv, create a virtual env for the app by running
```
python -m venv vispad_env
```

If you have a virutal env or after creating one, activate it by running
```
source vispad_env/bin/activate
```

Once the virtual env has been activated, the necessary packages can be installed within that environment.

## Install necessary libraries:

After activating your environment, install the required libraries using 

```
pip install -r requirements.txt
```

# Usage:

Some of the required data files are included. Data that contains person identifying information cannot be shared publicly but can be made available on request after agreeing to the terms of sharing. Please reach out by sending an email to [this address](mailto:pratheeksha.nair@mail.mcgill.ca) for dataset access. 

However, in order to demonstrate how the tool works, we've included some synthetic [dummy data](/synthetic_data/synthetic_data_changed.csv) in this repository. The text column in this dataframe does not make sense as it is a constructed by randomly sampling words from the english dictionary. 

## VisPaD
```
python app.py
```

There is a top summary banner and 2 main components (`Inspect Clusters` and `Analysis`) in VisPad.

In the `Inspect Clusters` tab you see,

1. Metadata summary of the selected micro-clusters which appears in the header of the dashboard
2. Ads posted over time for each micro-cluster color-coded by meta-cluster ID
3. Metadata over time for each micro-cluster color coded by meta-cluster ID. Size of the bubble indicates count. 
4. Geographical spread of the ads across all selected micro-cluster color coded by meta-cluster ID. Size of the bubble indicates the count. 
5. Description of ads in the current micro-cluster. 


In the `Analysis` tab you see,

1. Feature embeddings of the micro-cluster. You can choose between ICA (Independent Component Analysis), t-SNE and UMAP which are all dimensionality reduction techniques, for plotting the 9 dimensional micro-cluster feature vectors in 2 dimensions in a meaningful way. Each point on the plot represents a micro-cluster in the data and on hovering on a point, you can see its feature values.
2. If you choose t-SNE/UMAP, it displays the plots for different values of the respective parameters and clicking on one of the sub-plots enlarges it for a better view.
3. The `Meta-Cluster Labels` button indicates by color, which micro-clusters belong to the same meta-cluster. Interesting groups of micro-clusters can be selected using the mouse pointer for further analysis.
4. On making a selection, the pair-plots on the right get updated to show you the 4 features that are most relevant to the selected micro-clusters. You can also look at the box-plot distribution of the individual feautres by clicking on the `Show Individual` button. You can also toggle between showing only selected points and all the points.
5. On making a selection, the top summary panel also gets updated to reflect the selected points and on going back to the `Inspect Clusters` tab, you can see the updated information for those micro-clusters. 

## Deactivate your virtual environment

When you're done using the app, the virtual env can be deactivated by running
```
deactivate
```
