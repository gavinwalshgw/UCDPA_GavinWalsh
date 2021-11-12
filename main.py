# Importing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import kaggle
# Plot Style
plt.style.use('ggplot')

# ##### Assignment Part 1 Real World Dataset: Netflix Data from Kaggle
# ##### Assignment Part 2 'Importing Data': API
# ##### Assignment Part 2 'Importing Data': Import a CSV file into a Pandas DataFrame

# Checking to see if the data is downloaded already
import os
# If the file is not present in the working directory
if not os.path.isfile('netflix-shows.zip'):
    # Execute a command in console that downloads the dataset through the Kaggle API
    # Note: API command copied and pasted from Kaggle Dataset
    os.system('kaggle datasets download -d shivamb/netflix-shows')
# Read the dataset from zip archive
import zipfile


# Create an object for the archive
data_archive = zipfile.PyZipFile('netflix-shows.zip')
# Get filename of the first file in the archive (This is the dataset)
data_filename = data_archive.filelist[0].filename

# With the opened file
with data_archive.open(data_filename) as datafile:
    # Read the file (.csv dataset) into pandas dataframe
    # Specify the date column (its format will be inferred automatically)
    data = pd.read_csv(datafile, parse_dates=['date_added'])

# ##### Assignment Part 3 Analysing Data: Replace missing values and using .loc
print(data.isna().sum())
# Replace missing for "date added"
# Replace "date added" with the date of release
# Reasonable assumption that date added is approximately equal to date of release
# Mask of missing data
missing = data.date_added.isna()
# Assign to corresponding places in the frame
data.loc[missing, 'date_added'] = data.release_year[missing]


# ##### Assignment Part 4 Python: Define a custom function to create reusable code
# ##### Assignment Part 3 Analysing Data: Sorting
# ##### Assignment Part 3 Analysing Data: Grouping
# A function to plot a histogram of items in a Series
# Works similarly to "histplot" from seaborn
# Also group least common entries in "Others" category


def plot_hist(series, entries_to_plot=15, title=None):
    # Count unique values
    counted = series.value_counts().sort_values(ascending=False)

    # Only keep some number of most common ones
    to_plot = counted[:entries_to_plot]
    # If the are "leftovers", group them together and add as "Others"
    if entries_to_plot < len(counted):
        to_plot['Others'] = counted[entries_to_plot:].sum()

    #  ##### Assignment Part 5 Visualise: MatPlotlib
    # Use pandas shortcut to matplotlib barplot to display the values
    to_plot.plot(kind='bar', figsize=(entries_to_plot * 0.4 + 3, 6), title=title)
    # Add a label to the Y axis
    plt.ylabel('Number of Movies and TV shows')
    plt.show()

# ##### Assignment Part 6 Generate valuable insights: Insight 1
# ##### Assignment Part 4 Python: Lists
# ## Distribution of the number of countries per Movies and TV Shows Projects
# Each entry is a list of values: list of countries of list of categories
# Drop missing, split by commas, count the number of items, convert integers for nicer output (no trailing ".0")


countries_counts = data.country.dropna().str.split(', ').str.len().astype(int)
plot_hist(countries_counts, title='Number of countries a movie or a TV show was produced in')

# ## Distribution of the number of categories each item is listed in
# Same for categories
categories_counts = data.listed_in.dropna().str.split(', ').str.len().astype(int)
plot_hist(categories_counts, title='Number of categories a movie or a TV show is listed in')


# ##### Assignment Part 6 Generate valuable insights: Insight 2
# ## Most common categories
# Look at which categories are most commonly present
# First split every string into a list, then unroll all the lists
categories = data.listed_in.str.split(', ').explode()
plot_hist(categories, title='Most common categories\n (a movie or a TV show can be present in more than one)')


# ##### Assignment Part 6 Generate valuable insights: Insight 3
# ## Most common combinations of categories
# Use ".apply(tuple)" to convert lists to tuples
# Which is needed to count the number of unique values
# The values have to be hashable for pandas to work well
categories = data.listed_in.str.split(', ').apply(tuple)
plot_hist(categories, 25, title='Most common combination of movie and TV show categories')

# ## Most common combinations of countries
# Use frozenset instead of tuple to ignore the order of countries listed.
# Don't want to count (USA, UK) and (UK, USA) as two different entries
countries = data.country.dropna().str.split(', ').apply(frozenset)
plot_hist(countries, 30, title='Most common combinations of movies and TV shows location of production')


# ##### Assignment Part 6 Generate valuable insights: Insight 4
# ##### Assignment Part 3 Analysing Data: Merge Dataframes & Indexing
# ## Cross-map of countries and categories
# Creating a dataframe that only has two columns — countries and categories.
# `Merge` two frames by the index.
# `explode` both columns (in a sense converting the frame from the "wide" to the "long" format)
# Reset the index.
# Rename columns to have easier to read/ display labels in the plot.

data2 = pd.merge(data.country.dropna().str.split(', ').to_frame(),
                 data.listed_in.str.split(', ').to_frame(),
                 left_index=True, right_index=True)

data2 = data2.explode('country').explode('listed_in').reset_index().drop(columns='index')
data2 = data2.rename(columns={'country': 'Country', 'listed_in': 'Category'})

# ##### Assignment Part 6 Generate valuable insights: Insight 5
# ##### Assignment Part 3 Analysing Data: Looping & Iterrows & Slicing
# There are 127 unique countries in this new frame (unique_countries = data2.Country.value_counts()).
# Displaying them all would make the plot too overloaded and almost unreadable,
# Therefore limited to  only a few most commonly present countries.

# Creating a list of the most common countries.
# Iterate over every row in the new frame and check whether the country is present in this list.
# If it is, save its index to the list of rows we will delete later.
# After the looping is done, `drop` the rows with uncommon countries.

# The 2D histogram is built with the pandas method `crosstab`.
# It produces a new (third) frame that has [most common] countries as rows and categories as columns.
# Use the seaborn method `heatmap` to display the calculated histogram.

# building cross-table "country/category"
# get names of most popular countries
unique_countries = data2.Country.value_counts()
print(unique_countries)

most_popular_countries = data2.Country.value_counts().index[:10]


to_delete = []  # holds indices of rows that will be deleted later
for row in data2.index:  # iterate over every row
    # if row has an uncommon country
    if data2.loc[row, 'Country'] not in most_popular_countries:
        to_delete.append(row)  # mark for deletion

# delete "uninteresting" rows
data2 = data2.drop(to_delete)

# calculate the heatmap
countries_categories_heatmap = pd.crosstab(data2.Category, data2.Country)

# display the heatmap
plt.figure(figsize=(21, 4))
# grey colormap seems to give the best picture
# `robust` means that quantiles for colormap are selected so that
# outliers don't influence that much (USA is overrepresented, for example)
# ### Assignment Part 5 Visualise: Seaborn
sns.heatmap(countries_categories_heatmap.T, cmap='Greys', robust=True)
plt.show()


# ### Assignment Part 4 Python: Numpy
# NOTE:
#   Numpy package was not used to generate insights as insights did not require mathematical operations
#   The below code is to demonstrate the capability of using the Numpy Package
# Standard array when added together will simply join the two arrays together
Array1 = [1, 2, 3, 4, 5]
Array2 = [6, 7, 8, 9, 10]
ArraySum = Array1 + Array2
print(ArraySum)
# Using a numpy array will add each individual term
npArray1 = np.array([1, 2, 3, 4, 5])
npArray2 = np.array([6, 7, 8, 9, 10])
npArraySum = npArray1 + npArray2
print(npArraySum)
