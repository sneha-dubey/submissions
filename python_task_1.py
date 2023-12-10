#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np


# In[7]:


# Question1
import pandas as pd
def generate_car_matrix():
  df = pd.read_csv('dataset-1.csv')
  result_df = df.pivot(index='id_1', columns='id_2', values='car').fillna(0)
  for i in result_df.index:
      result_df.at[i, i] = 0
      return result_df
result_dataframe = generate_car_matrix()
print(result_dataframe)


# In[8]:


#Question2.
import pandas as pd
def get_type_count():
    df = pd.read_csv('dataset-1.csv')
# Add a new categorical column 'car_type'
    conditions = [
        (df['car'] <= 15),
        (df['car'] > 15) & (df['car'] <= 25),
        (df['car'] > 25)
    ]
    choices = ['low', 'medium', 'high']
    df['car_type'] = pd.Series(np.select(conditions, choices, default='Unknown'), dtype="category")

    # Calculate the count of occurrences for each 'car_type' category
    type_count = df['car_type'].value_counts().to_dict()

    # Sort the dictionary alphabetically based on keys
    sorted_type_count = dict(sorted(type_count.items()))

    return sorted_type_count
type_count_result = get_type_count()
print(type_count_result)


# In[9]:


#Question3.
import pandas as pd

def get_bus_indexes():
    # Read the CSV file into a DataFrame
    df = pd.read_csv('dataset-1.csv')

    # Calculate the mean value of the 'bus' column
    mean_bus_value = df['bus'].mean()

    # Identify indices where 'bus' values are greater than twice the mean
    bus_indexes = df[df['bus'] > 2 * mean_bus_value].index.tolist()

    # Sort the indices in ascending order
    bus_indexes.sort()

    return bus_indexes

bus_indexes_result = get_bus_indexes()
print(bus_indexes_result)


# In[10]:


#Question 4.

def filter_routes():
    df = pd.read_csv('dataset-1.csv')

    # Group by 'route' and calculate the average of 'truck' column for each route
    route_avg_truck = df.groupby('route')['truck'].mean()

    # Filter routes where the average of 'truck' column is greater than 7
    selected_routes = route_avg_truck[route_avg_truck > 7].index.tolist()

    # Sort the list of selected routes
    selected_routes.sort()

    return selected_routes

filtered_routes_result = filter_routes()
print(filtered_routes_result)


# In[13]:


#Question 5.)
import pandas as pd

def multiply_matrix(input_df):
    # Deep copy the input DataFrame to avoid modifying the original
    modified_df = input_df.copy()

    # Apply the modification logic to each value in the DataFrame
    for i in modified_df.index:
        for j in modified_df.columns:
            if modified_df.at[i, j] > 20:
                modified_df.at[i, j] *= 0.75
            else:
                modified_df.at[i, j] *= 1.25

    # Round values to 1 decimal place
    modified_df = modified_df.round(1)

    return modified_df

# Assuming result_dataframe is the DataFrame obtained from Question 1
# Call the function
modified_result_dataframe = multiply_matrix(result_dataframe)
print(modified_result_dataframe)


# In[18]:


#Question 6.)
import pandas as pd
df = pd.read_csv('dataset-2.csv')

def check_timestamp_completeness(df):
    # Combine 'startDay' and 'startTime' columns into a single datetime column
    df['start_timestamp'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])

    # Combine 'endDay' and 'endTime' columns into a single datetime column
    df['end_timestamp'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])

    # Create a new column to represent the day of the week (Monday=0, Sunday=6)
    df['day_of_week'] = df['start_timestamp'].dt.dayofweek

    # Group by 'id' and 'id_2' and check if each group covers a full 24-hour period
    completeness_check = df.groupby(['id', 'id_2']).apply(lambda group: check_time_coverage(group))

    return completeness_check

def check_time_coverage(group):
    # Check if the group covers a full 24-hour period and spans all 7 days of the week
    time_coverage_check = (
        (group['start_timestamp'].min().time() == pd.Timestamp('12:00:00').time()) and
        (group['end_timestamp'].max().time() == pd.Timestamp('23:59:59').time()) and
        (set(group['day_of_week']) == set(range(7)))
    )

    return time_coverage_check
completeness_result = check_timestamp_completeness(df)
print(completeness_result)

