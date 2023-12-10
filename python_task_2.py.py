#!/usr/bin/env python
# coding: utf-8

# In[4]:


#Question1.)
import pandas as pd
import networkx as nx

def calculate_distance_matrix():
    # Read the CSV file into a DataFrame
    df = pd.read_csv('dataset-3.csv')

    # Create a graph
    G = nx.Graph()

    # Add edges and their weights to the graph
    for index, row in df.iterrows():
        G.add_edge(row['start_id'], row['end_id'], weight=row['distance'])

    # Create a dictionary to store the cumulative distances
    distance_dict = {}

    # Calculate cumulative distances between toll locations
    for start_node in G.nodes():
        for end_node in G.nodes():
            if start_node != end_node:
                if nx.has_path(G, start_node, end_node):
                    path_distance = nx.shortest_path_length(G, source=start_node, target=end_node, weight='weight')
                    distance_dict[(start_node, end_node)] = path_distance
                    distance_dict[(end_node, start_node)] = path_distance

    # Create a DataFrame from the distance dictionary
    distance_matrix = pd.DataFrame.from_dict(distance_dict, orient='index', columns=['distance'])
    distance_matrix.index.names = ['start_id', 'end_id']
    distance_matrix.reset_index(inplace=True)

    return distance_matrix
distance_matrix_result = calculate_distance_matrix()
print(distance_matrix_result)


# In[13]:


# Add edges and their distances to the graph
df = pd.read_csv('dataset-3.csv')
import pandas as pd
import networkx as nx

def calculate_distance_matrix(dataframe):
# Create a directed graph to represent toll locations and distances
G = nx.DiGraph()
for _, row in df.iterrows():
    G.add_edge(row['toll_booth_A'], row['toll_booth_B'], distance=row['distance'])
    G.add_edge(row['toll_booth_B'], row['toll_booth_A'], distance=row['distance'])

# Use the Floyd-Warshall algorithm to calculate the shortest paths between all pairs of nodes
distances = nx.floyd_warshall(G, weight='distance')

# Create a DataFrame from the calculated distances
distance_matrix = pd.df(distances, index=G.nodes, columns=G.nodes)

# Set diagonal values to 0
distance_matrix.values[[range(distance_matrix.shape[0])]*2] = 0

return distance_matrix


# In[ ]:





# In[5]:


#Question2.)
import pandas as pd

def unroll_distance_matrix(distance_matrix):
    # Create an empty DataFrame to store unrolled distances
    unrolled_df = pd.DataFrame(columns=['id_start', 'id_end', 'distance'])

    # Iterate over each row in the distance matrix
    for index, row in distance_matrix.iterrows():
        id_start = row['start_id']
        id_end = row['end_id']
        distance = row['distance']

        # Append a row for the current combination
        unrolled_df = unrolled_df.append({'id_start': id_start, 'id_end': id_end, 'distance': distance}, ignore_index=True)

        # If id_start and id_end are different, append a row for the reverse combination
        if id_start != id_end:
            unrolled_df = unrolled_df.append({'id_start': id_end, 'id_end': id_start, 'distance': distance}, ignore_index=True)

    return unrolled_df

# Assuming distance_matrix_result is the DataFrame obtained from Question 1
unrolled_distance_df = unroll_distance_matrix(distance_matrix_result)

# Display the result DataFrame
print(unrolled_distance_df)


# In[9]:


distance_df = df.pivot(index='id_1', columns='id_2', values='distance')

np.fill_diagonal(distance_df.values, 0)

for i in range(len(distance_df)):
    for j in range(len(distance_df)):
        if i != j and not np.isnan(distance_df.iloc[i, j]):
            for k in range(len(distance_df)):
                if (k != i and k != j and 
                    not np.isnan(distance_df.iloc[i, k]) and 
                    not np.isnan(distance_df.iloc[k, j])):
                    distance_df.iloc[i, j] = distance_df.iloc[i, k] + distance_df.iloc[k, j]

distance_df = distance_df.combine(distance_df.T, max, fill_value=np.nan)

return distance_df


# In[6]:


#Question3.
import pandas as pd

def find_ids_within_ten_percentage_threshold(distance_df, reference_value):
    # Filter rows for the reference value in 'id_start' column
    reference_rows = distance_df[distance_df['id_start'] == reference_value]

    # Calculate the average distance for the reference value
    reference_avg_distance = reference_rows['distance'].mean()

    # Define the percentage threshold
    threshold_percentage = 10

    # Calculate the lower and upper bounds for the threshold
    lower_bound = reference_avg_distance - (reference_avg_distance * threshold_percentage / 100)
    upper_bound = reference_avg_distance + (reference_avg_distance * threshold_percentage / 100)

    # Filter rows within the threshold
    within_threshold = distance_df[
        (distance_df['distance'] >= lower_bound) &
        (distance_df['distance'] <= upper_bound)
    ]

    # Get unique values from the 'id_start' column and sort them
    result_ids = sorted(within_threshold['id_start'].unique())

    return result_ids

# Assuming your DataFrame is named 'distance_df'
# and reference_value is the reference value for 'id_start'

reference_value = 123  # Replace with the actual reference value
result_ids_within_threshold = find_ids_within_ten_percentage_threshold(distance_df, reference_value)
print(result_ids_within_threshold)


# In[7]:


#Question 4
import pandas as pd

def calculate_toll_rate(distance_df):
    # Create a copy of the input DataFrame to avoid modifying the original
    result_df = distance_df.copy()

    # Define rate coefficients for each vehicle type
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}

    # Calculate toll rates for each vehicle type
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        result_df[vehicle_type] = result_df['distance'] * rate_coefficient

    return result_df

# Assuming distance_df is the DataFrame obtained from the previous step
# Call the function
result_with_toll_rates = calculate_toll_rate(distance_df)

# Display the result DataFrame with toll rates
print(result_with_toll_rates)


# In[14]:


#Question5.)
import pandas as pd
from datetime import datetime, timedelta, time

def calculate_time_based_toll_rates(input_df):
    # Create a copy of the input DataFrame to avoid modifying the original
    result_df = input_df.copy()

    # Define time ranges and discount factors
    weekday_time_ranges = [(time(0, 0, 0), time(10, 0, 0)), (time(10, 0, 0), time(18, 0, 0)), (time(18, 0, 0), time(23, 59, 59))]
    weekend_time_ranges = [(time(0, 0, 0), time(23, 59, 59))]
    weekday_discount_factors = [0.8, 1.2, 0.8]
    weekend_discount_factor = 0.7

    # Iterate over each row in the DataFrame
    for index, row in result_df.iterrows():
        # Determine the day of the week for the start_day
        start_day_of_week = datetime.strptime(row['start_day'], '%A').strftime('%A')

        # Determine the time range and discount factor based on the day of the week
        if start_day_of_week in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            time_ranges = weekday_time_ranges
            discount_factors = weekday_discount_factors
        else:
            time_ranges = weekend_time_ranges
            discount_factors = [weekend_discount_factor]

        # Apply discount factors based on the time range
        for i, (start_time_range, end_time_range) in enumerate(time_ranges):
            if start_time_range <= row['start_time'] <= end_time_range and end_time_range >= row['end_time'] >= start_time_range:
                result_df.at[index, 'toll_rate'] = result_df.at[index, 'distance'] * discount_factors[i]

    return result_df

result_with_time_based_toll_rates = calculate_time_based_toll_rates(distance_df)
print(result_with_time_based_toll_rates)

