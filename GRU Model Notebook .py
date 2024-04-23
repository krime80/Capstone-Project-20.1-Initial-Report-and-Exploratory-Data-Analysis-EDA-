#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import GRU, Dense, Masking

# Define File Path
file_path = "/Users/krista.rime/Documents/AIML/capstone_project.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Convert 'EVENT_OCCURRED_UTC' to datetime
df['EVENT_OCCURRED_UTC'] = pd.to_datetime(df['EVENT_OCCURRED_UTC'])

# Sort DataFrame based on 'EVENT_OCCURRED_UTC'
df.sort_values(by='EVENT_OCCURRED_UTC', inplace=True)

# Reset index
df.reset_index(drop=True, inplace=True)

# Drop the 'HIT_ID' column 
if 'HIT_ID' in df.columns:
    df.drop(columns=['HIT_ID'], inplace=True)

# Convert 'EVENT_OCCURRED_UTC' to Unix timestamps
df['EVENT_OCCURRED_UNIX'] = df['EVENT_OCCURRED_UTC'].apply(lambda x: x.timestamp())

# Define X
X = df.copy()

# Define datetime features and numerical features
sequences = df.groupby(['CLIENT_ID', 'SESSION_ID'])
datetime_features = df.select_dtypes(include=['datetime64']).columns
numerical_features = df.select_dtypes(include=['float64', 'int64']).columns

# Filter out non-numeric data
numeric_df = df[df['EVENT_OCCURRED_UTC'].apply(lambda x: isinstance(x, (int, float)))]

# Debugging: Print numeric_df to check if it contains numerical data
print("Numeric DataFrame:")
print(numeric_df.head())

# Check if there are any rows left in the numeric_df
if len(numeric_df) > 0:
    # Define numerical features
    numerical_features = ['EVENT_OCCURRED_UTC']

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    numeric_df[numerical_features] = imputer.fit_transform(numeric_df[numerical_features])
else:
    print("No numerical data found.")
    # If 'EVENT_OCCURRED_UTC' contains non-numeric data, consider converting it to numerical format
    # For example, you can convert datetime objects to Unix timestamps
    df['EVENT_OCCURRED_UTC'] = pd.to_datetime(df['EVENT_OCCURRED_UTC']).apply(lambda x: x.timestamp())
    print("Converted 'EVENT_OCCURRED_UTC' column to Unix timestamps.")

# Convert datetime features into numerical format
df[datetime_features] = df[datetime_features].astype(np.int64)

# Debugging: Print the DataFrame info to check data types
print("\nDataFrame Info:")
print(df.info())

# Define Sequencing Rules
for _, group in sequences:
    # Iterate over each group and access its data
    for index, row in group.iterrows():
        # Access each row's data within the group
        eventname = row['EVENTNAME']
        # Process the row's data or apply your sequencing rules here

def is_in_sequence(event_sequence):
    # Define your sequencing rules here
    # Initialize flags for clipStart, clipEnd, cmPodBegin, cmPodEnd
    clipStart_flag = False
    clipEnd_flag = False
    cmPodBegin_flag = False
    cmPodEnd_flag = False

    # Iterate through the event sequence
    for _, event in event_sequence.iterrows():
        if event['EVENTNAME'] == 'clipStart':
            clipStart_flag = True
        elif event['EVENTNAME'] == 'clipEnd':
            clipEnd_flag = True
        elif event['EVENTNAME'] == 'cmPodBegin':
            cmPodBegin_flag = True
        elif event['EVENTNAME'] == 'cmPodEnd':
            cmPodEnd_flag = True

    # Check if the sequencing rules are satisfied
    if (clipStart_flag and clipEnd_flag) or (cmPodBegin_flag and cmPodEnd_flag):
        return True
    else:
        return False

# Mark Events in Sequence
# Define the function to convert nanoseconds to seconds since the start of the day
def convert_to_seconds(dt_int):
    # Convert nanoseconds to seconds
    dt_sec = (dt_int - dt_int.normalize()).total_seconds()
    # Calculate the number of seconds since the start of the day
    seconds_since_start = dt_sec % (24 * 3600)
    return seconds_since_start

# Define a function to mark events in sequence
def mark_in_sequence(df):
    # Calculate the time difference between consecutive events
    df['TIME_DIFF'] = df['EVENT_OCCURRED_UTC'].diff().fillna(pd.Timedelta(seconds=0))
    
    # Check if 'TIME_DIFF' is of timedelta type
    if pd.api.types.is_timedelta64_ns_dtype(df['TIME_DIFF']):
        # Define a threshold for considering events in sequence
        threshold = pd.Timedelta(minutes=5)  # Adjust as needed
        
        # Mark events as in sequence if the time difference is within a threshold, otherwise mark as out of sequence
        df['IN_SEQUENCE'] = (df['TIME_DIFF'] <= threshold).astype(int)
    else:
        # If 'TIME_DIFF' is not timedelta, mark all events as out of sequence
        df['IN_SEQUENCE'] = 0
        
    return df

# Reset the index of X
X_reset = X.reset_index(drop=True)

# Mark events as in or out of sequence
X_marked = X_reset.groupby('EVENTNAME').apply(mark_in_sequence)

# Reset the index of X_marked
X_marked_reset = X_marked.reset_index(drop=True)

# Group by 'EVENTNAME' and calculate the maximum value of 'IN_SEQUENCE'
y_aggregated = X_marked_reset.groupby('EVENTNAME')['IN_SEQUENCE'].max()

# Ensure the alignment of X_marked and y_aggregated
X_marked_aligned = X_marked_reset[X_marked_reset['EVENTNAME'].isin(y_aggregated.index)]

# Debugging: Print X_marked_aligned to check its contents
print("\nX_marked_aligned DataFrame:")
print(X_marked_aligned.head())

# Reset the index of y_aggregated
y_aggregated_reset = y_aggregated.reset_index()

# Merge X_marked_aligned and y_aggregated_reset on 'EVENTNAME'
if 'EVENTNAME' in X_marked_aligned.columns and 'EVENTNAME' in y_aggregated_reset.columns:
    merged_df = pd.merge(X_marked_aligned, y_aggregated_reset, on='EVENTNAME')
    
    # Drop the redundant 'IN_SEQUENCE_y' column
    merged_df.drop(columns=['IN_SEQUENCE_y'], inplace=True)
    
    # Check if 'IN_SEQUENCE_x' exists in merged_df before dropping it
    if 'IN_SEQUENCE_x' in merged_df.columns:
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            merged_df.drop('IN_SEQUENCE_x', axis=1),
            merged_df['IN_SEQUENCE_x'],  # Use the correct column name here
            test_size=0.2,
            stratify=merged_df['IN_SEQUENCE_x'],  # Use the correct column name here
            random_state=42
        )
        
        # Convert X_train and X_test to list of sequences
        X_train_sequences = [group.drop(columns=['EVENTNAME']).select_dtypes(include=['float64', 'int64']).values.tolist() for _, group in X_train.groupby(level=0)]
        X_test_sequences = [group.drop(columns=['EVENTNAME']).select_dtypes(include=['float64', 'int64']).values.tolist() for _, group in X_test.groupby(level=0)]

        # Pad sequences to a fixed length
        max_sequence_length = max(max(len(seq) for seq in X_test_sequences), max(len(seq) for seq in X_train_sequences))
        X_train_padded = pad_sequences(X_train_sequences, maxlen=max_sequence_length, padding='post', truncating='post', value=0, dtype='float32')
        X_test_padded = pad_sequences(X_test_sequences, maxlen=max_sequence_length, padding='post', truncating='post', value=0, dtype='float32')

        # Convert the target data to float32
        y_train = y_train.astype(np.float32)
        y_test = y_test.astype(np.float32)

        # Define the input shape
        input_shape = (max_sequence_length, X_train_padded.shape[2])  # Shape of input data for GRU

        # Define the GRU model with Masking layer
        model = Sequential()
        model.add(Masking(mask_value=0., input_shape=input_shape))  # Masking zero-padded values
        model.add(GRU(units=50))
        model.add(Dense(units=1, activation='sigmoid'))

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(X_train_padded, y_train, epochs=10, batch_size=32, validation_split=0.2)

        # Evaluate the model
        test_loss, test_accuracy = model.evaluate(X_test_padded, y_test)
        print("Test Loss:", test_loss)
        print("Test Accuracy:", test_accuracy)
    else:
        print("Column 'IN_SEQUENCE' not found in merged_df. Check the merging step.")
else:
    print("Column 'EVENTNAME' not found in either X_marked_aligned or y_aggregated_reset.")


# In[3]:


from tabulate import tabulate

# Define a function to mark events in sequence and identify out-of-sequence events
def mark_and_identify_in_sequence(df):
    # Calculate the time difference between consecutive events
    df['TIME_DIFF'] = df['EVENT_OCCURRED_UTC'].diff().fillna(pd.Timedelta(seconds=0))
    
    # Check if 'TIME_DIFF' is of timedelta type
    if pd.api.types.is_timedelta64_ns_dtype(df['TIME_DIFF']):
        # Define a threshold for considering events in sequence
        threshold = pd.Timedelta(minutes=5)  # Adjust as needed
        
        # Mark events as in sequence if the time difference is within a threshold, otherwise mark as out of sequence
        df['IN_SEQUENCE'] = (df['TIME_DIFF'] <= threshold).astype(int)
        
        # Identify out-of-sequence events
        df['OUT_OF_SEQUENCE'] = df['IN_SEQUENCE'].apply(lambda x: not bool(x))
        
    else:
        # If 'TIME_DIFF' is not timedelta, mark all events as out of sequence
        df['IN_SEQUENCE'] = 0
        df['OUT_OF_SEQUENCE'] = 1
        
    return df

# Mark events as in or out of sequence and identify out-of-sequence events
X_marked_with_out_of_sequence = X_reset.groupby('EVENTNAME').apply(mark_and_identify_in_sequence)

# Reset the index of X_marked_with_out_of_sequence
X_marked_with_out_of_sequence_reset = X_marked_with_out_of_sequence.reset_index(drop=True)

# Display events that are out of sequence along with their associated metadata
out_of_sequence_events = X_marked_with_out_of_sequence_reset[X_marked_with_out_of_sequence_reset['OUT_OF_SEQUENCE'] == 1]

print("Out of Sequence Events:")
print(out_of_sequence_events)


# In[4]:


# Mark events as in or out of sequence and identify out-of-sequence events
X_marked_with_out_of_sequence = X_reset.groupby('EVENTNAME').apply(mark_and_identify_in_sequence)

# Reset the index of X_marked_with_out_of_sequence
X_marked_with_out_of_sequence_reset = X_marked_with_out_of_sequence.reset_index(drop=True)

# Display events that are out of sequence along with their associated metadata
out_of_sequence_events = X_marked_with_out_of_sequence_reset[X_marked_with_out_of_sequence_reset['OUT_OF_SEQUENCE'] == 1]

# Convert the DataFrame to a list of lists for tabulate
out_of_sequence_table = out_of_sequence_events.values.tolist()

# Define the headers for the table
headers = out_of_sequence_events.columns.tolist()

# Print the table using tabulate
print("Out of Sequence Events:")
print(tabulate(out_of_sequence_table, headers=headers, tablefmt='grid'))


# In[11]:


# Summarize the events out of sequence
def summarize_out_of_sequence(df):
    # Total count of events
    total_events = len(df)
    
    # Total count of events out of sequence
    total_out_of_sequence = df['OUT_OF_SEQUENCE'].sum()
    
    # Percentage of events out of sequence
    percentage_out_of_sequence = (total_out_of_sequence / total_events) * 100
    
    # Total count of events out of sequence by app
    out_of_sequence_by_app = df.groupby('APP_NAME')['OUT_OF_SEQUENCE'].sum()
    
    # Total count of events out of sequence by app version
    out_of_sequence_by_app_version = df.groupby(['APP_NAME', 'APP_VERSION'])['OUT_OF_SEQUENCE'].sum()
    
    # Total events out of sequence by session count
    out_of_sequence_by_session_count = df.groupby('SESSION_ID')['OUT_OF_SEQUENCE'].sum()
    
    # Total events out of sequence by client ID count
    out_of_sequence_by_client_count = df.groupby('CLIENT_ID')['OUT_OF_SEQUENCE'].sum()
    
    # Total count of session IDs with out-of-sequence events
    total_sessions_with_out_of_sequence = (out_of_sequence_by_session_count > 0).sum()
    
    # Total count of client IDs with out-of-sequence events
    total_clients_with_out_of_sequence = (out_of_sequence_by_client_count > 0).sum()
    
    # Calculate total count of session IDs
    total_session_ids = len(out_of_sequence_by_session_count)
    
    # Calculate total count of client IDs
    total_client_ids = len(out_of_sequence_by_client_count)
    
    # Calculate total percentage of sessions with out-of-sequence events
    percentage_sessions_with_out_of_sequence = (total_sessions_with_out_of_sequence / total_session_ids) * 100
    
    # Calculate total percentage of clients with out-of-sequence events
    percentage_clients_with_out_of_sequence = (total_clients_with_out_of_sequence / total_client_ids) * 100
    
    return total_out_of_sequence, percentage_out_of_sequence, out_of_sequence_by_app, out_of_sequence_by_app_version, out_of_sequence_by_session_count, out_of_sequence_by_client_count, total_sessions_with_out_of_sequence, total_clients_with_out_of_sequence, percentage_sessions_with_out_of_sequence, percentage_clients_with_out_of_sequence

# Call the summarize function and unpack the results
total_out_of_sequence, percentage_out_of_sequence, out_of_sequence_by_app, out_of_sequence_by_app_version, out_of_sequence_by_session_count, out_of_sequence_by_client_count, total_sessions_with_out_of_sequence, total_clients_with_out_of_sequence, percentage_sessions_with_out_of_sequence, percentage_clients_with_out_of_sequence = summarize_out_of_sequence(X_marked_with_out_of_sequence_reset)

# Print the summaries
print("Total Count of Events Out of Sequence:", total_out_of_sequence)
print("Percentage of Events Out of Sequence:", percentage_out_of_sequence, "%")
print("\nTotal Count of Events Out of Sequence by App:")
print(out_of_sequence_by_app)
print("\nTotal Count of Events Out of Sequence by App Version:")
print(out_of_sequence_by_app_version)
print("\nTotal Count of Session IDs with Out-of-Sequence Events:", total_sessions_with_out_of_sequence)
print("Percentage of Session IDs with Out-of-Sequence Events:", percentage_sessions_with_out_of_sequence, "%")
print("\nTotal Count of Client IDs with Out-of-Sequence Events:", total_clients_with_out_of_sequence)
print("Percentage of Client IDs with Out-of-Sequence Events:", percentage_clients_with_out_of_sequence, "%")


# In[ ]:





# In[ ]:




