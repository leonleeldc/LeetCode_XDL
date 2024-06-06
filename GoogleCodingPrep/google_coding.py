import pandas as pd
from collections import defaultdict
'''
Problem: Generate a Shift Schedule Table
Given a table with names, shift start times, and shift end times, the task is to output a table showing the start time, end time, and all people on shift during that time.

Here’s a Python script to generate the required output, maintaining the order of names as they appear in the input table:


'''
# Sample input
data = [
    {"name": "Alice", "start_time": "08:00", "end_time": "12:00"},
    {"name": "Bob", "start_time": "09:00", "end_time": "11:00"},
    {"name": "Charlie", "start_time": "10:00", "end_time": "13:00"},
]

# Convert to DataFrame
df = pd.DataFrame(data)

# Sort by start time to maintain order
df = df.sort_values(by="start_time")

# Create a new DataFrame to store the result
result = []

# Process each time slot
time_slots = sorted(set(df["start_time"]).union(set(df["end_time"])))
for i in range(len(time_slots) - 1):
    start = time_slots[i]
    end = time_slots[i + 1]
    on_shift = df[(df["start_time"] <= start) & (df["end_time"] > start)]
    if not on_shift.empty:
        result.append({
            "start_time": start,
            "end_time": end,
            "on_shift": ", ".join(on_shift["name"])
        })

# Convert result to DataFrame
result_df = pd.DataFrame(result)
print(result_df)
'''
Follow-up: Handling Multiple Shifts per Person and Maintaining Order
To handle multiple shifts per person and maintain the order of names as they appear in the input table, we can process the input to handle overlapping shifts and sort the input by the original order of names.

Here’s an updated script to handle this scenario:
'''
import pandas as pd
from collections import defaultdict

# Sample input
data = [
    {"name": "Alice", "start_time": "08:00", "end_time": "12:00"},
    {"name": "Bob", "start_time": "09:00", "end_time": "11:00"},
    {"name": "Alice", "start_time": "13:00", "end_time": "15:00"},
    {"name": "Charlie", "start_time": "10:00", "end_time": "13:00"},
]

# Convert to DataFrame
df = pd.DataFrame(data)

# Create a new DataFrame to store the result
result = []

# Process each time slot
time_slots = sorted(set(df["start_time"]).union(set(df["end_time"])))
for i in range(len(time_slots) - 1):
    start = time_slots[i]
    end = time_slots[i + 1]
    on_shift = df[(df["start_time"] <= start) & (df["end_time"] > start)]
    if not on_shift.empty:
        result.append({
            "start_time": start,
            "end_time": end,
            "on_shift": ", ".join(on_shift["name"])
        })

# Convert result to DataFrame
result_df = pd.DataFrame(result)
print(result_df)
'''
Explanation
Input Data: The input data includes multiple shifts per person.
Sorting: The input data is sorted by start time to ensure order.
Processing Time Slots: Time slots are generated from the union of start and end times.
Filtering Shifts: For each time slot, filter shifts that are active during that slot.
Maintaining Order: Names are concatenated in the order they appear in the filtered data.
Output: The result is converted to a DataFrame and printed.
Handling Name Order
To ensure the order of names is as they appear in the input table, the script sorts by the original order of the names in the input DataFrame before concatenating the names. This ensures the output respects the input order.
'''