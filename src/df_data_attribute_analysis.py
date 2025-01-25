import pandas as pd

import pandas as pd

def categorize_columns(df):
    # Initialize arrays to hold column names
    categories = {
        'categorical': [],
        'numerical': [],
        'miscellaneous': []
    }

    # Iterate through each column and categorize based on its dtype
    for column in df.columns:
        dtype = df[column].dtype

        if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):  # Treat booleans as categorical
            categories['categorical'].append(column)
        elif pd.api.types.is_float_dtype(dtype):
            categories['numerical'].append(column)
        else:
            categories['miscellaneous'].append(column)

    return categories


# def categorize_columns(df):
#     # Initialize arrays to hold column names
#     categories = {
#         'categorical': [],
#         'numerical': [],
#         'boolean': [],
#         'miscellaneous': []
#     }

#     # Iterate through each column and categorize based on its dtype
#     for column in df.columns:
#         dtype = df[column].dtype

#         if pd.api.types.is_object_dtype(dtype):
#             categories['categorical'].append(column)
#         elif pd.api.types.is_float_dtype(dtype):
#             categories['numerical'].append(column)
#         elif pd.api.types.is_bool_dtype(dtype):
#             categories['boolean'].append(column)
#         else:
#             categories['miscellaneous'].append(column)

#     return categories
