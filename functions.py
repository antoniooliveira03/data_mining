# This file will consist of functions used through the EDA notebook

# Imports

# Feature Engineering
def avg_hour(row):
    total_orders = row.sum()
    
    if total_orders == 0:
        return None  
    
    weighted_sum_hours = (row.index.str.replace('HR_', '').astype(int) * row).sum()
    return weighted_sum_hours / total_orders