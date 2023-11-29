from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt
import math
import seaborn as sns

"""
Function to  histograms of numerical variables

Args:
- data (pd.DataFrame): The DataFrame to be processed
- variables: The variable(s) to hist  
- bins=10: bins of the plot, set to 10 
- plots_per_row=2: number of plots per row displayed, set to 2

Returns:
displays histograms
"""

def plot_distribution(data, variables, bins=10, plots_per_row=2):
        
        num_variables = len(variables)
        num_rows = math.ceil(num_variables / plots_per_row)
        
        fig, axes = plt.subplots(num_rows, plots_per_row, figsize=(12, 4 * num_rows))
        fig.subplots_adjust(hspace=0.5)
        
        for i, variable in enumerate(variables):
            row = i // plots_per_row
            col = i % plots_per_row
            ax = axes[row, col] if num_rows > 1 else axes[col]
            
            ax.hist(data[variable], bins=bins, density=True, edgecolor='black', alpha=0.7)
            ax.set_title(f'Density Histogram of {variable}')
            ax.set_xlabel(variable)
            ax.set_ylabel('Density')

        plt.show()    


"""
Function to calculate the number of unique values for given columns

Args:
data (pd.DataFrame): The DataFrame to be processed
columns: The column(s) to calculate unique values 

Returns: counts of unique values for given columns
"""

def calculate_unique_counts(data, columns):
        unique_counts = {}
        for column in columns:
            unique_count = data[column].nunique()
            unique_counts[column] = unique_count
        for column, count in unique_counts.items():
            print(f"Number of unique values in {column}: {count}")
        return unique_counts    


"""
Function to generate a correlation heatmap for variables with absolute correlation values >= threshold.

Args:
- data (DataFrame): The DataFrame containing the data.
- variables (list): The list of variables to include in the heatmap.
- threshold (float): optional (default=0.1). The threshold for including correlations in the heatmap.

Returns: cdisplays correlation matrix.
"""

def plot_correlation_heatmap(data, variables, threshold=0.1):
        
       # Extract the relevant subset of the DataFrame
       df_for_heatmap = data[variables]
    
       # Calculate the correlation matrix
       df_for_heatmap_corr = df_for_heatmap.corr()

       # Filter correlations with abs(correlation) >= threshold
       mask = (df_for_heatmap_corr.abs() >= threshold)
       df_for_heatmap_corr_filtered = df_for_heatmap_corr[mask]

       # Plot the correlation heatmap
       fig, ax = plt.subplots(figsize=(10, 8))
       ax = sns.heatmap(df_for_heatmap_corr_filtered, annot=True, cmap='coolwarm', fmt=".2f")
       plt.title(f'Correlation Matrix - Absolute Values >= {threshold}')
       plt.show()    



"""
Function to group by a column, apply a summary function to another column, then plot the top N results.

Args:
- data (DataFrame): The DataFrame containing the data.
- group_column (str): The column by which to group the DataFrame.
- target_column (str): The column for which to apply the summary function.
- summary_function (str): The summary function to apply ('sum', 'mean', 'median', etc.).
- top_n (int): optional (default=10). The number of top results to display in the bar plot.

Raises Error: if it's invalid summary function.

Returns: displays the bar plot.
"""

def bar_plot(data, group_column, target_column, summary_function, top_n = 10):
        # Validate summary function
        if summary_function not in ['sum', 'mean', 'median', 'mode']:
            raise ValueError(f"Invalid summary function. Supported functions: 'sum', 'mean', 'median', 'mode'.")

        # Group by the specified column and apply the summary function to the target column
        grouped = data.groupby(group_column)[target_column].agg(summary_function).reset_index()

        # Sort the DataFrame by the summary value in descending order
        grouped = grouped.sort_values(by=target_column, ascending=False)

        # Select the top N categories
        top_categories = grouped.head(top_n)

        # Create a bar plot
        plt.figure(figsize=(8, 6))
        plt.bar(top_categories[group_column], top_categories[target_column])
        plt.xlabel(group_column.capitalize())  # Capitalize the column name for x-axis label
        plt.ylabel(f"{summary_function.capitalize()} {target_column.capitalize()}")  # Label with summary function
        plt.title(f'Top {top_n} {group_column.capitalize()}s by {summary_function.capitalize()} {target_column.capitalize()}')
    
        # Set smaller x-tick font size and rotate
        plt.xticks(rotation=90, fontsize=8)

        # Show the plot
        plt.show()    


"""
Function or bar plotting, mentioning two columns of the DataFrame as xticks.

Args:
- data (DataFrame): The DataFrame containing the data.
- top_n (int): optional (default=10). The number of top items to display in the bar plot.
- *columns (variable length arguments): The columns to use for track, artist, popularity, etc. (in order).

Returns: displays the bar plot.
"""

def plot_top_items(data, top_n = 10, *columns):
        # Sort the DataFrame by total popularity in descending order
        sorted_df = data.sort_values(by=columns[-1], ascending=False)

        # Select the top N most popular items
        top_items = sorted_df.head(top_n)

        # Combine column values for x-tick labels
        item_labels = [' - '.join(map(str, item)) for item in zip(*[top_items[col] for col in columns[:-1]])]

        # Create a bar plot
        plt.figure(figsize=(8, 6))
        plt.bar(item_labels, top_items[columns[-1]])
        plt.xlabel(' - '.join(map(str, columns[:-1])))
        plt.ylabel(columns[-1].capitalize())  # Capitalize the last column name for y-axis label
        plt.title(f'Top {top_n} Most Popular {columns[-3].capitalize()}s')

        # Set smaller x-tick font size and rotate
        plt.xticks(rotation=90, fontsize=8)

        # Show the plot
        plt.show()    
        

def looking_for_outliers(data):
    numerical_columns = [col for col in data.select_dtypes(include='number').columns]
    plt.figure(figsize=(15, 8))

    sns.boxplot(data=data[numerical_columns], orient="h", palette="Set2")
    plt.title("Boxplots")
    plt.xlabel("Value")

    plt.tight_layout()

    # Show the plot
    plt.show()