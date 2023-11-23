from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt
import math
import seaborn as sns

class eda(metaclass=ABCMeta):

    """
    Abstract base class for exploratory data analysis
    """

    def __init__(self, name):
        self.name = name
        
        @abstractmethod
        def plot_distribution(self):
            return NotImplementedError
        
        @abstractmethod
        def calculate_unique_counts(self):
            return NotImplementedError
        
        @abstractmethod
        def plot_correlation_heatmap(self):
            return NotImplementedError
        
        @abstractmethod
        def plot_correlation_heatmap(self):
            return NotImplementedError
        
        @abstractmethod
        def plot_top_items(self):
            return NotImplementedError



class hist(eda):

    """
    Class for plot histograms of numerical variables
        
    Args:
        data (pd.DataFrame): The DataFrame to be processed
        variables: The variable(s) to hist  
        bins=10: bins of the plot, set to 10 
        plots_per_row=2: number of plots per row displayed, set to 2    
    """

    def __init__(self, data, variables, bins=10, plots_per_row=2):
        self.data = data
        self.variables = variables
        self.bins = bins
        self.plots_per_row = plots_per_row

        """
        Function to  histograms of numerical variables

        Returns:
            displays histograms
        """

    def plot_distribution(self):
        
        num_variables = len(self.variables)
        num_rows = math.ceil(num_variables / self.plots_per_row)
        
        fig, axes = plt.subplots(num_rows, self.plots_per_row, figsize=(12, 4 * num_rows))
        fig.subplots_adjust(hspace=0.5)
        
        for i, variable in enumerate(self.variables):
            row = i // self.plots_per_row
            col = i % self.plots_per_row
            ax = axes[row, col] if num_rows > 1 else axes[col]
            
            ax.hist(self.data[variable], bins=self.bins, density=True, edgecolor='black', alpha=0.7)
            ax.set_title(f'Density Histogram of {variable}')
            ax.set_xlabel(variable)
            ax.set_ylabel('Density')

        plt.show()

class unique_count(eda):

    """
    Class for calculating a number of unique values
        
    Args:
        data (pd.DataFrame): The DataFrame to be processed
        columns: The column(s) to calculate unique values  
    """

    def __init__(self, data, columns):
        self.data = data
        self.columns = columns

        """
        Function to calculate the number of unique values for given columns

        Returns: counts of unique values for given columns
        """

    def calculate_unique_counts(self):
        unique_counts = {}
        for column in self.columns:
            unique_count = self.data[column].nunique()
            unique_counts[column] = unique_count
        for column, count in unique_counts.items():
            print(f"Number of unique values in {column}: {count}")
        return unique_counts


class correlations(eda):

    """
    Class for creating correlation matrix
        
    Args:
    - data (DataFrame): The DataFrame containing the data.
    - variables (list): The list of variables to include in the heatmap.
    - threshold (float): optional (default=0.1). The threshold for including correlations in the heatmap.
    """

    def __init__(self, data, variables, threshold=0.1):
        self.data = data
        self.variables = variables
        self.threshold = threshold

        """
        Function to generate a correlation heatmap for variables with absolute correlation values >= threshold.

        Returns: cdisplays correlation matrix.
        """

    def plot_correlation_heatmap(self):
        
       # Extract the relevant subset of the DataFrame
       df_for_heatmap = self.data[self.variables]
    
       # Calculate the correlation matrix
       df_for_heatmap_corr = df_for_heatmap.corr()

       # Filter correlations with abs(correlation) >= threshold
       mask = (df_for_heatmap_corr.abs() >= self.threshold)
       df_for_heatmap_corr_filtered = df_for_heatmap_corr[mask]

       # Plot the correlation heatmap
       fig, ax = plt.subplots(figsize=(10, 8))
       ax = sns.heatmap(df_for_heatmap_corr_filtered, annot=True, cmap='coolwarm', fmt=".2f")
       plt.title(f'Correlation Matrix - Absolute Values >= {self.threshold}')
       plt.show()


class bar_plot_grouped(eda):

    """
    Class for bar plotting, grouped by one of the columns if the DataFrame.
        
    Args:
    - data (DataFrame): The DataFrame containing the data.
    - group_column (str): The column by which to group the DataFrame.
    - target_column (str): The column for which to apply the summary function.
    - summary_function (str): The summary function to apply ('sum', 'mean', 'median', etc.).
    - top_n (int): optional (default=10). The number of top results to display in the bar plot.
    """

    def __init__(self, data, group_column, target_column, summary_function, top_n = 10):
        self.data = data
        self.group_column = group_column
        self.target_column = target_column
        self.summary_function = summary_function
        self.top_n = top_n

        """
        Function to group by a column, apply a summary function to another column, then plot the top N results.

        Raises Error: if it's invalid summary function.

        Returns: displays the bar plot.
        """

    def plot_correlation_heatmap(self):
        # Validate summary function
        if self.summary_function not in ['sum', 'mean', 'median', 'mode']:
            raise ValueError(f"Invalid summary function. Supported functions: 'sum', 'mean', 'median', 'mode'.")

        # Group by the specified column and apply the summary function to the target column
        grouped = self.data.groupby(self.group_column)[self.target_column].agg(self.summary_function).reset_index()

        # Sort the DataFrame by the summary value in descending order
        grouped = grouped.sort_values(by=self.target_column, ascending=False)

        # Select the top N categories
        top_categories = grouped.head(self.top_n)

        # Create a bar plot
        plt.figure(figsize=(8, 6))
        plt.bar(top_categories[self.group_column], top_categories[self.target_column])
        plt.xlabel(self.group_column.capitalize())  # Capitalize the column name for x-axis label
        plt.ylabel(f"{self.summary_function.capitalize()} {self.target_column.capitalize()}")  # Label with summary function
        plt.title(f'Top {self.top_n} {self.group_column.capitalize()}s by {self.summary_function.capitalize()} {self.target_column.capitalize()}')
    
        # Set smaller x-tick font size and rotate
        plt.xticks(rotation=90, fontsize=8)

        # Show the plot
        plt.show()

class plot_top(eda):

    """
    Class for bar plotting, mentioning two columns of the DataFrame as x-ticks.
        
    Args:
    - data (DataFrame): The DataFrame containing the data.
    - top_n (int): optional (default=10). The number of top items to display in the bar plot.
    """

    def __init__(self, data, top_n = 10):
        self.data = data
        self.top_n = top_n
        self.columns = {}

        """
        Function or bar plotting, mentioning two columns of the DataFrame as xticks.

        Args:
        - *columns (variable length arguments): The columns to use for track, artist, popularity, etc. (in order).

        Returns: displays the bar plot.
        """

    def plot_top_items(self, *columns):
        # Sort the DataFrame by total popularity in descending order
        sorted_df = self.data.sort_values(by=columns[-1], ascending=False)

        # Select the top N most popular items
        top_items = sorted_df.head(self.top_n)

        # Combine column values for x-tick labels
        item_labels = [' - '.join(map(str, item)) for item in zip(*[top_items[col] for col in columns[:-1]])]

        # Create a bar plot
        plt.figure(figsize=(8, 6))
        plt.bar(item_labels, top_items[columns[-1]])
        plt.xlabel(' - '.join(map(str, columns[:-1])))
        plt.ylabel(columns[-1].capitalize())  # Capitalize the last column name for y-axis label
        plt.title(f'Top {top_n} Most Popular {columns[-1].capitalize()}s')

        # Set smaller x-tick font size and rotate
        plt.xticks(rotation=90, fontsize=8)

        # Show the plot
        plt.show()
        