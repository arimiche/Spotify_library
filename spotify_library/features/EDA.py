from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt
import math

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
        # # Print the results
        for column, count in unique_counts.items():
            print(f"Number of unique values in {column}: {count}")
            return unique_counts

        