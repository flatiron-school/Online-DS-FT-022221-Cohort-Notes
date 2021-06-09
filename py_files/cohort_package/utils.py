import pandas as pd
import sys
def check_var_size(locals_):
    """checks the variables in memory and returns a series 
    of var names and sizes, sorted by size. 
    
    ## Example usage
    >> var_size = check_var_size(locals())
    >> var_size.head(20)
    """
    var_sizes ={}
    
    for var, obj in locals_.items():
        if var.startswith('_') == False:
            var_sizes[var] = sys.getsizeof(obj)

    return pd.Series(var_sizes).sort_values(ascending=False)