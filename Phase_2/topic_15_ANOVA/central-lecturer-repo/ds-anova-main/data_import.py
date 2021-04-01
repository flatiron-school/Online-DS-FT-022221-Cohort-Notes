import pandas as pd 

# Prep divy trips

def prep_divy(file_path):
    
    """
    Takes raw csv data downloaded from Chicago Data Portal and preps it for analysis. 
    
    Specifically, it converts right times to datetime objects, then creates columns for month, day of the week
    and hour.
    
    """
    divy_trips = pd.read_csv(file_path)
    divy_trips['started_at'] = pd.to_datetime(divy_trips['started_at'])
    divy_trips['ended_at'] = pd.to_datetime(divy_trips['ended_at'])
    
    # Create ride time feature for the purpose of plotting a continous variable
    divy_trips['ride_time'] = divy_trips['ended_at'] - divy_trips['started_at']
    divy_trips['ride_time'] = divy_trips['ride_time'].apply(lambda x: x.seconds)
    
    divy_trips['weekday'] = divy_trips['started_at'].apply(lambda x: x.isoweekday())
    divy_trips['hour'] = divy_trips['started_at'].apply(lambda x: x.hour)

    return divy_trips

