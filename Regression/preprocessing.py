import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly

la_crime = pd.read_csv("los_angeles_crime_preprocessed.csv")
la_traffic = pd.read_csv("traffic_collison_preprocessed.csv")

# Define crime categories

crime_categories = {
    'Theft': [
        'THEFT PLAIN - PETTY ($950 & UNDER)',
        'THEFT-GRAND ($950.01 & OVER)EXCPT,GUNS,FOWL,LIVESTK,PROD',
        'THEFT FROM MOTOR VEHICLE - GRAND ($950.01 AND OVER)'
        'SHOPLIFTING-GRAND THEFT ($950.01 & OVER)',
        'SHOPLIFTING - PETTY THEFT ($950 & UNDER)',
        'BUNCO, GRAND THEFT',
        'THEFT FROM MOTOR VEHICLE - PETTY ($950 & UNDER)',
        'THEFT OF IDENTITY',
        'THEFT PLAIN - ATTEMPT','THEFT, PERSON',
        'EMBEZZLEMENT, GRAND THEFT ($950.01 & OVER)',
        'THEFT FROM MOTOR VEHICLE - ATTEMPT',
        'SHOPLIFTING - ATTEMPT','BIKE - STOLEN',
        'DEFRAUDING INNKEEPER/THEFT OF SERVICES, $950 & UNDER',
        'DEFRAUDING INNKEEPER/THEFT OF SERVICES, OVER $950.01',
        'DISHONEST EMPLOYEE - PETTY THEFT',
        'DISHONEST EMPLOYEE - GRAND THEFT',
        'THEFT, COIN MACHINE - PETTY ($950 & UNDER)',
        'BOAT - STOLEN','ROBBERY','BURGLARY','BURGLARY FROM VEHICLE','BURGLARY, ATTEMPTED',
        'ATTEMPTED ROBBERY','PICKPOCKET',
        'VEHICLE, STOLEN - OTHER (MOTORIZED SCOOTERS, BIKES, ETC)',
        'DISHONEST EMPLOYEE - GRAND THEFT',
        'EMBEZZLEMENT, PETTY THEFT ($950 & UNDER)',
        'THEFT FROM PERSON - ATTEMPT',
        'THEFT FROM MOTOR VEHICLE - ATTEMPT',
        'BUNCO, ATTEMPT', 'PURSE SNATCHING',
        'BURGLARY','TRESPASSING',
        'BURGLARY FROM VEHICLE',
        'BURGLARY FROM VEHICLE, ATTEMPTED',
        'VEHICLE - STOLEN',
        'VEHICLE, STOLEN - OTHER (MOTORIZED SCOOTERS, BIKES, ETC)',
        'VEHICLE - ATTEMPT STOLEN',
        'VANDALISM - FELONY ($400 & OVER, ALL CHURCH VANDALISMS)',
        'VANDALISM - MISDEAMEANOR ($399 OR UNDER)',
        'DRIVING WITHOUT OWNER CONSENT (DWOC)',
         'BOAT - STOLEN',
    ],
    'Sexual Offenses': [
        'RAPE, FORCIBLE',
        'SEXUAL PENETRATION W/FOREIGN OBJECT',
        'LEWD/LASCIVIOUS ACTS WITH CHILD',
        'ORAL COPULATION',
        'HUMAN TRAFFICKING - COMMERCIAL SEX ACTS',
        'CHILD ABUSE (PHYSICAL) - SIMPLE ASSAULT',
        'CHILD ABUSE (PHYSICAL) - AGGRAVATED ASSAULT',
        'PANDERING','CHILD PORNOGRAPHY',
        'LEWD CONDUCT', 'PIMPING',
        'RAPE, ATTEMPTED',
        'CHILD ANNOYING (17YRS & UNDER)',
        'SEX,UNLAWFUL(INC MUTUAL CONSENT, PENETRATION W/ FRGN OBJ',
        'SODOMY/SEXUAL CONTACT B/W PENIS OF ONE PERS TO ANUS OTH',
        'INDECENT EXPOSURE','BATTERY WITH SEXUAL CONTACT',
    ],
    'Violence': [
        'ASSAULT WITH DEADLY WEAPON, AGGRAVATED ASSAULT',
        'BATTERY - SIMPLE ASSAULT',
        'INTIMATE PARTNER - SIMPLE ASSAULT',
        'INTIMATE PARTNER - AGGRAVATED ASSAULT',
        'BATTERY WITH SEXUAL CONTACT',
        'ASSAULT WITH DEADLY WEAPON ON POLICE OFFICER',
        'KIDNAPPING',
        'CRIMINAL HOMICIDE',
        'ATTEMPTED ROBBERY',
        'CRIMINAL THREATS - NO WEAPON DISPLAYED',
        'BRANDISH WEAPON','VIOLATION OF COURT ORDER',
        'OTHER ASSAULT','THROWING OBJECT AT MOVING VEHICLE',
        'CHILD STEALING',
        'DISCHARGE FIREARMS/SHOTS FIRED',
        'STALKING',
        'SHOTS FIRED AT MOVING VEHICLE, TRAIN OR AIRCRAFT',
        'BATTERY ON A FIREFIGHTER',
        'WEAPONS POSSESSION/BOMBING',
        'KIDNAPPING - GRAND ATTEMPT',
        'ARSON',
        'CHILD ANNOYING (17YRS & UNDER)',
        'SHOTS FIRED AT INHABITED DWELLING',
    ],
    'Financial Crimes': [
        'CREDIT CARDS, FRAUD USE ($950 & OVER)',
        'CREDIT CARDS, FRAUD USE ($950.01 & OVER)',
        'DOCUMENT FORGERY / STOLEN FELONY',
        'COUNTERFEIT',
        'EXTORTION',
        'CONTRIBUTING',
        'DOCUMENT WORTHLESS ($200.01 & OVER)',
        'EXTORTION',
        'BUNCO, PETTY THEFT',
        'CREDIT CARDS, FRAUD USE ($950 & UNDER',
       'COUNTERFEIT',
    ],
    'Threats':[
        'LETTERS, LEWD  -  TELEPHONE CALLS, LEWD',
        'THREATENING PHONE CALLS/LETTERS', 'PROWLER',
        'BOMB SCARE',
    ],
    'MISCELLANEOUS CRIME':[
        'OTHER MISCELLANEOUS CRIME','FALSE IMPRISONMENT',
        'RECKLESS DRIVING','FAILURE TO YIELD', 'CRUELTY TO ANIMALS',
         'CONTRIBUTING','CRM AGNST CHLD (13 OR UNDER) (14-15 & SUSP 10 YRS OLDER)',
         'DISTURBING THE PEACE',
        'ILLEGAL DUMPING',
    ],
    'Legal violations':[
        'VIOLATION OF TEMPORARY RESTRAINING ORDER',
        "VIOLATION OF RESTRAINING ORDER",
        'BATTERY POLICE (SIMPLE)','RESISTING ARREST',
        'CHILD ABANDONMENT',
        'FALSE POLICE REPORT', 'DISRUPT SCHOOL', 'LYNCHING',
        'DRUGS, TO A MINOR','FAILURE TO DISPERSE',
        'HUMAN TRAFFICKING - INVOLUNTARY SERVITUDE',
        'CHILD NEGLECT (SEE 300 W.I.C.)','PEEPING TOM',
        'SEX OFFENDER REGISTRANT OUT OF COMPLIANCE',
        'UNAUTHORIZED COMPUTER ACCESS',
        'CONTEMPT OF COURT',
    ],
}

# Function to categorize each row
def categorize_crime(description):
    for category, descriptions in crime_categories.items():
        if description in descriptions:
            return category
    return 'Other'  # or any other default category you wish to set for uncategorized crimes

# Apply the function to create the new column
la_crime['crime_categories'] = la_crime['Crm Cd Desc'].apply(categorize_crime)
la_crime['DATE OCC'] = pd.to_datetime(la_crime['DATE OCC'])

# Extracting 'Crime_Month', 'Crime_Day', and 'Crime_Time'
la_crime['Crime_Month'] = la_crime['DATE OCC'].dt.month
la_crime['Crime_Day_of_Week'] = la_crime['DATE OCC'].dt.dayofweek

def convert_time(t):
    # Convert to string and pad with zeros to ensure 4 characters
    t_str = str(t).zfill(4)
    # Extract hours and minutes
    hours, minutes = t_str[:2], t_str[2:]
    # Convert to time format
    return pd.to_datetime(f'{hours}:{minutes}', format='%H:%M').time()

# Apply the function to the 'TIME OCC' column
la_crime['Crime_Time'] = la_crime['TIME OCC'].apply(convert_time)

# Define the time intervals
time_intervals = {
    'T1': ('01:00:00', '04:59:59'),
    'T2': ('05:00:00', '08:59:59'),
    'T3': ('09:00:00', '12:59:59'),
    'T4': ('13:00:00', '16:59:59'),
    'T5': ('17:00:00', '20:59:59'),
    'T6': ('21:00:00', '00:59:59'),
}

# Function to categorize the crime times
def categorize_time(crime_time):
    for category, (start_time, end_time) in time_intervals.items():
        if start_time <= str(crime_time) <= end_time or (category == 'T6' and (str(crime_time) <= end_time or str(crime_time) >= start_time)):
            return category
    return 'Uncategorized'  # In case the time does not fall into any category

# Apply the function to the 'Crime_Time' column to create a new 'Time_Category' column
la_crime['Time_Category'] = la_crime['Crime_Time'].apply(categorize_time)
la_crime['Time_Category']

# Create a mapping of crime categories to unique IDs
category_to_id = {
    'Violence': 1,
    'Other': 2,
    'MISCELLANEOUS CRIME': 3,
    'Sexual Offenses': 4,
    'Theft': 5,
    'Threats': 6,
    'Legal violations': 7,
    'Financial Crimes': 8
}

# Apply the mapping to the 'crime_categories' column to create a new 'crime_category_ID' column
la_crime['crime_category_ID'] = la_crime['crime_categories'].map(category_to_id)

def convert_time(t):
    # Convert to string and pad with zeros to ensure 4 characters
    t_str = str(t).zfill(4)
    # Extract hours and minutes
    hours, minutes = t_str[:2], t_str[2:]
    # Convert to time format
    return pd.to_datetime(f'{hours}:{minutes}', format='%H:%M').time()

# Apply the function to the 'TIME OCC' column
la_traffic['Crime_Time'] = la_traffic['TIME OCC'].apply(convert_time)

# Define the time intervals
time_intervals = {
    'T1': ('01:00:00', '04:59:59'),
    'T2': ('05:00:00', '08:59:59'),
    'T3': ('09:00:00', '12:59:59'),
    'T4': ('13:00:00', '16:59:59'),
    'T5': ('17:00:00', '20:59:59'),
    'T6': ('21:00:00', '00:59:59'),
}

# Function to categorize the crime times
def categorize_time(crime_time):
    for category, (start_time, end_time) in time_intervals.items():
        if start_time <= str(crime_time) <= end_time or (category == 'T6' and (str(crime_time) <= end_time or str(crime_time) >= start_time)):
            return category
    return 'Uncategorized'  # In case the time does not fall into any category

# Apply the function to the 'Crime_Time' column to create a new 'Time_Category' column
la_traffic['Time_Category'] = la_traffic['Crime_Time'].apply(categorize_time)
la_traffic['Time_Category']
# Mapping from integers to day names
day_mapping = {
    0: 'Monday',
    1: 'Tuesday',
    2: 'Wednesday',
    3: 'Thursday',
    4: 'Friday',
    5: 'Saturday',
    6: 'Sunday'
}

# Apply the mapping to create the 'Day' column
la_traffic['Day'] = la_traffic['Day of Week'].map(day_mapping)

# Convert the "DATE OCC" column to datetime format
la_crime['DATE OCC'] = pd.to_datetime(la_crime['DATE OCC'])

# Create a new column "DATE_OCC_DATE" with only the date
la_crime['DATE_OCC_DATE'] = la_crime['DATE OCC'].dt.strftime('%m/%d/%Y')

# Now, you can drop the renamed 'DATE OCC Duplicate' column
la_crime = la_crime.drop('DATE OCC', axis=1)

la_crime = la_crime.rename(columns={'DATE_OCC_DATE': 'DATE OCC',
                                   })

sample_fraction = 0.02  
sample_fraction1 = 0.05

# Take a random sample from each DataFrame
la_features_sampled = la_features.sample(frac=sample_fraction, random_state=42)
la_traffic_sampled = la_traffic.sample(frac=sample_fraction1, random_state=42)

la_combined["Crime Code Description"].fillna("No Traffic Collision", inplace=True)


la_combined = la_combined.rename(columns={'Is Holiday_x': 'IS Holiday',
                                        'TIME OCC_x':'TIME OCC',
                                         'Crime_Location_x':'Crime_Location',
                                        'Day of Week_x':'Day of Week',
                                         'Time_Category_x':'Time_Category'})



# Mapping for ordinal encoding
traffic_mapping = {
    'TRAFFIC COLLISION': 0,
    'No Traffic Collision': 1,

}

# Apply the mapping to the 'Day' column
la_combined['Crime Code Description'] = la_combined['Crime Code Description'].map(traffic_mapping)

# Mapping for ordinal encoding
holiday_mapping = {
    'No': 0,
    'Yes': 1,

}

# Apply the mapping to the 'Day' column
la_combined['IS Holiday'] = la_combined['IS Holiday'].map(holiday_mapping)

la_combined['DATE_DUMMY'] = la_combined['DATE OCC'].copy()

# Extract the year from 'DATE OCC' and create a new 'Year' column
la_combined['DATE_DUMMY'] = pd.to_datetime(la_combined['DATE_DUMMY'])
la_combined['Year'] = la_combined['DATE_DUMMY'].dt.year

# Convert 'Time_Category' from T1-T6 to 1-6
time_category_mapping = {'T1': 1, 'T2': 2, 'T3': 3, 'T4': 4, 'T5': 5, 'T6': 6}
la_combined['Time_Category'] = la_combined['Time_Category'].map(time_category_mapping)

from sklearn.preprocessing import LabelEncoder


# Assuming la_combined is your DataFrame and 'Crime_Location' is the column you want to encode
label_encoder = LabelEncoder()
la_combined['Crime_Location'] = label_encoder.fit_transform(la_combined['Crime_Location'])

# Group by 'Area ID' and 'Crime_Month', then count the number of crimes
crime_rate = la_data_selected.groupby(['Area ID', 'Crime_Month']).size().reset_index(name='Monthly_Crime_Rate')

# Merge the crime rate data into the main dataset
la_data_selected = la_data_selected.merge(crime_rate, on=['Area ID', 'Crime_Month'], how='left')

features_1= ['TIME OCC','Crime_Location','crime_category_ID','Crime_Month','Day of Week','Time_Category','IS Holiday','Area ID','Crime Code Description','Year']
la_dt = la_combined[features_1]


la_dt.to_csv('Clean_data_regression.csv')
