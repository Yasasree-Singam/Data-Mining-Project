import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
import numpy as np

def clean_data():
    # Load crime and traffic data
    # crime = pd.read_csv("Myfiles/DS_Sem3/CSE 881/Project/los_angeles_crime_data.csv")
    # traffic = pd.read_csv("Myfiles/DS_Sem3/CSE 881/Project/Traffic_Collision_Data_from_2010_to_Present.csv")
   crime_data_url = "https://media.githubusercontent.com/media/Yasasree-Singam/Data-Mining-Project/main/Myfiles/DS_Sem3/CSE%20881/Project/los_Angeles_Crimedata%202010-2023.csv"
   traffic_data_url = "https://media.githubusercontent.com/media/Yasasree-Singam/Data-Mining-Project/main/Myfiles/DS_Sem3/CSE%20881/Project/Traffic_Collision_Data_from_2010_to_Present.csv"
    # Reading the CSV files
    crime = pd.read_csv(crime_data_url)
    traffic = pd.read_csv(traffic_data_url)
    # Combine 'AREA' and 'AREA ' columns
    # crime['COMBINED_AREA'] = crime['AREA'].combine_first(crime['AREA '])

    # # Drop unnecessary columns in the crime dataset
    # columns_to_remove = ['Crm Cd 1', 'Crm Cd 2', 'Crm Cd 3', 'Crm Cd 4', 'DR_NO', 'Date Rptd', 'AREA', 'Status Desc', 'Status', 'AREA ', 'Weapon Desc', 'Weapon Used Cd', 'Mocodes', 'Vict Age', 'Vict Sex', 'Vict Descent', 'Premis Desc', 'Part 1-2', 'Premis Cd', 'Cross Street']
    # df1 = crime.drop(columns=columns_to_remove)

    # Create a new column "DATE_OCC_DATE" with only the date
    crime['DATE_OCC_DATE'] = pd.to_datetime(crime['DATE OCC']).dt.strftime('%m/%d/%Y')
    # columns_to_remove = ['DATE OCC']
    # Drop the specified columns
    df2 = crime.drop(['DATE OCC'], axis =1)
    # Rename the "DATE_OCC_DATE" column to "DATE OCC"
    df3 = df2.rename(columns={'DATE_OCC_DATE': 'DATE OCC'})


    df3['TIME OCC'] = df3['TIME OCC'].astype(str).str.zfill(4)
    df3['Hour'] = df3['TIME OCC'].str[:2]

    # Map hours to time categories
    time_categories = {'00': 'Midnight', '01': 'Midnight', '02': 'Midnight', '03': 'Midnight', '04': 'Morning', '05': 'Morning', '06': 'Morning', '07': 'Morning', '08': 'Morning', '09': 'Morning', '10': 'Morning', '11': 'Morning', '12': 'Afternoon', '13': 'Afternoon', '14': 'Afternoon', '15': 'Afternoon', '16': 'Afternoon', '17': 'Evening', '18': 'Evening', '19': 'Evening', '20': 'Evening', '21': 'Evening', '22': 'Evening', '23': 'Evening'}
    df3['Time Category'] = df3['Hour'].map(time_categories)

    # Optional: Convert time categories to numeric values
    time_category_mapping = {'Midnight': 0, 'Morning': 1, 'Afternoon': 2, 'Evening': 3}
    df3['Time Category Numeric'] = df3['Time Category'].map(time_category_mapping)

    # Add holiday and day-related features
    df4 = df3.copy()
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=df4['DATE OCC'].min(), end=df4['DATE OCC'].max()) 
    df4['Is Holiday'] = df4['DATE OCC'].apply(lambda x: x in holidays)
    # df4['Day of Week'] = pd.to_datetime(df4['DATE OCC']).dt.dayofweek
    df4['Day of Week'] = pd.to_datetime(df4['DATE OCC'], format='%m/%d/%Y').dt.dayofweek
    df4['Day Type'] = df4['Day of Week'].map({0: 'Weekday', 1: 'Weekday', 2: 'Weekday', 3: 'Weekday', 4: 'Weekday', 5: 'Weekend', 6: 'Weekend'})
    # df4 = df4.drop(columns=['DATE_OCC'])

    # Convert True to 'Yes' and False to 'No' in the 'Is Holiday' column
    df4['Is Holiday'] = df4['Is Holiday'].replace({True: 'Yes', False: 'No'})

    # Drop columns that may not be useful in traffic dataset
    columns_to_drop = ['DR Number', 'MO Codes', 'Victim Age', 'Victim Sex', 'Victim Descent', 'Date Reported', 'Cross Street', 'Location', 'Premise Code', 'Address', 'Premise Description', 'Reporting District', 'Crime Code']
    traffic_cleaned = traffic.drop(columns=columns_to_drop).dropna()

    # Convert 'Area ID' to float in traffic dataset
    # traffic_cleaned['Area ID'] = traffic_cleaned['Area ID'].astype(float)

    # Preprocess time-related features in traffic dataset
    traffic_cleaned['Time Occurred'] = traffic_cleaned['Time Occurred'].astype(str).str.zfill(4)
    traffic_cleaned['Hour'] = traffic_cleaned['Time Occurred'].str[:2]
    traffic_cleaned['Time Category'] = traffic_cleaned['Hour'].map(time_categories)

    # Optional: Convert time categories to numeric values in traffic dataset
    traffic_cleaned['Time Category Numeric'] = traffic_cleaned['Time Category'].map(time_category_mapping)

    # Add holiday and day-related features in traffic dataset
    tf = traffic_cleaned.copy()
    holidays = cal.holidays(start=tf['Date Occurred'].min(), end=tf['Date Occurred'].max()).to_pydatetime()
    tf['Is Holiday'] = tf['Date Occurred'].apply(lambda x: x in holidays)
    tf['Day of Week'] = pd.to_datetime(tf['Date Occurred']).dt.dayofweek
    tf['Day Type'] = tf['Day of Week'].map({0: 'Weekday', 1: 'Weekday', 2: 'Weekday', 3: 'Weekday', 4: 'Weekday', 5: 'Weekend', 6: 'Weekend'})
    tf = tf.rename(columns={'Date Occurred': 'DATE OCC', 'Time Occurred': 'TIME OCC'})
    df4 = df4.rename(columns={'COMBINED_AREA': 'Area ID'})
    # Define crime categories
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

    # Map crime descriptions to crime categories
    df4['Crime Category'] = df4['Crm Cd Desc'].map(lambda x: next((category for category, crimes in crime_categories.items() if x in crimes), None))

    # Drop the original crime description column
    df4 = df4.drop(["Crm Cd Desc"], axis=1)

    # Sample a fraction of the datasets for efficient processing
    sample_fraction = 0.01  # Adjust as needed
    sample_fraction1 = 0.05

    # Take a random sample from each DataFrame
    df4_sampled = df4.sample(frac=sample_fraction, random_state=42)
    tf_sampled = tf.sample(frac=sample_fraction1, random_state=42)

    # Merge on both "DATE OCC" and "TIME OCC"
    columns_to_merge = ['Area ID', 'DATE OCC', 'Time Category', 'Day Type']
    merged_data = pd.merge(df4_sampled, tf_sampled, how='left', on=columns_to_merge)

    # Fill missing values in "Crime Code Description" with "No Traffic Collision"
    merged_data["Crime Code Description"].fillna("No Traffic Collision", inplace=True)

    # Drop unnecessary columns
    columns_to_drop = ['TIME OCC_y', 'AREA NAME', 'Rpt Dist No', 'Time Category Numeric_x', 'Crm Cd', 'Day of Week_x', 'Day of Week_y', 'Is Holiday_x', 'Is Holiday_y', 'Time Category Numeric_y', 'Hour_y', 'Area Name', 'Hour_x']
    cleaned_data = merged_data.drop(columns=columns_to_drop)

    # Rename columns for consistency
    cleaned_data = cleaned_data.rename(columns={'TIME OCC_x': 'TIME OCC'})

    # Create a balanced dataset with 'No Crime' label
    data_copy = cleaned_data.copy()
    no_crime_data = data_copy.dropna(subset=['Crime Category']).copy()
    shuffled_dates = np.random.permutation(no_crime_data['DATE OCC'])
    no_crime_data['DATE OCC'] = shuffled_dates
    no_crime_data['Crime Category'] = 'No Crime'
    balanced_data = pd.concat([data_copy, no_crime_data])

    # Convert "DATE OCC" to datetime format for holiday check
    balanced_data['DATE_OCC'] = pd.to_datetime(balanced_data['DATE OCC'])

    # Check if each date is a U.S. federal holiday
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=balanced_data['DATE_OCC'].min(), end=balanced_data['DATE_OCC'].max()).to_pydatetime()
    balanced_data['Is Holiday'] = balanced_data['DATE_OCC'].apply(lambda x: x in holidays)
    data_balance = balanced_data.copy()
    data_balance['Crime Category'] = data_balance['Crime Category'].apply(lambda x: 'Crime' if x != 'No Crime' else 'No Crime')
    # Convert 'DATE OCC' to datetime
    data_balance['DATE OCC'] = pd.to_datetime(data_balance['DATE OCC'])

    # Extract year, month, and day
    data_balance['Year'] = data_balance['DATE OCC'].dt.year
    data_balance['Month'] = data_balance['DATE OCC'].dt.month
    data_balance['Day'] = data_balance['DATE OCC'].dt.day

    return data_balance, crime
