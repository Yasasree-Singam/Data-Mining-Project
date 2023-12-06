import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster










def page1():
    st.title("Los Angeles Crime")
    st.write(" Regression, Classification, Association rule generation")
    data = pd.read_csv("C:/Users/singa/Myfiles/DS_Sem3/CSE 881/Project/los_angeles_crime_data.csv")
    st.write(data.describe())
    # Extract year from 'DATE OCC' column
    data['Year'] = pd.to_datetime(data['DATE OCC']).dt.year

    # Streamlit app
    st.title("Distribution of Crimes Over Different Years")

    # Plot the histogram using Plotly Express
    fig1 = px.histogram(data, x='Year', nbins=14, title="Distribution of Crimes Over Different Years")

    # Customize the layout
    fig1.update_layout(
        xaxis_title="Year",
        yaxis_title="Number of Crimes",
        xaxis=dict(tickvals=list(range(2010, 2024))),
        bargap=0.1,
    )

    # Display the plot
    st.plotly_chart(fig1)

    # Create a pie chart using Plotly Express
    fig2= px.pie(data['AREA NAME'].value_counts(), 
                names=data['AREA NAME'].value_counts().index,
                values=data['AREA NAME'].value_counts().values,
                title="Distribution of Crimes in Different Areas",
                hole=0.3,  # Set to 0 for a traditional pie chart
                color_discrete_sequence=px.colors.qualitative.Set3
                )

    # Customize the layout
    fig2.update_layout(
        legend=dict(title="Area Name"),
    )

    # Display the pie chart
    st.plotly_chart(fig2)

    st.write("Preprocessing")
    columns_to_remove = ['Crm Cd 1', 'Crm Cd 2', 'Crm Cd 3', 'Crm Cd 4', 'DR_NO','Date Rptd','AREA','Status Desc','Status','AREA ','Weapon Desc','Weapon Used Cd',
                   'Mocodes','Vict Age','Vict Sex','Vict Descent','Premis Desc', 'Part 1-2','Premis Cd','Cross Street']

    # Drop the specified columns
    df1 = data.drop(columns=columns_to_remove)


    df1['DATE OCC'] = pd.to_datetime(df1['DATE OCC'])

    # Create a new column "DATE_OCC_DATE" with only the date
    df1['DATE_OCC_DATE'] = df1['DATE OCC'].dt.strftime('%m/%d/%Y')

    df1.drop('DATE OCC',axis=1)

    df1.rename(columns={'DATE_OCC_DATE': 'DATE OCC'})

    import geopandas as gpd
    import folium
    from streamlit_folium import folium_static

    # Load your crime data
    # Replace 'your_crime_data.csv' with the actual filename or URL of your crime data
    # crime_data = pd.read_csv('your_crime_data.csv')

    # Create a GeoDataFrame from the crime data (assuming you have latitude and longitude columns)
    geometry = gpd.points_from_xy(data['LON'], data['LAT'])
    gdf = gpd.GeoDataFrame(data, geometry=geometry)

    # Streamlit App
    st.title("Crime Hotspots Map")

    # Display the raw data (optional)
    st.write("Raw Crime Data", gdf)

    # Sidebar for user input
    st.sidebar.header("Filter Options")
    selected_column = st.sidebar.selectbox("Select a column to visualize", data.columns)
    st.sidebar.slider(f"Filter by {selected_column}", float(gdf[selected_column].min()), float(gdf[selected_column].max()), (float(gdf[selected_column].min()), float(gdf[selected_column].max())), key='range')

    # Filter the data based on user input
    filtered_data = gdf[(gdf[selected_column] >= st.sidebar.slider[0]) & (gdf[selected_column] <= st.sidebar.slider[1])]

    # Display the filtered data (optional)
    st.write("Filtered Crime Data", filtered_data)

    # Create an interactive map using Folium
    m = folium.Map(location=[gdf['Latitude'].mean(), gdf['Longitude'].mean()], zoom_start=12, control_scale=True)

    # Add markers for each crime incident
    for idx, row in filtered_data.iterrows():
        folium.Marker([row['Latitude'], row['Longitude']], popup=f"{selected_column}: {row[selected_column]}").add_to(m)

    # Display the map in Streamlit
    folium_static(m)



def page2():
    st.title("classification")



def page3():
    st.title("Regression")


def page4():
    st.title("Association rules")

page_names_to_funcs = {
    "EDA": page1,
    "Classification": page2,
    "Regression": page3,
    "Association_Rules": page4
                       }

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
