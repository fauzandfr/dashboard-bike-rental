import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.title('Bike Sharing Demand Analysis Dashboard')

# Load data
@st.cache
def load_data(day_path, hour_path):
    day_data = pd.read_csv(day_path)
    hour_data = pd.read_csv(hour_path)
    day_data['dteday'] = pd.to_datetime(day_data['dteday'])
    hour_data['dteday'] = pd.to_datetime(hour_data['dteday'])
    return day_data, hour_data

day_df, hour_df = load_data('day.csv', 'hour.csv')

analysis_type = st.sidebar.selectbox("Select Analysis Type", 
                                      ["Trend Analysis", "Weather Impact", "User Type Patterns", "Hourly Usage Analysis", "Clustering Analysis"])

if analysis_type == "Trend Analysis":
    st.subheader("Daily Bike Usage Over Time")
    plt.figure(figsize=(10, 5))
    plt.plot(day_df['dteday'], day_df['cnt'], label='Total Bike Users')
    plt.title('Daily Bike Usage Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Bike Users')
    plt.legend()
    st.pyplot(plt)

elif analysis_type == "Weather Impact":
    st.subheader("Daily Bike Usage vs. Weather Conditions")
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    sns.scatterplot(x='temp', y='cnt', data=day_df, ax=axs[0, 0], alpha=0.6)
    axs[0, 0].set_title('Temperature vs. Bike Users')
    
    sns.scatterplot(x='atemp', y='cnt', data=day_df, ax=axs[0, 1], alpha=0.6)
    axs[0, 1].set_title('Felt Temperature vs. Bike Users')
    
    sns.scatterplot(x='hum', y='cnt', data=day_df, ax=axs[1, 0], alpha=0.6)
    axs[1, 0].set_title('Humidity vs. Bike Users')
    
    sns.scatterplot(x='windspeed', y='cnt', data=day_df, ax=axs[1, 1], alpha=0.6)
    axs[1, 1].set_title('Windspeed vs. Bike Users')
    
    for ax in axs.flat:
        ax.set(xlabel='', ylabel='')

    st.pyplot(fig)

elif analysis_type == "User Type Patterns":
    st.subheader("Daily Bike Usage by User Type")
    
    # Membuat DataFrame baru untuk agregat jumlah pengguna
    usage_types = day_df[['dteday', 'casual', 'registered']].melt(id_vars=['dteday'], var_name='User Type', value_name='Count')
    
    # Visualisasi 1: Penggunaan Harian berdasarkan Tipe Pengguna
    fig, ax = plt.subplots(figsize=(15, 7))
    sns.lineplot(x='dteday', y='Count', hue='User Type', data=usage_types, alpha=0.8, ax=ax)
    ax.set_title('Daily Bike Usage by User Type')
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Bike Users')
    ax.legend(title='User Type')
    st.pyplot(fig)

    # Menambahkan kolom 'day_type' untuk membedakan antara weekdays dan weekends
    day_df['day_type'] = day_df['weekday'].apply(lambda x: 'Weekend' if x in [0, 6] else 'Weekday')

    # Agregat dan melt data untuk visualisasi kedua
    usage_day_type = day_df.groupby(['dteday', 'day_type']).sum()[['casual', 'registered']].reset_index()
    usage_day_type = usage_day_type.melt(id_vars=['dteday', 'day_type'], var_name='User Type', value_name='Count')

    # Visualisasi 2: Penggunaan Harian berdasarkan Tipe Pengguna dan Tipe Hari
    fig, ax = plt.subplots(figsize=(15, 7))
    sns.lineplot(x='dteday', y='Count', hue='User Type', style='day_type', data=usage_day_type, alpha=0.8, ax=ax)
    ax.set_title('Daily Bike Usage by User Type and Day Type')
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Bike Users')
    ax.legend(title='User Type / Day Type')
    st.pyplot(fig)



elif analysis_type == "Hourly Usage Analysis":
    st.subheader("Hourly Bike Usage Analysis")

    # Menambahkan kolom 'day_type' untuk membedakan antara weekdays dan weekends
    hour_df['day_type'] = hour_df['weekday'].apply(lambda x: 'Weekend' if x in [0, 6] else 'Weekday')

    # Visualisasi 1: Rata-rata Penggunaan Sepeda per Jam
    fig, ax = plt.subplots(figsize=(15, 7))
    sns.pointplot(data=hour_df, x='hr', y='cnt', ci=None, color='blue', ax=ax)
    ax.set_title('Average Hourly Bike Usage Across All Days')
    ax.set_xlabel('Hour of the Day')
    ax.set_ylabel('Average Number of Bike Users')
    ax.set_xticks(range(0, 24))
    st.pyplot(fig)

    # Visualisasi 2: Perbandingan weekdays dan weekends
    fig, ax = plt.subplots(figsize=(15, 7))
    sns.pointplot(data=hour_df, x='hr', y='cnt', hue='day_type', ci=None, ax=ax)
    ax.set_title('Average Hourly Bike Usage: Weekday vs Weekend')
    ax.set_xlabel('Hour of the Day')
    ax.set_ylabel('Average Number of Bike Users')
    ax.set_xticks(range(0, 24))
    ax.legend(title='Day Type')
    st.pyplot(fig)

    # Visualisasi 3: Pengaruh Kondisi Cuaca
    fig, ax = plt.subplots(figsize=(15, 7))
    sns.pointplot(data=hour_df, x='hr', y='cnt', hue='weathersit', ci=None, ax=ax)
    ax.set_title('Average Hourly Bike Usage by Weather Situation')
    ax.set_xlabel('Hour of the Day')
    ax.set_ylabel('Average Number of Bike Users')
    ax.set_xticks(range(0, 24))
    ax.legend(title='Weather Situation', labels=['Clear', 'Mist + Cloudy', 'Light Snow / Light Rain', 'Heavy Rain / Ice Pallets'])
    st.pyplot(fig)

elif analysis_type == "Clustering Analysis":
    st.subheader("Clustering of Daily Bike Usage Based on Usage, Temperature, and Humidity")
    
    # Clustering
    features = day_df[['cnt', 'temp', 'hum']]
    features_scaled = StandardScaler().fit_transform(features)
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(features_scaled)
    day_df['Cluster'] = kmeans.labels_
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(day_df['cnt'], day_df['temp'], c=day_df['Cluster'], cmap='viridis')
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    ax.set_title('Clustering of Daily Bike Usage')
    ax.set_xlabel('Number of Bike Users')
    ax.set_ylabel('Normalized Temperature')
    st.pyplot(fig)
    
    # Show cluster analysis
    cluster_info = st.selectbox("Select Cluster to Analyze", options=sorted(day_df['Cluster'].unique()))
    st.write(day_df[day_df['Cluster'] == cluster_info][['cnt', 'temp', 'hum']].describe())

