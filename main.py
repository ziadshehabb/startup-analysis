#Import Libraries
import streamlit as st 
import pandas as pd 
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu
from PIL import Image 
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import datetime as dt
import matplotlib.pyplot as plt

#Page Configeration
st.set_page_config(page_title='Startup Case Study ', page_icon="🚀", layout = "wide")

#Read Csv
df = pd.read_csv("https://raw.githubusercontent.com/ziadshehabb/startup-analysis/main/visits.csv")
#Sidebar Navigation
selected = st.sidebar.radio(" Navigation Menu ",["Overview","Sales","Customers","Machine Learning"], index = 0)

if selected == "Overview": 
    #Design
    st.title("Tech Startup Case Study")
    st.image("https://us.123rf.com/450wm/stmool/stmool1603/stmool160300208/53250666-home-improvement-and-renovation-services.jpg?ver=6", width = 400 )
    st.write('One of the leading SaaS startups in the home services in the Gulf region, before COVID19, was massively hit by the pandemic just like everyone else in the world. However, the company was unable to rebound quickly as the complete stop of its operations has led to an overall restructure and major changes in its’ core functions. The purpose of application is to understand the dataset better on two levels, Sales and Customers, in order to know more about the industry and perhaps see the impact of COVID on the firm. The startup operates with Human Capital suppliers in order to employ their service providers in the best possible way, to provide customers with home services through different sale items : packages, one-offs, and jobs. ')
    #View Data
    data_load_state = st.header('View Data')
    agree = st.checkbox('First 10 rows')

    if agree:
     st.subheader('First ten rows')
     st.dataframe(df.head(10))

    agree = st.checkbox('Full data set')

    if agree:
        st.subheader('Full data set')
        st.write(df)

    #Data Description
    st.header('Description')
    transactions, variables, null_values = st.columns(3)
    transactions.metric("Number of Transactions", f"{len(df):,}")
    variables.metric("Number of Variables",df.shape[1])
    null_values.metric("Null Unique Variables", df["visit_id"].isna().sum())
    

if selected == "Sales":
    st.title("Sales Dashboard")
    #Creating Filters
    year, month, country = st.columns(3)
    with country : 
        Country = st.multiselect("Country",options=df.sort_values(by="country_id").country_id.unique(),default=df.sort_values(by="country_id").country_id.unique())
    with year:
        Year  = st.multiselect("Year",options=df["visit_year"].unique(),default=df["visit_year"].unique())
    with month:
        Month = st.multiselect("Month",options=df["visit_month"].unique(),default=df["visit_month"].unique())
    #Filtered dataframe
    df_filtered = df.query(
        "country_id == @Country & visit_year ==@Year & visit_month == @Month "
    )
    #Top KPIs
    total_sales = int(df_filtered["Price_dollar"].sum())
    average_monthly_sales = df_filtered.groupby(by=["visit_month"]).sum()[["Price_dollar"]]
    Number_of_service_providers = len(df_filtered["service_provider_id"].unique())

    k1,k2,k3 = st.columns(3)
    k1.metric("Total Sales",f"US $ {total_sales:,}")
    k2.metric("Average Sales per Month",f"US ${int(average_monthly_sales.mean()):,}")
    k3.metric("Number of Service Providers",f" {Number_of_service_providers:,}")

    st.markdown("""---""")   
    #Yearly Sales
    Yearly_sales = df_filtered.groupby(by=["visit_year"]).sum()[["Price_dollar"]]
    fig_yearly_sales = px.bar(
        Yearly_sales,
        x=Yearly_sales.index,
        y="Price_dollar",
        title="<b> Yearly Sales </b> ",
        color_discrete_sequence=["#0083B8"] * len(Yearly_sales),
        template="plotly_white",
    )
    fig_yearly_sales.update_layout(
        xaxis=dict(tickmode="linear"),
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=(dict(showgrid=False)),
    )

    # Monthly Sales 
    Monthly_sales = df_filtered.groupby(by=["visit_month"]).sum()[["Price_dollar"]]
    fig_monthly_sales = px.bar(
        Monthly_sales,
        x=Monthly_sales.index,
        y="Price_dollar",
        title="<b> Monthly Sales </b>",
        color_discrete_sequence=["#0083B8"] * len(Monthly_sales),
        template="plotly_white",
    )
    fig_monthly_sales.update_layout(
        xaxis=dict(tickmode="linear"),
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=(dict(showgrid=False)),
    )
    g1, g2 = st.columns(2)
    g1.plotly_chart(fig_yearly_sales, use_container_width=True)
    g2.plotly_chart(fig_monthly_sales, use_container_width=True)
    
    #Sales per Visit Type
    Sales_type = df_filtered.groupby(by=["visit_type"]).sum()[["Price_dollar"]]
    fig_sales_type = px.bar(
        Sales_type,
        x=Sales_type.index,
        y="Price_dollar",
        title="<b> Sales per type </b>",
        color_discrete_sequence=["#0083B8"] * len(Sales_type),
        template="plotly_white"
    )
    fig_sales_type.update_layout(
        xaxis=dict(tickmode="linear"),
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=(dict(showgrid=False)),
    )

    #Sales Per Country
    Sales_country = px.pie(df_filtered.groupby(by="country_id").count(), names = df_filtered['country_id'], values = df_filtered['contact_id'], hole = 0.3
    )
    Sales_country.update_layout(
    title_text="<b> Sales per Country </b>",
    annotations=[dict(text='Product', x=0.5, y=0.5, font_size=20, showarrow=False)])

    g3, g4 = st.columns(2)
    g3.plotly_chart(fig_sales_type, use_container_width=True)
    g4.plotly_chart(Sales_country, use_container_width=True)
    

if selected == "Customers":
    st.title("Customers Dashboard")
    #Creating Filters
    year, month, country = st.columns(3)
    with country : 
        Country = st.multiselect("Country",options=df.sort_values(by="country_id").country_id.unique(),default=df.sort_values(by="country_id").country_id.unique())
    with year:
        Year  = st.multiselect("Year",options=df["visit_year"].unique(),default=df["visit_year"].unique())
    with month:
        Month = st.multiselect("Month",options=df["visit_month"].unique(),default=df["visit_month"].unique())
    #Filtered dataframe
    df_filtered = df.query(
        "country_id == @Country & visit_year ==@Year & visit_month == @Month "
    )
    #Number of unique customers per year
    df2 = df_filtered.groupby(['visit_month','visit_year',])['contact_id'].nunique().reset_index()
    fig_customers_monthly = px.line(df2, x="visit_month", y="contact_id", color='visit_year', title="<b> Number of Unique Customers per month </b>")
    st.plotly_chart(fig_customers_monthly, use_container_width=True)
    #Number of unique customers per country
    df3 = df_filtered.groupby(['country_id','city_id',])['contact_id'].nunique().reset_index()
    fig_customers_country = px.bar(df3, x="country_id", y="contact_id", color="city_id", title="<b> Number of unique customers per country </b>")
    st.plotly_chart(fig_customers_country, use_container_width=True)

if selected == "Machine Learning":
    st.title("Machine Learning Application")
    # New dataframe for RFM part
    df_rfm = df.copy()
    #Keep wanted columns
    df_rfm = df_rfm[["contact_id", "c_visit_date","visit_id","Price_dollar",]]
    df_rfm.head()
    #Frequency Component
    recency = pd.DataFrame(df_rfm.groupby('contact_id')['c_visit_date'].max().reset_index())
    recency['c_visit_date'] = pd.to_datetime(recency['c_visit_date']).dt.date
    recency['MaxDate'] = recency['c_visit_date'].max()
    recency['recency'] = (recency['MaxDate'] - recency['c_visit_date']).dt.days + 1
    recency = recency[['contact_id','recency']]
    recency.head()
    #Frequency Component
    frequency = pd.DataFrame(df_rfm.groupby('contact_id')['visit_id'].nunique().reset_index())
    frequency.columns = ['fcontact_id','frequency']
    frequency.head()
    #Monetary Component
    monetary = pd.DataFrame(df_rfm.groupby('contact_id')['Price_dollar'].sum().reset_index())
    monetary.columns = ['mcontact_id','monetary']
    monetary.head()
    #Combine RFM elements in one dataframe
    rfm = pd.concat([recency,frequency,monetary], axis=1)
    rfm.drop(['fcontact_id','mcontact_id'], axis=1, inplace=True)
    rfm.head(10)

    # New dataframe for k-means part
    Kmeans_df = rfm.copy()

    # Drop Contact ID
    Kmeans_df = Kmeans_df.iloc[:,1:]

    # Scaling the variables and creating a new dataframe
    standard_scaler = StandardScaler()
    Kmeans_norm_df = standard_scaler.fit_transform(Kmeans_df)
    Kmeans_norm_df = pd.DataFrame(Kmeans_norm_df)
    Kmeans_norm_df.columns = ['recency','frequency','monetary']

    #Specify Optimal number of Clusters
    cluster3 = KMeans(n_clusters = 3)
    cluster3.fit(Kmeans_norm_df)
    Kmeans_df['clusters'] = cluster3.labels_
    Kmeans_df.groupby('clusters').mean().round(0)

    # Elbow-curve

    ssd = []
    for num_clusters in list(range(1,11)):
        model_clus = KMeans(n_clusters = num_clusters)
        model_clus.fit(Kmeans_norm_df)
        ssd.append(model_clus.inertia_)
        
    fig = plt.figure(figsize=(15,5))
    plt.plot(np.arange(1,11,1), ssd)
    plt.xlabel('Number of cluster', size=12)
    plt.ylabel('Sum of Square Distance(SSD)', size=12)
    st.subheader("Elbow Curve")
    st.pyplot(fig,use_container_width=True)

    #3D clusters
    fig1 = plt.figure(figsize = (15, 10))
    ax = plt.axes(projection ="3d")
    ax.scatter3D(Kmeans_df.recency, Kmeans_df.frequency, Kmeans_df.monetary, c=Kmeans_df.clusters, cmap='coolwarm')
    ax.set_xlabel('Recency')
    ax.set_ylabel('Frequency')
    ax.set_zlabel('Monetary')
    ax.set(facecolor='white')
    st.subheader("3D Clusters")
    st.pyplot(fig1)


