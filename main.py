import streamlit as st
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from datetime import datetime
from config import astra_client_id, astra_client_secret, astra_database_id, astra_app_name
from twilio.rest import Client
import os
import plotly.express as px
import plotly.graph_objects as go
import uuid
import bcrypt
import requests
import pandas as pd
import streamlit as st
import groq

import pickle
import os
from twilio.rest import Client

from sklearn.preprocessing import StandardScaler, LabelEncoder

st.set_page_config(page_title="Spaceve Promotional Content", page_icon="🎟️",layout="wide")






# Retrieve Twilio credentials from environment variables
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')
RECIPIENT_PHONE_NUMBER = os.getenv('RECIPIENT_PHONE_NUMBER')

def send_sms():
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    
    message = client.messages.create(
        body="Hello! This is a test message from Twilio.",
        from_=TWILIO_PHONE_NUMBER,
        to=RECIPIENT_PHONE_NUMBER
    )
    print(f"Message sent: {message.sid}")

# Call the function to send the SMS


def call():
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    client.calls.create(
        url='http://demo.twilio.com/docs/voice.xml',
        from_=TWILIO_PHONE_NUMBER,
        to=RECIPIENT_PHONE_NUMBER
    )
    print(f"called sent to {RECIPIENT_PHONE_NUMBER}")
    pass
# Function to generate engaging content using the Groq API
def generate_engaging_content_with_groq(user_name, product_name, price):
    """
    Generate personalized and engaging content using Groq API.
    
    Parameters:
        user_name (str): The name of the user.
        product_name (str): The name of the product or service.
        price (float): The price of the product or service.
    
    Returns:
        str: A personalized marketing content string.
    """
    # Set your Groq API key
    groq_api_key = os.getenv('groq_api_key')
    client = groq.Client(api_key=groq_api_key)
    
    # Define the prompt for Groq API
    prompt = (
        f"Create a professional and catchy promotional message for {user_name}. The product is '{product_name}' "
        f"priced at ₹{price:.2f}. Make the message urgent, urging the user to hurry up and purchase the tickets. "
        f"Ensure the tone is appealing, includes a strong call-to-action, and ends with 'Regards, Spaceve'."
    )
    
    try:
        # Use Groq API to generate content
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "system", "content": "You are a marketing assistant generating promotional content."},
                      {"role": "user", "content": prompt}])
        
        # Check if the API returned a response
        if response.choices:
            return response.choices[0].message.content
        else:
            return "No content was generated. Please try again."
    except Exception as e:
        return f"Error with Groq API: {str(e)}"
    




def display_floating_card():
    """
    Function to display a floating card with generated content.
    The card disappears after 5 seconds automatically.
    """
    # Streamlit app UI
    
    
    
    user_name = st.session_state['username']
    product_name = "Dinner at Skyline Restaurant"
    price = 1499.99
    # Generate promotional content
    content = generate_engaging_content_with_groq(user_name, product_name, price)

    # Define the CSS for the floating card animation (with 5-second animation duration)
    css = """
    <style>
    .floating-card {
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        background-color: #4CAF50;
        color: white;
        padding: 20px;
        border-radius: 10px;
        font-size: 16px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        opacity: 1;
        animation: floatCard 8s ease-out forwards;
    }

    @keyframes floatCard {
        0% {
            opacity: 1;
            transform: translateX(-50%) translateY(30px);
        }
        30% {
            opacity: 1;
            transform: translateX(-50%) translateY(0);
        }
        80% {
            opacity: 1;
            transform: translateX(-50%) translateY(0);
        }
        100% {
            opacity: 0;
            transform: translateX(-50%) translateY(-30px);
        }
    }
    </style>
"""



    # Inject the CSS into the Streamlit app
    st.markdown(css, unsafe_allow_html=True)

    # Create a placeholder for the popup
    popup = st.empty()

    # Display the floating card with generated content
    with popup.container():
        st.markdown(f'<div class="floating-card">{content}</div>', unsafe_allow_html=True)

    pass
        
    # No need for a button, because it disappears automatically after 5 seconds


def promote():
    send_sms()
    call()
    display_floating_card()





# Connect to Astra DB

def connect_db():
   
    cloud_config = {
        'secure_connect_bundle': 'secure-connect-datathon.zip'  # Path to your secure connect bundle
    }
    cluster = Cluster(cloud=cloud_config, auth_provider=PlainTextAuthProvider(astra_client_id, astra_client_secret))
    session = cluster.connect()
    session.set_keyspace("system1")
    return session

# Initialize connection()
session = connect_db()
# Test the connection
# if session:
#     rows = session.execute('SELECT release_version FROM system.local')
#     for row in rows:
#         print(f"Cassandra release version: {row.release_version}")

# # Load the saved KMeans model and scaler
# with open("kmeans_model.pkl", "rb") as f:
#     kmeans_model = pickle.load(f)

# with open("Kscaler.pkl", "rb") as f:
#     scaler = pickle.load(f)

# # Function to fetch data from tables
# def fetch_data(session, query):
#     rows = session.execute(query)
#     return pd.DataFrame(rows)

# # Load required data from Astra DB
# def load_data(session):
#     customers_query = """
#         SELECT customer_id, first_name, last_name, age, gender, location FROM customers;
#     """
#     bookings_query = """
#         SELECT customer_id, event_id, booking_date, total_spend FROM bookings ALLOW FILTERING;
#     """
#     events_query = """
#         SELECT event_id, event_name, event_type, location FROM events;
#     """
#     preferences_query = """
#         SELECT customer_id, preferred_event_types, upsell_preferences FROM customer_preferences ALLOW FILTERING;
#     """

#     customers_df = fetch_data(session, customers_query)
#     bookings_df = fetch_data(session, bookings_query)
#     events_df = fetch_data(session, events_query)
#     preferences_df = fetch_data(session, preferences_query)

#     return customers_df, bookings_df, events_df, preferences_df

# # Merge all customer-related data
# def merge_data(customers_df, bookings_df, events_df, preferences_df):
#     merged_df = pd.merge(bookings_df, customers_df, on="customer_id", how="left")
#     merged_df = pd.merge(merged_df, events_df, on="event_id", how="left")
#     merged_df = pd.merge(merged_df, preferences_df, on="customer_id", how="left")
#     return merged_df

# # Function to preprocess data
# def preprocess_data(df, max_categories=3):
#     # --- 1. Normalize/Scale numerical data ---
#     numerical_cols = ['age', 'total_spend']
#     df = df.copy()  # Create a copy of the DataFrame to avoid SettingWithCopyWarning
#     # df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

#     # --- 2. Encode categorical data ---
#     label_encoder = LabelEncoder()
#     df['gender'] = label_encoder.fit_transform(df['gender'])
    
#     # --- 2.1 Handle High Cardinality Categorical Columns ---
#     categorical_columns = ['event_type', 'preferred_event_types', 'upsell_preferences', 'location_x', 'location_y']
    
#     for col in categorical_columns:
#         # If column contains lists (e.g., 'preferred_event_types'), flatten the lists to strings
#         if df[col].apply(lambda x: isinstance(x, list)).any():
#             df[col] = df[col].apply(lambda x: ','.join(x) if isinstance(x, list) else x)
        
#         # Get unique values for the column
#         unique_vals = df[col].nunique()
        
#         if unique_vals > max_categories:
#             # If the column has more than `max_categories` unique values, use label encoding
#             df[col] = label_encoder.fit_transform(df[col])
#         else:
#             # Otherwise, apply one-hot encoding
#             df = pd.get_dummies(df, columns=[col], drop_first=True)

#     # --- 3. Feature Engineering ---
#     # Calculate booking frequency (how many bookings each customer has made)
#     booking_frequency = df.groupby('customer_id')['event_id'].count().reset_index()
#     booking_frequency.columns = ['customer_id', 'booking_frequency']
#     df = pd.merge(df, booking_frequency, on='customer_id', how='left')
    
#     # --- 3.1 Adding Recency Column ---
#     current_date = datetime.now()  # Current date for recency calculation
#     # Assuming 'booking_date' exists in df as a datetime object (if not, convert it to datetime)
#     df['booking_date'] = pd.to_datetime(df['booking_date'], errors='coerce')
#     df['recency'] = (current_date - df['booking_date']).dt.days  # Days since last booking
    
#     # For customers with no booking date, we can fill `recency` with a large value or NaN
#     df.loc[:, 'recency'].fillna(365 * 10, inplace=True)  # Assuming max recency as 10 years (for those with no bookings)

#     # --- 4. Handle missing values ---
#     df.fillna(0, inplace=True)

#     return df

# # Predict customer cluster
# def predict_cluster(customer_data):
#     feature_columns = ['age', 'total_spend', 'booking_frequency', 'recency']
#     customer_features = customer_data[feature_columns]
#     cluster_id = 0  # Predict cluster
#     return cluster_id

# # Update the cluster ID in Astra DB
# def update_cluster_in_db(session, customer_id, cluster_id):
#     update_query = f"""
#     UPDATE customers SET cluster = {cluster_id} WHERE customer_id = {customer_id};
#     """
#     session.execute(update_query)

# # Main pipeline to process each customer individually
# def process_customers(session, customer_id):
#     # Load and merge data
#     customers_df, bookings_df, events_df, preferences_df = load_data(session)
#     merged_df = merge_data(customers_df, bookings_df, events_df, preferences_df)

#     customer_data = merged_df[merged_df['customer_id'] == customer_id]
#     if not customer_data.empty:
#         customer_data = preprocess_data(customer_data)
#         cluster_id = predict_cluster(customer_data)
#         update_cluster_in_db(session, customer_id, cluster_id)
            

def model(customer_id):
    # process_customers(customer_id)
    promote()


def recommend_events():
    
    # Step 2: Fetch Customers and Events
    customers = list(session.execute("SELECT * FROM customers"))
    events = list(session.execute("SELECT * FROM events"))

    # Step 3: Match Customers to Events Based on Location
    event_recommendations = {}

    for customer in customers:
        matched_events = [event for event in events if event.location == customer.location]

        # If no matching event is found, suggest a fallback message and a set of popular events
        if not matched_events:
            print(f"⚠️ No matching events found for customer {customer.customer_id}. Suggesting popular events.")
            # Assuming a fallback or default set of popular events (first 3 events for simplicity)
            matched_events = events[:3]  # You can modify this logic as needed

        event_recommendations[customer.customer_id] = matched_events

    # Step 4: Generate Personalized Event Suggestions Using Groq
    recommendations = []

    for customer_id, recommended_events in event_recommendations.items():
        customer = next(c for c in customers if c.customer_id == customer_id)

        # If no matched events were found, show a different message
        if not recommended_events:
            event_details = "Unfortunately, we couldn't find any events matching your location. Here are some upcoming events you might like:\n"
            event_details += "\n".join(
                f"- {event.event_name} ({event.event_type}) on {event.start_date.strftime('%Y-%m-%d')} at {event.location}"
                for event in events[:3]  # Example fallback to popular events
            )
        else:
            event_details = "\n".join(
                f"- {event.event_name} ({event.event_type}) on {event.start_date.strftime('%Y-%m-%d')} at {event.location}"
                for event in recommended_events
            )

        context = f"""
        Customer Profile:
        - Name: {customer.first_name} {customer.last_name}
        - Age: {customer.age}
        - Location: {customer.location}
        - Family Status: {customer.family_status}
        - Device Type: {customer.device_type}

        Recommended Events:
        {event_details}

        Generate a personalized invitation message for the customer.
        """

        # Generate the personalized message
        groq_api_key = os.getenv('groq_api_key')
        client = groq.Client(api_key=groq_api_key)

        try:
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "system", "content": "Generate an engaging event recommendation."},
                          {"role": "user", "content": context}]
            )

            # If the response has choices, get the message content, otherwise return a default message
            personalized_message = response.choices[0].message.content if response.choices else "No message generated."

        except Exception as e:
            print(f"Error generating message: {e}")
            personalized_message = "An error occurred while generating the message."

        recommendations.append({
            "customer_id": customer_id,
            "recommended_events": recommended_events,
            "message": personalized_message
        })

    return recommendations

# Test the function to see the output



# Function to connect to Cassandra database

# def promote():



# Initialize Cassandra session



def fetch_campaign_data(session):
    query = """
        SELECT campaign_id, start_date, performance_score, event_id
        FROM campaigns
    """
    rows = session.execute(query)
    
    campaign_data = []
    for row in rows:
        event_query = f"SELECT event_name FROM events WHERE event_id = {row.event_id}"
        event_name_row = session.execute(event_query).one()
        
        campaign_data.append({
            "campaign_id": row.campaign_id,
            "event_id": row.event_id,
            "event_name": event_name_row.event_name if event_name_row else "Unknown Event",
            "start_date": row.start_date,
            "performance_score": round(row.performance_score * 100, 3)
        })
    
    return pd.DataFrame(campaign_data)

def fetch_engagement_data(session, event_id):
    query = f"""
        SELECT engagement_id, clicks, customer_id, engagement_date, event_id, purchases
        FROM engagements
        WHERE event_id = {event_id} ALLOW FILTERING
    """
    rows = session.execute(query)
    
    engagement_data = [{
        "engagement_id": row.engagement_id,
        "clicks": row.clicks,
        "customer_id": row.customer_id,
        "engagement_date": row.engagement_date,
        "event_id": row.event_id,
        "purchases": row.purchases
    } for row in rows]
    
    return pd.DataFrame(engagement_data)

def plot_performance_score(campaign_data, selected_campaign):
    filtered_data = campaign_data[campaign_data['campaign_id'] == selected_campaign]
    fig = px.line(
        filtered_data, x='start_date', y='performance_score',
        markers=True, title=f'Performance Over Time for Campaign {selected_campaign}',
        labels={'start_date': 'Date', 'performance_score': 'Performance Score'}
    )
    return fig

def plot_engagement_metrics(engagement_data):
    daily_clicks = engagement_data.groupby('engagement_date')['clicks'].sum()
    daily_purchases = engagement_data.groupby('engagement_date')['purchases'].sum()
    conversion_rate = (daily_purchases / daily_clicks * 100).fillna(0)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily_clicks.index, y=daily_clicks.values, mode='lines+markers', name='Daily Clicks'))
    fig.add_trace(go.Scatter(x=daily_purchases.index, y=daily_purchases.values, mode='lines+markers', name='Daily Purchases', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=conversion_rate.index, y=conversion_rate.values, mode='lines+markers', name='Conversion Rate (%)', line=dict(color='orange')))
    
    fig.update_layout(title='Engagement Metrics Over Time', xaxis_title='Date', yaxis_title='Value')
    return fig

def display_engagement_dashboard(session, event_id):
    st.header("Engagement Dashboard")
    engagement_data = fetch_engagement_data(session, event_id)
    
    if engagement_data.empty:
        st.warning("No engagement data available for this event.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Clicks", engagement_data['clicks'].sum())
    with col2:
        st.metric("Total Purchases", engagement_data['purchases'].sum())
    with col3:
        conversion_rate = (engagement_data['purchases'].sum() / engagement_data['clicks'].sum() * 100)
        st.metric("Conversion Rate", f"{conversion_rate:.2f}%")
    with col4:
        st.metric("Unique Customers", engagement_data['customer_id'].nunique())
    
    st.plotly_chart(plot_engagement_metrics(engagement_data))
    
    with st.expander("View Raw Data"):
        st.dataframe(engagement_data)

def analytics():
    st.title("Campaign Performance & Engagement Dashboard")
    
    campaign_data = fetch_campaign_data(session)
    
    if campaign_data.empty:
        st.warning("No campaign data available.")
        return
    
    st.subheader("Analytics Performance")
    # selected_campaign = st.selectbox("Select a campaign:", campaign_data['campaign_id'].unique())
    # st.plotly_chart(plot_performance_score(campaign_data, selected_campaign))
    
    campaign_selection = st.selectbox("Select an event to view engagement data:", campaign_data['event_id'].unique())
    if campaign_selection:
        display_engagement_dashboard(session, campaign_selection)

# Create necessary tables
def create_tables():
    session.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            customer_id UUID PRIMARY KEY,
            first_name TEXT,
            last_name TEXT,
            age INT,
            email TEXT,
            phone_number TEXT,
            family_status TEXT,
            device_type TEXT,
            location TEXT,
            join_date TIMESTAMP
        )
    """)
    
    session.execute("""
        CREATE TABLE IF NOT EXISTS customer_preferences (
            customer_id UUID PRIMARY KEY,
            preferred_activities TEXT,
            preferred_event_types TEXT,
            upsell_preferences TEXT
        )
    """)
    
    session.execute("""
        CREATE TABLE IF NOT EXISTS events (
            event_id UUID PRIMARY KEY,
            event_name TEXT,
            event_description TEXT,
            event_type TEXT,
            location TEXT,
            max_capacity INT,
            start_date TIMESTAMP,
            end_date TIMESTAMP
        )
    """)

    session.execute("""
        CREATE TABLE IF NOT EXISTS bookings (
            booking_id UUID PRIMARY KEY,
            booking_channel TEXT,
            booking_date TIMESTAMP,
            customer_id UUID,
            event_date TIMESTAMP,
            event_id UUID,
            number_of_guests INT,
            total_spend DECIMAL,
            upsells TEXT
        )
    """)

    session.execute("""
        CREATE TABLE IF NOT EXISTS profile (
            id UUID PRIMARY KEY,
            username TEXT,
            password TEXT,
            location TEXT
        )
    """)

# Function to fetch user's current location (optional)
def get_current_location():
    try:
        response = requests.get('http://ip-api.com/json/')  # Free geolocation API
        data = response.json()
        if response.status_code == 200:
            return f"{data['city']}"
        else:
            return "Location Unavailable"
    except Exception as e:
        st.error("Error fetching location. Ensure you have internet access.")
        return "Location Error"



def sign_up():
    st.title("Sign Up")
    st.write("Enter your details to create an account.")

    first_name = st.text_input("First Name")
    last_name = st.text_input("Last Name")
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    email = st.text_input("Email")
    phone_number = st.text_input("Phone Number")
    family_status = st.selectbox("Family Status", ["Single", "Married", "Divorced", "Widowed"])
    device_type = st.selectbox("Device Type", ["Smartphone", "Tablet", "Desktop", "Other"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    
    if st.button("Sign Up"):
        if password != confirm_password:
            st.error("Passwords do not match!")
        else:
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            customer_id = uuid.uuid4()
            join_date = datetime.now()
            location = get_current_location()

            # Insert user data into the customer and profile tables
            try:
                session.execute("""
                    INSERT INTO customers (customer_id, first_name, last_name, age, email, phone_number, family_status, device_type, location, join_date)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (customer_id, first_name, last_name, age, email, phone_number, family_status, device_type, location, join_date))
                
                session.execute("""
                    INSERT INTO profile (id, username, password, location) 
                    VALUES (%s, %s, %s, %s)
                """, (customer_id, username, hashed_password, location))

                st.success("Sign Up Successful! Please log in.")
            except Exception as e:
                st.error(f"Error: {e}")

# Login functionality
def login():
    st.title("Login")
    username = st.text_input("Enter username")
    password = st.text_input("Enter Password", type="password")
    
    if st.button("Login"):
        try:
            # Query the profile table for the given username
            query = "SELECT * FROM profile WHERE username = %s ALLOW FILTERING"
            result = session.execute(query, (username,))
            user = result.one()

            query = "SELECT * FROM customers"
            result1 = session.execute(query)
            user1 = result1.one()
            
            if user and bcrypt.checkpw(password.encode('utf-8'), user.password.encode('utf-8')):
                st.session_state["logged_in"] = True
                st.session_state["customer_id"] = user1.customer_id
                st.session_state["username"] = user1.first_name
                st.success(f"Welcome, {user1.first_name}!")
            else:
                st.error("Invalid username or password.")
        except Exception as e:
            st.error(f"Error: {e}")

# Admin functionality to manage events
def admin():

    tab1, tab2 = st.tabs(["events","admin panel"])

    with tab1:

        st.title("Admin Page")
        event_name = st.text_input("Event Name")
        event_description = st.text_area("Event Description")
        event_type = st.text_input("Event Type")
        event_location = st.text_input("Event Location")
        max_capacity = st.number_input("Max Capacity", min_value=1)
        start_date = st.date_input("Start Date")
        end_date = st.date_input("End Date")

        if st.button("Add Event"):
            event_id = uuid.uuid4()
            try:
                session.execute("""
                    INSERT INTO events (event_id, event_name, event_description, event_type, location, max_capacity, start_date, end_date)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (event_id, event_name, event_description, event_type, event_location, max_capacity, start_date, end_date))
                st.success("Event added successfully!")
            except Exception as e:
                st.error(f"Error: {e}")

    with tab2:
        st.title("Event Analytics")
        analytics()



# Call the function to display the floating card

def fetch_events():
    rows = session.execute("SELECT * FROM events")
    events = [
        f"- {row.event_name} ({row.event_type}) on {row.start_date.strftime('%Y-%m-%d')} at {row.location}"
        for row in rows
    ]
    return "\n".join(events) if events else "No events available."

def chat_with_ai(customer_name, user_input):
    groq_api_key = os.getenv('groq_api_key')
    client = groq.Client(api_key=groq_api_key)

    # Fetch events
    event_details = fetch_events()

    # Define chatbot context
    context = f"""
    Customer: {customer_name}
    
    Available Events:
    {event_details}
    
    User Query: {user_input}
    
    Provide a friendly, engaging response with:
    - A friendly greeting 🎉
    - Relevant event recommendations 📅
    - Call-to-action (CTA) 🚀
    - Sign off with "Regards, Spaceve"  
    """

    # Generate response using Groq API
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a helpful event recommendation assistant."},
                {"role": "user", "content": context}
            ]
        )
        return response.choices[0].message.content if response.choices else "I'm unable to respond right now."
    except Exception as e:
        return f"Error: {e}"

def tabs(customer_id):
    # Create Streamlit tabs
    tab1, tab2 = st.tabs(["Events", "Chatbot"])

    with tab1:
        st.title("Suggest Events and Book")
        if st.button("Run Model"):
            model(st.session_state["customer_id"])  # Ensure model() is properly defined
            st.success("Model executed successfully!")

        # Fetch all events from the events table
        events_query = "SELECT * FROM events"
        events = session.execute(events_query)
        
        # Cache events in a list for reuse
        events_list = []
        for event in events:
            events_list.append({
                "id": event.event_id,
                "name": event.event_name,
                "type": event.event_type,
                "location": event.location,
                "date": event.start_date
            })
        
        if not events_list:
            st.warning("No events available.")
            return

        # Display events in a grid format
        st.write("### Available Events")
        for row_index in range(0, len(events_list), 4):  # 4 columns per row
            cols = st.columns(4)
            for col_index, event in enumerate(events_list[row_index:row_index + 4]):
                with cols[col_index]:
                    st.markdown(
                        f"""
                        <div style="
                            background-color: #f8f9fa;
                            border-radius: 10px;
                            padding: 15px;
                            margin-bottom: 15px;
                            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
                            text-align: left;
                        ">
                            <h3 style="margin: 0; color: #0d6efd;">{event['name']}</h3>
                            <p style="margin: 0; color: #6c757d;"><strong>Type:</strong> {event['type']}</p>
                            <p style="margin: 0; color: #6c757d;"><strong>Location:</strong> {event['location']}</p>
                            <p style="margin: 0; color: #6c757d;"><strong>Date:</strong> {event['date']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    
                    if st.button(f"Book '{event['name']}'", key=f"book-{event['id']}"):
                        st.write(f"### Booking Form for {event['name']}")
                        with st.form(key=f"form-{event['id']}"):
                            guests = st.number_input("Number of Guests", min_value=1, step=1, key=f"guests-{event['id']}")
                            channel = st.text_input("Booking Channel", key=f"channel-{event['id']}")
                            spend = st.number_input("Total Spend", min_value=0.0, step=0.01, key=f"spend-{event['id']}")
                            upsells = st.text_area("Upsells (Optional)", key=f"upsells-{event['id']}")

                            # Ensure UUID format for booking_id
                            booking_id = uuid.uuid4()

                            if st.form_submit_button("Submit"):
                                
                                # Insert booking details into the database
                                try:
                                    current_time = datetime.now()
                                    booking_date = current_time.strftime('%Y-%m-%d %H:%M:%S')  # Format as string
                                    
                                    # Using parameterized query for booking insertion
                                    booking_query = """
                                    INSERT INTO bookings (booking_id, customer_id, event_id, number_of_guests, booking_channel, total_spend, upsells, booking_date)
                                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                                    """
                                    
                                    # Execute the query to insert booking
                                    session.execute(booking_query, (
                                        booking_id, 
                                        customer_id, 
                                        event['id'], 
                                        guests, 
                                        channel, 
                                        spend, 
                                        upsells, 
                                        booking_date
                                    ))
                                    
                                    st.success(f"Booking confirmed for '{event['name']}' with {guests} guests!")
                                    print(f"Booking confirmed for '{event['name']}' with {guests} guests!")
                                    
                                    
                                except Exception as e:
                                    st.error(f"Failed to save booking: {e}")
    with tab2:
        st.subheader("🤖 Ask the AI Chatbot")
        customer_name = st.text_input("Enter your name:")
        user_input = st.text_area("Ask about upcoming events:")

        if st.button("Send"):
            if customer_name and user_input:
                response = chat_with_ai(st.session_state['username'], user_input)
                st.markdown(f"**AI Response:**\n\n{response}")
            else:
                st.warning("Please enter both your name and a question.")

    

def main():
    create_tables()

    if "logged_in" in st.session_state and st.session_state["logged_in"]:
        st.sidebar.write(f"Welcome, {st.session_state['username']}!")
        if st.sidebar.button("Log Out"):
            st.session_state.clear()
            st.rerun()

        tabs(st.session_state["customer_id"],)

        
    else:
        st.sidebar.title("Authentication")
        choice = st.sidebar.radio("Choose", ["Login", "Sign Up","Admin"])
        if choice == "Login":
            login()
        elif choice == "Sign Up":
            sign_up()
        elif choice == "Admin":
            admin()

# Run the application
if __name__ == "__main__":
    main()
