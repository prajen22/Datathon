import uuid
import random
from datetime import datetime, timedelta
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from config import astra_client_id, astra_client_secret, astra_app_name

# Astra DB Configuration
ASTRA_DB_CLIENT_ID = astra_client_id
ASTRA_DB_CLIENT_SECRET = astra_client_secret
ASTRA_DB_SECURE_CONNECT_BUNDLE_PATH = "secure-connect-datathon.zip"

# Connect to Astra DB
def connect_to_astra():
    auth_provider = PlainTextAuthProvider(ASTRA_DB_CLIENT_ID, ASTRA_DB_CLIENT_SECRET)
    cluster = Cluster(cloud={'secure_connect_bundle': ASTRA_DB_SECURE_CONNECT_BUNDLE_PATH}, auth_provider=auth_provider)
    session = cluster.connect()
    session.set_keyspace(astra_app_name)  # Replace with your keyspace name
    return session

# Generate customers
def generate_customers(num_customers=200):
    first_names = ["Liam", "Olivia", "Noah", "Emma", "James", "Sophia"]
    last_names = ["Smith", "Johnson", "Brown", "Williams", "Jones", "Miller"]
    locations = ["New York", "London", "Berlin", "Tokyo", "Sydney", "Dubai"]
    genders = ["Male", "Female", "Non-binary"]
    device_types = ["iOS", "Android", "Windows", "Mac", "Tablet"]
    family_statuses = ["Single", "Married", "With Kids"]

    customers = []
    for _ in range(num_customers):
        customer_id = uuid.uuid4()
        first_name = random.choice(first_names)
        last_name = random.choice(last_names)
        email = f"{first_name.lower()}.{last_name.lower()}{random.randint(1, 1000)}@gmail.com"
        phone_number = f"+1{random.randint(1000000000, 9999999999)}"
        age = random.randint(18, 65)
        gender = random.choice(genders)
        location = random.choice(locations)
        device_type = random.choice(device_types)
        family_status = random.choice(family_statuses)
        join_date = datetime.now() - timedelta(days=random.randint(10, 1000))

        customers.append((
            customer_id, first_name, last_name, email, phone_number, age, gender, 
            location, join_date, device_type, family_status
        ))
    
    return customers

# Generate events
def generate_events(num_events=50):
    event_names = ["Zero-G Dance", "Martian Concert", "Moon Dinner", "Space VR Adventure"]
    event_types = ["Adventure", "Luxury", "Romance", "Educational"]
    locations = ["Mars", "Moon", "Space Station", "Orbital Colony"]

    events = []
    for _ in range(num_events):
        event_id = uuid.uuid4()
        event_name = f"{random.choice(event_names)} {random.randint(1, 100)}"
        event_type = random.choice(event_types)
        location = random.choice(locations)
        start_date = datetime.now() + timedelta(days=random.randint(1, 60))
        end_date = start_date + timedelta(hours=random.randint(2, 6))
        max_capacity = random.randint(50, 500)

        # Set the status based on the event's end date
        status = "Active" if end_date >= datetime.now() else "Closed"
        events.append((event_id, event_name, event_type, f"Description for {event_name}", start_date, end_date, max_capacity, location, status))
    return events

# Generate bookings
def generate_bookings(customers, events, num_bookings=500):
    bookings = []
    for _ in range(num_bookings):
        booking_id = uuid.uuid4()
        customer_id = random.choice(customers)[0]
        event_id = random.choice(events)[0]
        booking_date = datetime.now() - timedelta(days=random.randint(1, 30))
        event_date = datetime.now() + timedelta(days=random.randint(1, 60))
        number_of_guests = random.randint(1, 4)
        total_spend = round(random.uniform(100, 1000), 2)
        booking_channel = random.choice(["website", "mobile app", "call center"])
        
        # Assign realistic upsells based on event type
        possible_upsells = [
            "VIP Seating", "Meal Package", "Merchandise", "Fast Pass",
            "Photo Package", "Exclusive Access", "Transportation Service"
        ]
        upsells = random.sample(possible_upsells, k=random.randint(0, 3))  # 0 to 3 upsells per booking
        
        # Convert upsells to a string format for database storage
        upsells_str = ", ".join(upsells) if upsells else None  # None if no upsells

        bookings.append((booking_id, booking_channel, booking_date, None, customer_id, 
                         event_date, event_id, number_of_guests, total_spend, upsells_str))
    
    return bookings

# Generate customer preferences
def generate_customer_preferences(customers, bookings, engagements):
    session = connect_to_astra()
    
    preferences = []
    for customer in customers:
        customer_id = customer[0]

        # Find all bookings of this customer
        customer_bookings = [b for b in bookings if b[1] == customer_id]
        if customer_bookings:
            # Preferred event types based on bookings
            preferred_event_types = list(set([session.execute(
                "SELECT event_type FROM events WHERE event_id = %s", (b[2],)
            ).one()[0] for b in customer_bookings]))

            # Preferred locations based on bookings
            preferred_locations = list(set([session.execute(
                "SELECT location FROM events WHERE event_id = %s", (b[2],)
            ).one()[0] for b in customer_bookings]))

            # Calculate average spend
            avg_spend = sum([b[6] for b in customer_bookings]) / len(customer_bookings)
            
            # Most frequent booking channel
            preferred_booking_channel = max(set([b[7] for b in customer_bookings]), key=[b[7] for b in customer_bookings].count)
        else:
            preferred_event_types = [random.choice(["Adventure", "Luxury", "Romance", "Educational"])]
            preferred_locations = [random.choice(["Mars", "Moon", "Space Station", "Orbital Colony"])]
            avg_spend = round(random.uniform(100, 1000), 2)
            preferred_booking_channel = random.choice(["website", "mobile app", "call center"])

        # Find engagement data for clicks and purchases
        customer_engagements = [e for e in engagements if e[1] == customer_id]
        total_clicks = sum([e[3] for e in customer_engagements]) if customer_engagements else random.randint(5, 50)
        total_purchases = sum([e[4] for e in customer_engagements]) if customer_engagements else random.randint(1, total_clicks)

        # Create preferred activities and upsell preferences
        # Assuming 'preferred_activities' can be derived from preferred_event_types and locations
        preferred_activities = [f"{event_type} at {location}" for event_type, location in zip(preferred_event_types, preferred_locations)]
        
        # Generate random upsell preferences based on event type or spend
        possible_upsells = [
            "VIP Seating", "Meal Package", "Merchandise", "Fast Pass", 
            "Photo Package", "Exclusive Access", "Transportation Service"
        ]
        # Random upsell choices based on avg_spend or preferences
        upsell_preferences = random.sample(possible_upsells, k=random.randint(0, 3))  # 0 to 3 upsells per customer
        
        # Insert customer preferences with upsell_preferences as a list
        preferences.append((customer_id, preferred_activities, preferred_event_types, upsell_preferences))
    
    return preferences

# Generate campaigns
def generate_campaigns(events, num_campaigns=50):
    campaigns = []
    for _ in range(num_campaigns):
        campaign_id = uuid.uuid4()
        event_id = random.choice(events)[0]
        impressions = random.randint(1000, 100000)
        ad_spend = round(random.uniform(500, 5000), 2)
        start_date = datetime.now() - timedelta(days=random.randint(1, 60))
        end_date = start_date + timedelta(days=random.randint(1, 15))
        campaigns.append((campaign_id, event_id, impressions, ad_spend, start_date, end_date))
    return campaigns

# Generate engagements
def generate_engagements(customers, events, num_engagements=500):
    engagements = []
    for _ in range(num_engagements):
        engagement_id = uuid.uuid4()
        customer_id = random.choice(customers)[0]
        event_id = random.choice(events)[0]
        clicks = random.randint(1, 50)
        purchases = random.randint(0, clicks)  # Purchases can't exceed clicks
        engagement_date = datetime.now() - timedelta(days=random.randint(1, 30))
        engagements.append((engagement_id, customer_id, event_id, clicks, purchases, engagement_date))
    return engagements

# Generate retention data
def generate_retentions(customers, num_records=200):
    retentions = []
    for customer in customers[:num_records]:
        customer_id = customer[0]
        first_purchase_date = datetime.now() - timedelta(days=random.randint(60, 300))
        last_purchase_date = first_purchase_date + timedelta(days=random.randint(10, 200))
        repeat_purchases = random.randint(1, 20)
        retentions.append((customer_id, first_purchase_date, last_purchase_date, repeat_purchases))
    return retentions

# Generate revenues
def generate_revenues(events, num_revenues=50):
    revenues = []
    for _ in range(num_revenues):
        event_id = random.choice(events)[0]
        revenue = round(random.uniform(1000, 20000), 2)
        booking_date = datetime.now() - timedelta(days=random.randint(1, 30))
        revenues.append((event_id, revenue, booking_date))
    return revenues

# Insert synthetic data into the database
def insert_synthetic_data():
    session = connect_to_astra()

    # Generate synthetic data
    customers = generate_customers()
    events = generate_events()
    bookings = generate_bookings(customers, events)
    campaigns = generate_campaigns(events)
    engagements = generate_engagements(customers, events)
    retentions = generate_retentions(customers)
    revenues = generate_revenues(events)
    customer_preferences = generate_customer_preferences(customers, bookings, engagements)

    # Insert data into the respective tables
    for customer in customers:
        session.execute("""
            INSERT INTO customers (customer_id, first_name, last_name, email, phone_number, age, gender, location, join_date, device_type, family_status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, customer)

    for event in events:
        session.execute("""
            INSERT INTO events (event_id, event_name, event_type, event_description, start_date, end_date, max_capacity, location, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, event)

    for booking in bookings:
        session.execute("""
            INSERT INTO bookings (booking_id, booking_channel, booking_date, ctr, customer_id, event_date, event_id, number_of_guests, total_spend, upsells)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, booking)

    for preference in customer_preferences:
        session.execute("""
            INSERT INTO customer_preferences (customer_id, preferred_activities, preferred_event_types, upsell_preferences)
            VALUES (%s, %s, %s, %s)
        """, preference)

    for campaign in campaigns:
        session.execute("""
            INSERT INTO campaigns (campaign_id, event_id, impressions, ad_spend, start_date, end_date)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, campaign)

    for engagement in engagements:
        session.execute("""
            INSERT INTO engagements (engagement_id, customer_id, event_id, clicks, purchases, engagement_date)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, engagement)

    for retention in retentions:
        session.execute("""
            INSERT INTO retentions (customer_id, first_purchase_date, last_purchase_date, repeat_purchases)
            VALUES (%s, %s, %s, %s)
        """, retention)

    for revenue in revenues:
        session.execute("""
            INSERT INTO revenues (event_id, revenue, booking_date)
            VALUES (%s, %s, %s)
        """, revenue)

    print("Synthetic data inserted successfully.")

# Run the function to insert data
insert_synthetic_data()