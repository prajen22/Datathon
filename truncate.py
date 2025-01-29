from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from config import astra_client_id, astra_client_secret, astra_app_name

# Astra DB Configuration
ASTRA_DB_CLIENT_ID = astra_client_id
ASTRA_DB_CLIENT_SECRET = astra_client_secret
ASTRA_DB_SECURE_CONNECT_BUNDLE_PATH = "secure-connect-datathon.zip"
ASTRA_KEYSPACE = astra_app_name  # Replace with your keyspace name

# Connect to Astra DB
def connect_to_astra():
    auth_provider = PlainTextAuthProvider(ASTRA_DB_CLIENT_ID, ASTRA_DB_CLIENT_SECRET)
    cluster = Cluster(cloud={'secure_connect_bundle': ASTRA_DB_SECURE_CONNECT_BUNDLE_PATH}, auth_provider=auth_provider)
    session = cluster.connect()
    session.set_keyspace(ASTRA_KEYSPACE)  # Set the keyspace
    return session

# Function to truncate explicitly defined tables
def truncate_specified_tables():
    session = connect_to_astra()

    # List of tables to truncate (explicitly defined)
    tables_to_truncate = [
        "customers",
        "events",
        "bookings",
        "customer_preferences", 
        "campaigns",
        "engagements",
        "retentions",
        "revenues"
    ]
    
    # Loop through the tables and truncate each
    for table_name in tables_to_truncate:
        try:
            print(f"Truncating table: {table_name}")
            truncate_query = f"TRUNCATE {ASTRA_KEYSPACE}.{table_name}"
            session.execute(truncate_query)
        except Exception as e:
            print(f"Error truncating table {table_name}: {str(e)}")
    
    print("Specified tables truncated successfully!")

# Run the function to truncate specified tables
truncate_specified_tables()