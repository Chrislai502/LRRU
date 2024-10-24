import sqlite3
import os

def list_tables_and_schemas(db_path):
    """List all tables, their schemas, and first 10 rows in a given SQLite database."""
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    
    # Create a cursor object to execute SQL queries
    cursor = conn.cursor()

    # Get the list of all tables in the database
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    # Iterate through each table and print its schema
    for table in tables:
        table_name = table[0]
        print(f"\nTable: {table_name}")
        
        # Get the schema for each table
        cursor.execute(f"PRAGMA table_info({table_name});")
        schema = cursor.fetchall()

        # Print the schema in a readable format
        print(f"Schema for '{table_name}':")
        for column in schema:
            print(f"  {column[1]} ({column[2]})")

        # Fetch the first 10 rows from the table
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 20")
        rows = cursor.fetchall()

        print(f"\nFirst 10 entries in the '{table_name}' table:")
        for row in rows:
            print(row)

    # Close the connection
    conn.close()

def count_rows_in_table(db_path, table_name):
    """Count the number of rows in a given table within a .db file."""
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    
    # Create a cursor object to execute SQL queries
    cursor = conn.cursor()
    
    # Count rows in the specified table
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    row_count = cursor.fetchone()[0]
    
    # Close the connection
    conn.close()
    
    return row_count

def drop_table(db_path, table_name):
    """Drop a table from the SQLite database."""
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        
        # Create a cursor object to execute SQL queries
        cursor = conn.cursor()
        
        # Drop the table
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        
        # Commit the changes
        conn.commit()
        print(f"Table '{table_name}' has been dropped successfully.")
    
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
    
    finally:
        # Close the connection
        conn.close()

if __name__ == "__main__":
    # Example: Specify the path to your .db file
    db_path = './eval_datasets/Kitti_eval_base_(-1.0_1.0_1.0).db'  # Replace with your actual .db file path
    if not os.path.exists(db_path):
        print(f"Error: {db_path} does not exist.")
        exit()

    # List all tables, schemas, and first 10 rows in the database
    list_tables_and_schemas(db_path)
    
    # Example: Count the number of rows in a specific table (if known)
    table_name = 'evaluations'  # Replace with your actual table name if you know it
    row_count = count_rows_in_table(db_path, table_name)
    
    print(f"\nTotal number of rows in the '{table_name}' table: {row_count}")
    
    # # Drop a table from the database
    # table_name = 'statistics'  # Replace with your actual table name if you know it
    # drop_table(db_path, table_name)
