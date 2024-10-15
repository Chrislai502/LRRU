import sqlite3

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

if __name__ == "__main__":
    # Specify the path to your .db file
    db_path = './eval_datasets/Kitti_eval_base_(-0.0_1.0_1.0).db.db'  # Replace with your actual .db file path
    db_path = "./eval_datasets/20241015/Kitti_eval_mini_(-0.0_1.0_1.0).db"
    db_path = "./eval_datasets/Kitti_eval_base_(-6.0_7.0_2.0).db"
    table_name = 'evaluations'  # Replace with your actual table name
    
    # Count the number of rows in the table
    row_count = count_rows_in_table(db_path, table_name)
    
    print(f"Total number of rows in the '{table_name}' table: {row_count}")
