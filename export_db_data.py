#!/usr/bin/env python3
"""
Database Data Export Utility

This script extracts data from the SQLite database and exports each table to CSV files
in the uploads folder. It also displays metadata information about the database structure.
Enhanced with JSON parsing for better readability of metadata columns.
"""

import sqlite3
import csv
import os
import sys
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union


def connect_to_database(db_path):
    """Connect to the SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # Enable column name access
        return conn
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
        return None


def get_table_names(conn):
    """Get all table names from the database."""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    return [table[0] for table in tables]


def get_table_info(conn, table_name):
    """Get column information for a specific table."""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = cursor.fetchall()
    return columns


def get_table_data(conn, table_name):
    """Get all data from a specific table."""
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table_name};")
    return cursor.fetchall()


def parse_json_field(value: Any, field_name: str = "") -> str:
    """Parse JSON field and return human-readable string."""
    if value is None:
        return ""
    
    if isinstance(value, str):
        try:
            # Try to parse as JSON
            parsed = json.loads(value)
            if isinstance(parsed, list):
                if not parsed:  # Empty list
                    return ""
                # Convert list to comma-separated string
                return ", ".join(str(item) for item in parsed)
            elif isinstance(parsed, dict):
                # Convert dict to key: value pairs
                return "; ".join(f"{k}: {v}" for k, v in parsed.items())
            else:
                return str(parsed)
        except json.JSONDecodeError:
            # Return original string if not valid JSON
            return str(value)
    
    return str(value)


def validate_json_field(value: Any, field_name: str = "") -> Dict[str, Any]:
    """Validate JSON field and return parsing info."""
    result = {
        'original': value,
        'is_json': False,
        'is_valid': False,
        'parsed_value': None,
        'readable_value': "",
        'error': None
    }
    
    if value is None:
        result['readable_value'] = ""
        return result
    
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            result['is_json'] = True
            result['is_valid'] = True
            result['parsed_value'] = parsed
            result['readable_value'] = parse_json_field(value, field_name)
        except json.JSONDecodeError as e:
            result['is_json'] = True  # Looks like JSON but invalid
            result['error'] = str(e)
            result['readable_value'] = str(value)
    else:
        result['readable_value'] = str(value) if value else ""
    
    return result


def export_table_to_csv(conn, table_name, output_dir):
    """Export table data to CSV file with enhanced JSON handling."""
    try:
        # Get table data
        data = get_table_data(conn, table_name)
        
        if not data:
            print(f"  No data found in table '{table_name}'")
            return
        
        # Identify JSON columns (specifically for documents table)
        json_columns = ['main_topics', 'past_events', 'future_actions', 'participants']
        column_names = list(data[0].keys())
        
        # Check which JSON columns exist in this table
        existing_json_columns = [col for col in json_columns if col in column_names]
        
        if existing_json_columns:
            print(f"  Found JSON columns in '{table_name}': {existing_json_columns}")
        
        # Export raw data CSV
        raw_csv_file = os.path.join(output_dir, f"{table_name}_raw.csv")
        
        with open(raw_csv_file, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=column_names)
            writer.writeheader()
            
            for row in data:
                writer.writerow(dict(row))
        
        print(f"  Exported {len(data)} rows (raw format) to '{raw_csv_file}'")
        
        # Export human-readable CSV if JSON columns exist
        if existing_json_columns:
            readable_csv_file = os.path.join(output_dir, f"{table_name}_readable.csv")
            
            with open(readable_csv_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=column_names)
                writer.writeheader()
                
                json_stats = {col: {'total': 0, 'valid_json': 0, 'empty': 0, 'errors': 0} for col in existing_json_columns}
                
                for row in data:
                    processed_row = dict(row)
                    
                    # Process JSON columns
                    for col in existing_json_columns:
                        if col in processed_row:
                            validation = validate_json_field(processed_row[col], col)
                            processed_row[col] = validation['readable_value']
                            
                            # Update statistics
                            json_stats[col]['total'] += 1
                            if validation['is_valid']:
                                json_stats[col]['valid_json'] += 1
                            elif not validation['readable_value']:
                                json_stats[col]['empty'] += 1
                            elif validation['error']:
                                json_stats[col]['errors'] += 1
                    
                    writer.writerow(processed_row)
            
            print(f"  Exported {len(data)} rows (readable format) to '{readable_csv_file}'")
            
            # Display JSON statistics
            print(f"  JSON Column Statistics for '{table_name}':")
            for col, stats in json_stats.items():
                print(f"    {col}: {stats['valid_json']} valid JSON, {stats['empty']} empty, {stats['errors']} errors out of {stats['total']} total")
        else:
            # For non-JSON tables, just create a regular CSV (same as raw)
            csv_file = os.path.join(output_dir, f"{table_name}.csv")
            with open(csv_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=column_names)
                writer.writeheader()
                
                for row in data:
                    writer.writerow(dict(row))
            
            print(f"  Exported {len(data)} rows to '{csv_file}'")
        
    except Exception as e:
        print(f"  Error exporting table '{table_name}': {e}")


def display_database_metadata(conn, table_names):
    """Display comprehensive metadata about the database."""
    print("="*80)
    print("DATABASE METADATA INFORMATION")
    print("="*80)
    
    for table_name in table_names:
        print(f"\nTable: {table_name}")
        print("-" * 50)
        
        # Get column information
        columns = get_table_info(conn, table_name)
        print("Column Information:")
        print(f"{'Column Name':<25} {'Type':<15} {'Not Null':<10} {'Default':<15} {'Primary Key'}")
        print("-" * 80)
        
        for col in columns:
            cid, name, col_type, not_null, default_val, pk = col
            print(f"{name:<25} {col_type:<15} {'YES' if not_null else 'NO':<10} {str(default_val) if default_val else 'NULL':<15} {'YES' if pk else 'NO'}")
        
        # Get row count
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        row_count = cursor.fetchone()[0]
        print(f"\nTotal Rows: {row_count}")
        
        # Show sample data if available with enhanced JSON parsing
        if row_count > 0:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 3;")
            sample_data = cursor.fetchall()
            print(f"\nSample Data (first 3 rows):")
            
            # Check if this table has JSON columns
            json_columns = ['main_topics', 'past_events', 'future_actions', 'participants']
            column_names = [col[1] for col in columns]  # Extract column names
            existing_json_columns = [col for col in json_columns if col in column_names]
            
            for i, row in enumerate(sample_data, 1):
                row_dict = dict(row)
                print(f"Row {i}:")
                
                # Display regular columns
                for key, value in row_dict.items():
                    if key not in existing_json_columns:
                        # Truncate long values for readability
                        display_value = str(value)[:100] + "..." if value and len(str(value)) > 100 else str(value)
                        print(f"  {key}: {display_value}")
                
                # Display JSON columns with enhanced parsing
                for json_col in existing_json_columns:
                    if json_col in row_dict:
                        raw_value = row_dict[json_col]
                        validation = validate_json_field(raw_value, json_col)
                        
                        print(f"  {json_col} (JSON):")
                        print(f"    Raw: {str(raw_value)[:100]}{'...' if raw_value and len(str(raw_value)) > 100 else ''}")
                        if validation['is_valid']:
                            print(f"    Parsed: {validation['readable_value']}")
                            print(f"    Type: {type(validation['parsed_value']).__name__}")
                            if isinstance(validation['parsed_value'], list):
                                print(f"    Count: {len(validation['parsed_value'])} items")
                        elif validation['error']:
                            print(f"    Error: {validation['error']}")
                        else:
                            print(f"    Status: Not JSON data")
                
                print()  # Empty line between rows
        
        print("=" * 80)


def main():
    """Main function to export database data."""
    # Database path
    db_path = "meeting_documents.db"
    sessions_db_path = "sessions.db"
    
    # Output directory
    output_dir = "uploads"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Database Export Utility")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output Directory: {output_dir}")
    print("="*80)
    
    # Process main database
    if os.path.exists(db_path):
        print(f"\nProcessing main database: {db_path}")
        conn = connect_to_database(db_path)
        
        if conn:
            try:
                # Get all table names
                table_names = get_table_names(conn)
                print(f"Found {len(table_names)} tables: {', '.join(table_names)}")
                
                # Display metadata
                display_database_metadata(conn, table_names)
                
                # Export each table to CSV
                print(f"\nExporting tables to CSV files...")
                for table_name in table_names:
                    print(f"Exporting table: {table_name}")
                    export_table_to_csv(conn, table_name, output_dir)
                
            finally:
                conn.close()
        else:
            print(f"Could not connect to database: {db_path}")
    else:
        print(f"Database file not found: {db_path}")
    
    # Process sessions database
    if os.path.exists(sessions_db_path):
        print(f"\nProcessing sessions database: {sessions_db_path}")
        conn = connect_to_database(sessions_db_path)
        
        if conn:
            try:
                # Get all table names
                table_names = get_table_names(conn)
                if table_names:
                    print(f"Found {len(table_names)} session tables: {', '.join(table_names)}")
                    
                    # Display metadata for session tables
                    print("\nSESSIONS DATABASE METADATA")
                    print("="*80)
                    display_database_metadata(conn, table_names)
                    
                    # Export session tables with prefix
                    print(f"\nExporting session tables to CSV files...")
                    for table_name in table_names:
                        print(f"Exporting session table: {table_name}")
                        # Add sessions_ prefix to distinguish from main DB tables
                        export_table_to_csv(conn, table_name, output_dir)
                        # Rename the file to include sessions prefix
                        old_path = os.path.join(output_dir, f"{table_name}.csv")
                        new_path = os.path.join(output_dir, f"sessions_{table_name}.csv")
                        if os.path.exists(old_path):
                            os.rename(old_path, new_path)
                            print(f"  Renamed to 'sessions_{table_name}.csv'")
                else:
                    print("No tables found in sessions database")
                
            finally:
                conn.close()
        else:
            print(f"Could not connect to sessions database: {sessions_db_path}")
    else:
        print(f"Sessions database file not found: {sessions_db_path}")
    
    print(f"\nExport completed! CSV files saved in '{output_dir}' directory.")
    
    # List all created CSV files
    csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
    if csv_files:
        print(f"\nCreated CSV files:")
        raw_files = [f for f in csv_files if '_raw.csv' in f]
        readable_files = [f for f in csv_files if '_readable.csv' in f]
        regular_files = [f for f in csv_files if '_raw.csv' not in f and '_readable.csv' not in f]
        
        if raw_files:
            print(f"\n  Raw Format (original JSON strings):")
            for csv_file in sorted(raw_files):
                file_path = os.path.join(output_dir, csv_file)
                file_size = os.path.getsize(file_path)
                print(f"    {csv_file} ({file_size:,} bytes)")
        
        if readable_files:
            print(f"\n  Human-Readable Format (parsed JSON):")
            for csv_file in sorted(readable_files):
                file_path = os.path.join(output_dir, csv_file)
                file_size = os.path.getsize(file_path)
                print(f"    {csv_file} ({file_size:,} bytes)")
        
        if regular_files:
            print(f"\n  Standard Format (no JSON columns):")
            for csv_file in sorted(regular_files):
                file_path = os.path.join(output_dir, csv_file)
                file_size = os.path.getsize(file_path)
                print(f"    {csv_file} ({file_size:,} bytes)")
        
        print(f"\n  Total files created: {len(csv_files)}")
        
        # Add usage instructions
        print(f"\nUsage Notes:")
        print(f"  - *_raw.csv files contain original JSON strings (for data integrity)")
        print(f"  - *_readable.csv files contain parsed JSON as comma-separated values (for spreadsheet use)")
        print(f"  - Tables without JSON columns are exported as standard CSV files")
        print(f"  - JSON columns: main_topics, past_events, future_actions, participants")
    else:
        print(f"\nNo CSV files were created (no data found in database tables).")


if __name__ == "__main__":
    main()