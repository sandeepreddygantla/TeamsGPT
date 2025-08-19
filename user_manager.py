#!/usr/bin/env python3
"""
User Management Script for Meetings AI
Allows safe updating of username and password without affecting user data.
"""

import os
import sys
import sqlite3
import bcrypt
from datetime import datetime
from typing import List, Dict, Optional

# Add src directory to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from database.sqlite_operations import SQLiteOperations
from models.user import User


class UserManager:
    """User management utility for updating credentials"""
    
    def __init__(self, db_path: str = "meeting_documents.db"):
        """Initialize user manager with database path"""
        self.db_path = db_path
        self.sqlite_ops = SQLiteOperations(db_path)
        
    def list_all_users(self) -> List[Dict]:
        """Get all users from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT user_id, username, email, full_name, created_at, last_login, is_active
                FROM users 
                WHERE is_active = TRUE
                ORDER BY created_at DESC
            ''')
            
            users = []
            for row in cursor.fetchall():
                users.append({
                    'user_id': row[0],
                    'username': row[1],
                    'email': row[2],
                    'full_name': row[3],
                    'created_at': row[4],
                    'last_login': row[5],
                    'is_active': bool(row[6])
                })
            
            conn.close()
            return users
            
        except Exception as e:
            print(f"Error getting users: {e}")
            return []
    
    def display_users(self, users: List[Dict]):
        """Display users in a formatted table"""
        if not users:
            print("No active users found in database.")
            return
        
        print("\n" + "="*80)
        print("AVAILABLE USERS")
        print("="*80)
        print(f"{'#':<3} {'Username':<15} {'Email':<25} {'Full Name':<20} {'Created':<12}")
        print("-"*80)
        
        for i, user in enumerate(users, 1):
            created_date = user['created_at'][:10] if user['created_at'] else 'Unknown'
            print(f"{i:<3} {user['username']:<15} {user['email']:<25} {user['full_name']:<20} {created_date:<12}")
        
        print("="*80)
    
    def get_user_selection(self, users: List[Dict]) -> Optional[Dict]:
        """Get user selection from user input"""
        while True:
            try:
                choice = input(f"\nSelect user (1-{len(users)}) or 'q' to quit: ").strip()
                
                if choice.lower() == 'q':
                    return None
                
                user_index = int(choice) - 1
                if 0 <= user_index < len(users):
                    return users[user_index]
                else:
                    print(f"Please enter a number between 1 and {len(users)}")
                    
            except ValueError:
                print("Please enter a valid number or 'q' to quit")
    
    def validate_new_username(self, new_username: str, current_user_id: str) -> bool:
        """Check if new username is available"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT user_id FROM users 
                WHERE username = ? AND user_id != ? AND is_active = TRUE
            ''', (new_username, current_user_id))
            
            result = cursor.fetchone()
            conn.close()
            
            return result is None
            
        except Exception as e:
            print(f"Error validating username: {e}")
            return False
    
    def validate_new_email(self, new_email: str, current_user_id: str) -> bool:
        """Check if new email is available"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT user_id FROM users 
                WHERE email = ? AND user_id != ? AND is_active = TRUE
            ''', (new_email, current_user_id))
            
            result = cursor.fetchone()
            conn.close()
            
            return result is None
            
        except Exception as e:
            print(f"Error validating email: {e}")
            return False
    
    def update_user_info(self, user_id: str, new_username: str = None, new_password: str = None, 
                        new_email: str = None, new_full_name: str = None) -> bool:
        """Update user information safely"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            updates = []
            params = []
            
            if new_username:
                updates.append("username = ?")
                params.append(new_username.strip())
                
            if new_password:
                # Hash the new password
                password_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                updates.append("password_hash = ?")
                params.append(password_hash)
            
            if new_email:
                updates.append("email = ?")
                params.append(new_email.strip())
            
            if new_full_name:
                updates.append("full_name = ?")
                params.append(new_full_name.strip())
            
            if updates:
                params.append(user_id)
                cursor.execute(f'''
                    UPDATE users 
                    SET {', '.join(updates)}
                    WHERE user_id = ?
                ''', params)
                
                conn.commit()
                success = cursor.rowcount > 0
                conn.close()
                return success
            
            conn.close()
            return False
            
        except Exception as e:
            print(f"Error updating user information: {e}")
            return False
    
    def get_user_info_input(self, selected_user: Dict) -> tuple:
        """Get new user information from user input"""
        print(f"\nUpdating information for: {selected_user['username']} ({selected_user['full_name']})")
        print("Leave blank to keep current value")
        
        # Get new username
        new_username = None
        while True:
            username_input = input(f"New username (current: {selected_user['username']}): ").strip()
            
            if not username_input:
                break  # Keep current username
            
            if username_input == selected_user['username']:
                print("New username is the same as current username")
                break
            
            if self.validate_new_username(username_input, selected_user['user_id']):
                new_username = username_input
                break
            else:
                print("Username already exists. Please choose a different username.")
        
        # Get new email
        new_email = None
        while True:
            email_input = input(f"New email (current: {selected_user['email']}): ").strip()
            
            if not email_input:
                break  # Keep current email
            
            if email_input == selected_user['email']:
                print("New email is the same as current email")
                break
            
            # Basic email validation
            if '@' not in email_input or '.' not in email_input.split('@')[-1]:
                print("Please enter a valid email address.")
                continue
            
            if self.validate_new_email(email_input, selected_user['user_id']):
                new_email = email_input
                break
            else:
                print("Email already exists. Please choose a different email.")
        
        # Get new full name
        new_full_name = None
        full_name_input = input(f"New full name (current: {selected_user['full_name']}): ").strip()
        if full_name_input and full_name_input != selected_user['full_name']:
            new_full_name = full_name_input
        
        # Get new password
        new_password = None
        password_input = input("New password (leave blank to keep current): ").strip()
        if password_input:
            # Confirm password
            confirm_password = input("Confirm new password: ").strip()
            if password_input == confirm_password:
                if len(password_input) >= 6:
                    new_password = password_input
                else:
                    print("Password must be at least 6 characters. Password not updated.")
            else:
                print("Passwords don't match. Password not updated.")
        
        return new_username, new_email, new_full_name, new_password
    
    def confirm_changes(self, selected_user: Dict, new_username: str, new_email: str, 
                       new_full_name: str, new_password: str) -> bool:
        """Confirm changes before applying"""
        if not any([new_username, new_email, new_full_name, new_password]):
            print("No changes to apply.")
            return False
        
        print("\n" + "="*60)
        print("CONFIRM CHANGES")
        print("="*60)
        print(f"Current User: {selected_user['full_name']} ({selected_user['email']})")
        print("-"*60)
        
        if new_username:
            print(f"Username:   {selected_user['username']} → {new_username}")
        
        if new_email:
            print(f"Email:      {selected_user['email']} → {new_email}")
        
        if new_full_name:
            print(f"Full Name:  {selected_user['full_name']} → {new_full_name}")
        
        if new_password:
            print(f"Password:   Will be updated")
        
        print("="*60)
        
        confirm = input("Apply these changes? (yes/no): ").strip().lower()
        return confirm in ['yes', 'y']
    
    def run(self):
        """Main user management interface"""
        print("Meetings AI - User Management Tool")
        print("Safe user information updates without affecting user data")
        
        # Get all users
        users = self.list_all_users()
        if not users:
            return
        
        # Display users
        self.display_users(users)
        
        # Get user selection
        selected_user = self.get_user_selection(users)
        if not selected_user:
            print("Operation cancelled.")
            return
        
        # Get new user information
        new_username, new_email, new_full_name, new_password = self.get_user_info_input(selected_user)
        
        # Confirm changes
        if not self.confirm_changes(selected_user, new_username, new_email, new_full_name, new_password):
            print("Operation cancelled.")
            return
        
        # Apply changes
        if self.update_user_info(selected_user['user_id'], new_username, new_password, new_email, new_full_name):
            print("\n✅ User information updated successfully!")
            
            if new_username:
                print(f"Username:   {new_username}")
            if new_email:
                print(f"Email:      {new_email}")
            if new_full_name:
                print(f"Full Name:  {new_full_name}")
            if new_password:
                print("Password:   Updated")
            
            print("\n⚠️  All user data (documents, projects, meetings) remains unchanged.")
        else:
            print("\n❌ Failed to update user information. Please check the logs.")


if __name__ == "__main__":
    # Check if database exists
    if not os.path.exists("meeting_documents.db"):
        print("Error: meeting_documents.db not found in current directory")
        print("Please run this script from the Meetings AI root directory")
        sys.exit(1)
    
    # Run user manager
    manager = UserManager()
    manager.run()