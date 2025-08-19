#!/usr/bin/env python3
"""
User Removal Script for Meetings AI
Safely removes users and their associated data without affecting other users.
"""

import os
import sys
import sqlite3
import shutil
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# Add src directory to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from database.sqlite_operations import SQLiteOperations
from database.vector_operations import VectorOperations
from models.user import User


class UserRemoval:
    """Safe user removal utility"""
    
    def __init__(self, db_path: str = "meeting_documents.db", index_path: str = "vector_index.faiss"):
        """Initialize user removal tool"""
        self.db_path = db_path
        self.index_path = index_path
        self.sqlite_ops = SQLiteOperations(db_path)
        self.vector_ops = VectorOperations(index_path, 3072) if os.path.exists(index_path) else None
        
    def get_user_data_summary(self, user_id: str) -> Dict:
        """Get summary of user's data for impact assessment"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            summary = {
                'documents': 0,
                'projects': 0,
                'meetings': 0,
                'chunks': 0,
                'chat_sessions': 0,
                'storage_mb': 0
            }
            
            # Count documents
            cursor.execute('SELECT COUNT(*), COALESCE(SUM(file_size), 0) FROM documents WHERE user_id = ?', (user_id,))
            doc_result = cursor.fetchone()
            summary['documents'] = doc_result[0] if doc_result else 0
            summary['storage_mb'] = round((doc_result[1] if doc_result[1] else 0) / (1024 * 1024), 2)
            
            # Count projects
            cursor.execute('SELECT COUNT(*) FROM projects WHERE user_id = ?', (user_id,))
            proj_result = cursor.fetchone()
            summary['projects'] = proj_result[0] if proj_result else 0
            
            # Count meetings
            cursor.execute('SELECT COUNT(*) FROM meetings WHERE user_id = ?', (user_id,))
            meet_result = cursor.fetchone()
            summary['meetings'] = meet_result[0] if meet_result else 0
            
            # Count chunks
            cursor.execute('SELECT COUNT(*) FROM chunks WHERE user_id = ?', (user_id,))
            chunk_result = cursor.fetchone()
            summary['chunks'] = chunk_result[0] if chunk_result else 0
            
            # Count chat sessions (if table exists)
            try:
                cursor.execute('SELECT COUNT(*) FROM chat_sessions WHERE user_id = ?', (user_id,))
                chat_result = cursor.fetchone()
                summary['chat_sessions'] = chat_result[0] if chat_result else 0
            except sqlite3.OperationalError:
                # Table might not exist
                summary['chat_sessions'] = 0
            
            conn.close()
            return summary
            
        except Exception as e:
            print(f"Error getting user data summary: {e}")
            return summary
    
    def list_users_with_data(self) -> List[Dict]:
        """Get all users with their data summaries"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT user_id, username, email, full_name, created_at, last_login, is_active
                FROM users 
                ORDER BY created_at DESC
            ''')
            
            users = []
            for row in cursor.fetchall():
                user_data = {
                    'user_id': row[0],
                    'username': row[1],
                    'email': row[2],
                    'full_name': row[3],
                    'created_at': row[4],
                    'last_login': row[5],
                    'is_active': bool(row[6])
                }
                
                # Get data summary
                user_data['data_summary'] = self.get_user_data_summary(row[0])
                users.append(user_data)
            
            conn.close()
            return users
            
        except Exception as e:
            print(f"Error getting users: {e}")
            return []
    
    def display_users_with_data(self, users: List[Dict]):
        """Display users with their data counts"""
        if not users:
            print("No users found in database.")
            return
        
        print("\n" + "="*100)
        print("USERS AND THEIR DATA")
        print("="*100)
        print(f"{'#':<3} {'Username':<15} {'Full Name':<20} {'Status':<8} {'Docs':<5} {'Proj':<5} {'Meet':<5} {'MB':<6}")
        print("-"*100)
        
        for i, user in enumerate(users, 1):
            status = "Active" if user['is_active'] else "Inactive"
            summary = user['data_summary']
            
            print(f"{i:<3} {user['username']:<15} {user['full_name']:<20} {status:<8} "
                  f"{summary['documents']:<5} {summary['projects']:<5} {summary['meetings']:<5} {summary['storage_mb']:<6}")
        
        print("="*100)
        print("Legend: Docs=Documents, Proj=Projects, Meet=Meetings, MB=Storage Size")
    
    def get_user_selection(self, users: List[Dict]) -> Optional[Dict]:
        """Get user selection for removal"""
        while True:
            try:
                choice = input(f"\nSelect user to remove (1-{len(users)}) or 'q' to quit: ").strip()
                
                if choice.lower() == 'q':
                    return None
                
                user_index = int(choice) - 1
                if 0 <= user_index < len(users):
                    return users[user_index]
                else:
                    print(f"Please enter a number between 1 and {len(users)}")
                    
            except ValueError:
                print("Please enter a valid number or 'q' to quit")
    
    def display_removal_impact(self, user: Dict):
        """Display detailed impact of user removal"""
        summary = user['data_summary']
        
        print("\n" + "="*70)
        print("USER REMOVAL IMPACT")
        print("="*70)
        print(f"User:        {user['full_name']} ({user['username']})")
        print(f"Email:       {user['email']}")
        print(f"Status:      {'Active' if user['is_active'] else 'Inactive'}")
        print("-"*70)
        print("DATA TO BE DELETED:")
        print(f"• {summary['documents']} documents ({summary['storage_mb']} MB)")
        print(f"• {summary['projects']} projects")
        print(f"• {summary['meetings']} meetings")
        print(f"• {summary['chunks']} document chunks")
        print(f"• {summary['chat_sessions']} chat sessions")
        print("-"*70)
        print("⚠️  WARNING: This action cannot be undone!")
        print("⚠️  All user data will be permanently deleted!")
        print("✅ Other users and their data will NOT be affected.")
        print("="*70)
    
    def choose_removal_type(self) -> str:
        """Choose between soft delete (deactivate) or hard delete"""
        print("\nRemoval Options:")
        print("1. Soft Delete (Deactivate) - User cannot login but data remains")
        print("2. Hard Delete (Complete)   - User and ALL data permanently removed")
        
        while True:
            choice = input("Choose removal type (1 or 2): ").strip()
            if choice == '1':
                return 'soft'
            elif choice == '2':
                return 'hard'
            else:
                print("Please enter 1 or 2")
    
    def soft_delete_user(self, user_id: str) -> bool:
        """Deactivate user (soft delete)"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE users 
                SET is_active = FALSE
                WHERE user_id = ?
            ''', (user_id,))
            
            conn.commit()
            success = cursor.rowcount > 0
            conn.close()
            
            return success
            
        except Exception as e:
            print(f"Error deactivating user: {e}")
            return False
    
    def hard_delete_user_data(self, user_id: str) -> Tuple[bool, str]:
        """Completely remove user and all associated data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Start transaction
            cursor.execute('BEGIN TRANSACTION')
            
            deleted_items = []
            
            # Delete chunks and collect chunk_ids for vector deletion
            cursor.execute('SELECT chunk_id FROM chunks WHERE user_id = ?', (user_id,))
            chunk_ids = [row[0] for row in cursor.fetchall()]
            
            if chunk_ids:
                cursor.execute('DELETE FROM chunks WHERE user_id = ?', (user_id,))
                deleted_items.append(f"{cursor.rowcount} chunks")
            
            # Delete documents
            cursor.execute('DELETE FROM documents WHERE user_id = ?', (user_id,))
            if cursor.rowcount > 0:
                deleted_items.append(f"{cursor.rowcount} documents")
            
            # Delete meetings
            cursor.execute('DELETE FROM meetings WHERE user_id = ?', (user_id,))
            if cursor.rowcount > 0:
                deleted_items.append(f"{cursor.rowcount} meetings")
            
            # Delete projects
            cursor.execute('DELETE FROM projects WHERE user_id = ?', (user_id,))
            if cursor.rowcount > 0:
                deleted_items.append(f"{cursor.rowcount} projects")
            
            # Delete chat sessions (if table exists)
            try:
                cursor.execute('DELETE FROM chat_sessions WHERE user_id = ?', (user_id,))
                if cursor.rowcount > 0:
                    deleted_items.append(f"{cursor.rowcount} chat sessions")
            except sqlite3.OperationalError:
                pass  # Table might not exist
            
            # Delete from deleted_documents table
            try:
                cursor.execute('DELETE FROM deleted_documents WHERE user_id = ? OR deleted_by = ?', (user_id, user_id))
                if cursor.rowcount > 0:
                    deleted_items.append(f"{cursor.rowcount} deleted document records")
            except sqlite3.OperationalError:
                pass  # Table might not exist
            
            # Finally, delete the user
            cursor.execute('DELETE FROM users WHERE user_id = ?', (user_id,))
            if cursor.rowcount > 0:
                deleted_items.append("user account")
            
            # Commit transaction
            cursor.execute('COMMIT')
            conn.close()
            
            # Remove vectors from FAISS index
            if self.vector_ops and chunk_ids:
                try:
                    # Note: FAISS doesn't support individual vector deletion easily
                    # In a production system, you'd need to rebuild the index
                    print("⚠️  Note: Vector embeddings will be cleaned up on next application restart")
                except Exception as e:
                    print(f"Warning: Could not clean up vector embeddings: {e}")
            
            return True, f"Deleted: {', '.join(deleted_items)}"
            
        except Exception as e:
            try:
                cursor.execute('ROLLBACK')
                conn.close()
            except:
                pass
            return False, f"Error during deletion: {e}"
    
    def confirm_removal(self, user: Dict, removal_type: str) -> bool:
        """Final confirmation before removal"""
        action = "deactivated" if removal_type == 'soft' else "permanently deleted"
        
        print(f"\n🔴 FINAL CONFIRMATION")
        print(f"User '{user['username']}' and associated data will be {action}")
        
        # Triple confirmation for hard delete
        if removal_type == 'hard':
            confirm1 = input("Type 'DELETE' to confirm permanent removal: ").strip()
            if confirm1 != 'DELETE':
                return False
            
            confirm2 = input(f"Type the username '{user['username']}' to confirm: ").strip()
            if confirm2 != user['username']:
                return False
            
            confirm3 = input("Type 'YES' for final confirmation: ").strip().upper()
            return confirm3 == 'YES'
        else:
            confirm = input("Type 'DEACTIVATE' to confirm: ").strip().upper()
            return confirm == 'DEACTIVATE'
    
    def create_backup(self) -> bool:
        """Create backup before user removal"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_db = f"backup_meeting_documents_{timestamp}.db"
            backup_vector = f"backup_vector_index_{timestamp}.faiss"
            
            # Backup SQLite database
            shutil.copy2(self.db_path, backup_db)
            print(f"✅ Database backed up to: {backup_db}")
            
            # Backup FAISS index if exists
            if os.path.exists(self.index_path):
                shutil.copy2(self.index_path, backup_vector)
                print(f"✅ Vector index backed up to: {backup_vector}")
            
            return True
            
        except Exception as e:
            print(f"❌ Backup failed: {e}")
            return False
    
    def run(self):
        """Main user removal interface"""
        print("Meetings AI - User Removal Tool")
        print("⚠️  CAUTION: This tool permanently removes users and their data")
        
        # Offer backup option
        backup_choice = input("\nCreate backup before proceeding? (recommended) [Y/n]: ").strip().lower()
        if backup_choice != 'n':
            if not self.create_backup():
                proceed = input("Backup failed. Continue anyway? [y/N]: ").strip().lower()
                if proceed != 'y':
                    print("Operation cancelled for safety.")
                    return
        
        # Get all users with data
        users = self.list_users_with_data()
        if not users:
            return
        
        # Display users
        self.display_users_with_data(users)
        
        # Get user selection
        selected_user = self.get_user_selection(users)
        if not selected_user:
            print("Operation cancelled.")
            return
        
        # Show impact
        self.display_removal_impact(selected_user)
        
        # Choose removal type
        removal_type = self.choose_removal_type()
        
        # Final confirmation
        if not self.confirm_removal(selected_user, removal_type):
            print("Operation cancelled.")
            return
        
        # Execute removal
        if removal_type == 'soft':
            if self.soft_delete_user(selected_user['user_id']):
                print(f"\n✅ User '{selected_user['username']}' has been deactivated")
                print("📝 Data remains in database but user cannot login")
            else:
                print(f"\n❌ Failed to deactivate user '{selected_user['username']}'")
        else:
            success, message = self.hard_delete_user_data(selected_user['user_id'])
            if success:
                print(f"\n✅ User '{selected_user['username']}' has been permanently removed")
                print(f"📝 {message}")
                print("⚠️  Consider restarting the application to clean up vector embeddings")
            else:
                print(f"\n❌ Failed to remove user '{selected_user['username']}'")
                print(f"📝 {message}")


if __name__ == "__main__":
    # Check if database exists
    if not os.path.exists("meeting_documents.db"):
        print("Error: meeting_documents.db not found in current directory")
        print("Please run this script from the Meetings AI root directory")
        sys.exit(1)
    
    # Run user removal tool
    removal_tool = UserRemoval()
    removal_tool.run()