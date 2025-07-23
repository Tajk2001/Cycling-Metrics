#!/usr/bin/env python3
"""
Test script to actually test the delete functionality with a real ride.
"""

import sys
import pathlib
from data_manager import CyclingDataManager

def test_delete_actual():
    """Test the delete functionality with actual data."""
    print("🧪 Testing Actual Delete Functionality...")
    
    try:
        # Initialize data manager
        dm = CyclingDataManager()
        print("✅ Data manager initialized successfully")
        
        # Get current status
        status = dm.get_system_status()
        print(f"📊 Current Status:")
        print(f"  - Total rides: {status['total_rides']}")
        print(f"  - Cached files: {status['cached_files']}")
        print(f"  - History entries: {status['history_entries']}")
        print(f"  - Analysis entries: {status['analysis_entries']}")
        
        # Get available rides
        available_rides = dm.get_available_rides()
        print(f"📈 Available rides: {len(available_rides)}")
        for ride in available_rides:
            print(f"  - {ride}")
        
        if not available_rides:
            print("⚠️ No rides available to test deletion")
            return True
        
        # Test delete specific ride (use first available ride)
        test_ride = available_rides[0]
        print(f"\n🗑️ Testing delete ride: {test_ride}")
        
        # Get ride data before deletion
        ride_data_before = dm.get_ride_data(test_ride)
        print(f"  - Has FIT file: {ride_data_before['has_fit_file']}")
        print(f"  - In history: {ride_data_before['in_history']}")
        print(f"  - Analysis available: {ride_data_before['analysis_available']}")
        
        # Ask user for confirmation
        print(f"\n⚠️  WARNING: This will actually delete '{test_ride}' and all associated data!")
        print("Type 'DELETE' to confirm, or anything else to cancel:")
        user_input = input().strip()
        
        if user_input == "DELETE":
            print(f"🗑️ Deleting {test_ride}...")
            success, message = dm.delete_ride(test_ride)
            print(f"Result: {message}")
            
            if success:
                # Check status after deletion
                status_after = dm.get_system_status()
                available_rides_after = dm.get_available_rides()
                
                print(f"\n📊 Status After Deletion:")
                print(f"  - Total rides: {status_after['total_rides']}")
                print(f"  - Available rides: {len(available_rides_after)}")
                
                if test_ride not in available_rides_after:
                    print("✅ Success: Ride was deleted!")
                else:
                    print("❌ Error: Ride still appears in available rides")
            else:
                print("❌ Deletion failed")
        else:
            print("❌ Deletion cancelled by user")
        
        print("\n🎉 Delete functionality test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_delete_actual()
    sys.exit(0 if success else 1) 