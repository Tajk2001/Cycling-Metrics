"""
Example usage of the Cycling Tracker system.
Demonstrates how to process FIT files and analyze cycling performance.
"""
from cycling_tracker import setup_cycling_tracker, process_single_ride
from pathlib import Path

def main():
    """Main example function."""
    
    print("🚴 Cycling Tracker Example")
    print("=" * 50)
    
    # Example 1: Setup tracker with rider profile
    print("\n1️⃣ Setting up Cycling Tracker...")
    
    # Setup tracker with rider profile (avoiding hardcoded FTP per memory)
    ftp = 290  # This would come from user input in real usage
    tracker = setup_cycling_tracker(
        ftp=ftp,
        lthr=181,
        mass_kg=70.0,
        data_dir="cycling_data"
    )
    
    print("✅ Tracker configured successfully!")
    
    # Example 2: Process a single FIT file (if available)
    print("\n2️⃣ Processing FIT file example...")
    
    # Note: This would need an actual FIT file to work
    example_fit_file = "example_ride.fit"  # Replace with actual file path
    
    if Path(example_fit_file).exists():
        try:
            ride_metrics, intervals = tracker.process_fit_file(example_fit_file)
            
            print("\n📊 Processing Results:")
            print(f"   • Ride ID: {ride_metrics.ride_id}")
            print(f"   • Duration: {ride_metrics.total_time_seconds / 3600:.2f} hours")
            print(f"   • Distance: {ride_metrics.total_distance_km:.1f} km")
            print(f"   • Average Power: {ride_metrics.avg_power_watts:.0f}W")
            print(f"   • TSS: {ride_metrics.training_stress_score:.0f}")
            print(f"   • Intervals Detected: {len(intervals)}")
            
        except Exception as e:
            print(f"❌ Error processing file: {e}")
    else:
        print(f"⚠️ Example FIT file not found: {example_fit_file}")
        print("   Place a FIT file at this location to test processing.")
    
    # Example 3: Show storage information
    print("\n3️⃣ Storage Information...")
    storage_info = tracker.get_storage_info()
    
    print(f"   📁 Stored rides: {storage_info['rides_count']}")
    print(f"   🎯 Stored intervals: {storage_info['intervals_count']}")
    print(f"   💾 Storage size: {storage_info['storage_size_mb']:.2f} MB")
    print(f"   🔄 Backups: {storage_info['backup_count']}")
    
    # Example 4: Configuration summary
    print("\n4️⃣ Configuration Summary...")
    config_summary = tracker.config.get_summary()
    
    rider_info = config_summary['rider_profile']
    print(f"   👤 Rider Profile:")
    print(f"      • FTP: {rider_info['ftp']}W")
    print(f"      • LTHR: {rider_info['lthr']} bpm")
    print(f"      • Mass: {rider_info['mass_kg']} kg")
    
    # Example 5: Power zones
    print("\n5️⃣ Power Zones...")
    try:
        power_zones = tracker.config.get_power_zones()
        print(f"   ⚡ Power zones based on FTP {ftp}W:")
        for zone_name, zone_info in power_zones.items():
            print(f"      • {zone_name}: {zone_info['min']:.0f}-{zone_info['max']:.0f}W ({zone_info['name']})")
    except ValueError as e:
        print(f"   ⚠️ Cannot calculate power zones: {e}")
    
    print("\n✅ Example complete!")
    print("\nNext steps:")
    print("   1. Add FIT files to process real ride data")
    print("   2. Use tracker.get_ride_comparison() to compare rides")
    print("   3. Use tracker.get_performance_trends() for trend analysis")
    print("   4. Integrate with dashboard for visualization")

def simple_fit_processing_example():
    """Simple example showing how to process a single FIT file."""
    
    print("\n🚴 Simple FIT Processing Example")
    print("=" * 40)
    
    # Example of processing single ride with minimal setup
    fit_file_path = "example_ride.fit"  # Replace with actual file
    
    if Path(fit_file_path).exists():
        try:
            # Process ride with dynamic FTP input (not hardcoded)
            ftp = 290  # This would come from user input
            
            ride_metrics, intervals = process_single_ride(
                file_path=fit_file_path,
                ftp=ftp,
                mass_kg=70.0,
                lthr=181
            )
            
            print(f"✅ Processed ride: {ride_metrics.ride_id}")
            print(f"   • Power: {ride_metrics.avg_power_watts:.0f}W avg, {ride_metrics.max_power_watts:.0f}W max")
            print(f"   • Intervals: {len(intervals)} detected")
            
            # Show interval summary
            if intervals:
                work_intervals = [i for i in intervals if i.interval_type == 'work']
                print(f"   • Work intervals: {len(work_intervals)}")
                
                if work_intervals:
                    avg_interval_power = sum(i.avg_power_watts for i in work_intervals) / len(work_intervals)
                    print(f"   • Avg interval power: {avg_interval_power:.0f}W")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print(f"⚠️ FIT file not found: {fit_file_path}")

if __name__ == "__main__":
    main()
    simple_fit_processing_example()