#!/usr/bin/env python3
"""
Simple USD inspector for F1TENTH vehicle.
Usage (in container):
    ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/f1tenth/inspect_f1tenth_usd.py
Or simply check if file exists:
    ls -lh source/isaaclab_assets/isaaclab_assets/f1tenth/f1tenth.usd
"""

import os
import sys

# Check if USD file exists first
usd_path = "/workspace/isaaclab/source/isaaclab_assets/isaaclab_assets/f1tenth/f1tenth.usd"

if not os.path.exists(usd_path):
    print(f"❌ ERROR: USD file not found at {usd_path}")
    print("\nPlease convert URDF to USD first:")
    print("  cd source/isaaclab_assets/isaaclab_assets/f1tenth")
    print("  ./convert_to_usd.sh")
    sys.exit(1)

# Try to import pxr (only available in Isaac Sim Python)
try:
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app

    from pxr import Usd, UsdGeom
except ImportError:
    print("❌ ERROR: This script must be run with Isaac Sim Python")
    print("\nUsage:")
    print("  ./isaaclab.sh -p source/isaaclab_assets/isaaclab_assets/f1tenth/inspect_f1tenth_usd.py")
    sys.exit(1)

print("="*80)
print(f"Inspecting USD: {usd_path}")
print("="*80)

# Open USD stage
stage = Usd.Stage.Open(usd_path)

if not stage:
    print(f"ERROR: Could not open USD file at {usd_path}")
    exit(1)

print("\n📦 USD Structure:")
print("-"*80)

# Find all prims
all_prims = [prim for prim in stage.Traverse()]
print(f"Total prims: {len(all_prims)}\n")

# Find links
print("🔗 Links:")
links = [prim for prim in all_prims if "link" in str(prim.GetPath()).lower()]
for link in links[:10]:  # Show first 10
    print(f"  - {link.GetPath()}")
if len(links) > 10:
    print(f"  ... and {len(links)-10} more")

# Find joints
print(f"\n⚙️  Joints:")
joints = [prim for prim in all_prims if prim.GetTypeName() in ['PhysicsRevoluteJoint', 'PhysicsJoint']]
for joint in joints:
    print(f"  - {joint.GetPath()} [{joint.GetTypeName()}]")

# Find rigid bodies
print(f"\n🏋️  Rigid Bodies:")
rigid_bodies = [prim for prim in all_prims if UsdGeom.Mesh(prim)]
for rb in rigid_bodies[:10]:  # Show first 10
    print(f"  - {rb.GetPath()}")
if len(rigid_bodies) > 10:
    print(f"  ... and {len(rigid_bodies)-10} more")

# Check for specific components
print(f"\n🔍 Checking F1TENTH Components:")
print("-"*80)

components = {
    "base_link": False,
    "laser/lidar": False,
    "steering": False,
    "wheel": False,
}

for prim in all_prims:
    path_str = str(prim.GetPath()).lower()
    if "base_link" in path_str:
        components["base_link"] = True
        print(f"  ✅ Base link found: {prim.GetPath()}")
    if "laser" in path_str or "lidar" in path_str:
        components["laser/lidar"] = True
        print(f"  ✅ LiDAR found: {prim.GetPath()}")
    if "steering" in path_str and "joint" in path_str:
        components["steering"] = True
        print(f"  ✅ Steering joint found: {prim.GetPath()}")
    if "wheel" in path_str and "joint" in path_str:
        components["wheel"] = True
        print(f"  ✅ Wheel joint found: {prim.GetPath()}")

print("\n" + "="*80)
print("Summary:")
print("="*80)
for component, found in components.items():
    status = "✅" if found else "❌"
    print(f"  {status} {component}")

print("\n" + "="*80)
print("Inspection complete!")
print("="*80)
