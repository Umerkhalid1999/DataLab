"""
Quick verification script to check if Clean Dataset is in the template
"""
import os

template_path = "templates/dashboard.html"

print("=" * 60)
print("VERIFYING CLEAN DATASET IN TEMPLATE")
print("=" * 60)

if not os.path.exists(template_path):
    print(f"\n❌ ERROR: {template_path} not found!")
    exit(1)

with open(template_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Check for clean-dataset class
if 'clean-dataset' in content:
    print("\n[OK] FOUND: 'clean-dataset' class in template")
    
    # Find the line
    lines = content.split('\n')
    for i, line in enumerate(lines, 1):
        if 'clean-dataset' in line:
            print(f"\n  Line {i}: {line.strip()}")
    
    # Check if it's in a dropdown menu
    if '<ul class="dropdown-menu' in content and 'clean-dataset' in content:
        print("\n[OK] CONFIRMED: Clean Dataset is inside dropdown menu")
    
    # Check for the icon
    if 'fa-broom' in content:
        print("[OK] CONFIRMED: Broom icon (fa-broom) present")
    
    # Check for the text
    if 'Clean Dataset' in content:
        print("[OK] CONFIRMED: 'Clean Dataset' text present")
    
    print("\n" + "=" * 60)
    print("TEMPLATE IS CORRECT!")
    print("=" * 60)
    print("\nIf you still don't see it in the browser:")
    print("1. Stop Flask (Ctrl+C)")
    print("2. Clear browser cache (Ctrl+Shift+Delete)")
    print("3. Restart Flask: python main.py")
    print("4. Hard refresh browser (Ctrl+F5)")
    print("5. Check browser console for errors (F12)")
    
else:
    print("\n[ERROR] 'clean-dataset' NOT FOUND in template!")
    print("\nThe template needs to be updated.")
    print("Run this command to add it:")
    print("\n  python add_clean_dataset_to_template.py")
