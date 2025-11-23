"""
Test script for Workflow Management System
Tests all workflow features to ensure they work correctly
"""

import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:5000"
TEST_DATASET_ID = 1  # Change this to match your dataset ID

def test_workflow_system():
    """Test all workflow endpoints"""
    
    print("=" * 60)
    print("WORKFLOW MANAGEMENT SYSTEM TEST")
    print("=" * 60)
    
    # Create a session to maintain cookies
    session = requests.Session()
    
    # Test 1: Access workflow page
    print("\n[TEST 1] Accessing workflow page...")
    try:
        response = session.get(f"{BASE_URL}/workflow/{TEST_DATASET_ID}")
        if response.status_code == 200:
            print("✅ Workflow page accessible")
        else:
            print(f"❌ Failed to access workflow page: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Error accessing workflow page: {e}")
        return
    
    # Test 2: Get dataset info
    print("\n[TEST 2] Getting dataset column information...")
    try:
        response = session.get(f"{BASE_URL}/workflow/api/dataset/{TEST_DATASET_ID}/info")
        data = response.json()
        if data.get('success'):
            print(f"✅ Dataset info retrieved: {len(data['columns'])} columns")
            print(f"   Columns: {[col['name'] for col in data['columns'][:5]]}")
        else:
            print(f"❌ Failed to get dataset info: {data.get('message')}")
    except Exception as e:
        print(f"❌ Error getting dataset info: {e}")
    
    # Test 3: Create a pipeline
    print("\n[TEST 3] Creating a test pipeline...")
    pipeline_data = {
        "name": "Test Preprocessing Pipeline",
        "description": "Automated test pipeline",
        "steps": [
            {
                "type": "missing_value_handling",
                "name": "Handle Missing Values",
                "description": "Fill missing values with mean",
                "parameters": {
                    "strategy": "mean",
                    "columns": []
                }
            },
            {
                "type": "scaling",
                "name": "Scale Features",
                "description": "Standardize numerical features",
                "parameters": {
                    "method": "standard",
                    "columns": []
                }
            }
        ]
    }
    
    try:
        response = session.post(
            f"{BASE_URL}/workflow/api/{TEST_DATASET_ID}/create_pipeline",
            json=pipeline_data,
            headers={'Content-Type': 'application/json'}
        )
        data = response.json()
        if data.get('success'):
            pipeline_id = data['pipeline']['id']
            print(f"✅ Pipeline created successfully (ID: {pipeline_id})")
            print(f"   Name: {data['pipeline']['name']}")
            print(f"   Steps: {len(data['pipeline']['steps'])}")
        else:
            print(f"❌ Failed to create pipeline: {data.get('message')}")
            return
    except Exception as e:
        print(f"❌ Error creating pipeline: {e}")
        return
    
    # Test 4: Get all pipelines
    print("\n[TEST 4] Retrieving all pipelines...")
    try:
        response = session.get(f"{BASE_URL}/workflow/api/{TEST_DATASET_ID}/pipelines")
        data = response.json()
        if data.get('success'):
            print(f"✅ Retrieved {len(data['pipelines'])} pipeline(s)")
            for p in data['pipelines']:
                print(f"   - {p['name']} (v{p['version']}, {len(p['steps'])} steps)")
        else:
            print(f"❌ Failed to get pipelines: {data.get('message')}")
    except Exception as e:
        print(f"❌ Error getting pipelines: {e}")
    
    # Test 5: Execute pipeline
    print("\n[TEST 5] Executing pipeline...")
    try:
        response = session.post(f"{BASE_URL}/workflow/api/pipeline/{pipeline_id}/execute")
        data = response.json()
        if data.get('success'):
            print("✅ Pipeline executed successfully")
            print(f"   Input shape: {data['execution_record']['input_shape']}")
            print(f"   Output shape: {data['execution_record']['output_shape']}")
            print(f"   Steps executed: {len(data['execution_record']['execution_log'])}")
        else:
            print(f"❌ Pipeline execution failed: {data.get('message')}")
    except Exception as e:
        print(f"❌ Error executing pipeline: {e}")
    
    # Test 6: Export as Jupyter notebook
    print("\n[TEST 6] Exporting pipeline as Jupyter notebook...")
    try:
        response = session.post(f"{BASE_URL}/workflow/api/pipeline/{pipeline_id}/export_notebook")
        if response.status_code == 200:
            print("✅ Jupyter notebook exported successfully")
            print(f"   File size: {len(response.content)} bytes")
        else:
            print(f"❌ Failed to export notebook: {response.status_code}")
    except Exception as e:
        print(f"❌ Error exporting notebook: {e}")
    
    # Test 7: Export documentation
    print("\n[TEST 7] Exporting pipeline documentation...")
    try:
        response = session.post(f"{BASE_URL}/workflow/api/pipeline/{pipeline_id}/export_documentation")
        if response.status_code == 200:
            print("✅ Documentation exported successfully")
            print(f"   File size: {len(response.content)} bytes")
        else:
            print(f"❌ Failed to export documentation: {response.status_code}")
    except Exception as e:
        print(f"❌ Error exporting documentation: {e}")
    
    # Test 8: Create version
    print("\n[TEST 8] Creating pipeline version...")
    try:
        response = session.post(
            f"{BASE_URL}/workflow/api/pipeline/{pipeline_id}/version",
            json={"notes": "Test version creation"},
            headers={'Content-Type': 'application/json'}
        )
        data = response.json()
        if data.get('success'):
            print(f"✅ Version created: {data['new_version']}")
        else:
            print(f"❌ Failed to create version: {data.get('message')}")
    except Exception as e:
        print(f"❌ Error creating version: {e}")
    
    # Test 9: Share pipeline
    print("\n[TEST 9] Generating share link...")
    try:
        response = session.post(
            f"{BASE_URL}/workflow/api/pipeline/{pipeline_id}/share",
            json={"type": "link"},
            headers={'Content-Type': 'application/json'}
        )
        data = response.json()
        if data.get('success'):
            print(f"✅ Share link generated: {data['share_link']}")
        else:
            print(f"❌ Failed to generate share link: {data.get('message')}")
    except Exception as e:
        print(f"❌ Error generating share link: {e}")
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print("All workflow features have been tested.")
    print("Check the results above for any failures.")
    print("=" * 60)

if __name__ == "__main__":
    print("\n⚠️  IMPORTANT: Make sure the Flask app is running before testing!")
    print("⚠️  Update TEST_DATASET_ID if needed (currently set to 1)")
    input("\nPress Enter to start testing...")
    
    test_workflow_system()
