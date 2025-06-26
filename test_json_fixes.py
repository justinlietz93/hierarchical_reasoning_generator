#!/usr/bin/env python3
"""
Test script to verify the JSON parsing fixes for the Gemini client.
"""

import sys
import os
import json

# Add the hierarchical_planner directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'hierarchical_planner'))

from gemini_client import _sanitize_json_response, _normalize_steps_format

def test_json_sanitization():
    """Test the JSON sanitization function with problematic JSON."""
    
    # Test case similar to the error in the log
    problematic_json = '''
{
  "steps": [
    {
      "step 1": "Create a new directory named `ironclad-system` for the project and navigate into it. This will serve as the root of the project repository."
    },
    {
      "step 2": "Initialize a new Git repository in the `ironclad-system` directory to enable version control from the very beginning. Hint: Use the `git init` command."
    }
    {
      "step 3": "Initialize a Node.js project by creating a `package.json` file. Use `npm init -y` for a quick setup."
    }
  ]
}
'''
    
    print("Testing JSON sanitization...")
    print("Original JSON has missing comma before step 3")
    
    try:
        # This should fail
        json.loads(problematic_json)
        print("❌ Original JSON unexpectedly parsed successfully")
    except json.JSONDecodeError as e:
        print(f"✅ Original JSON failed as expected: {e}")
    
    # Test sanitization
    try:
        sanitized = _sanitize_json_response(problematic_json)
        parsed = json.loads(sanitized)
        print("✅ Sanitized JSON parsed successfully")
        print(f"   Found {len(parsed.get('steps', []))} steps")
    except json.JSONDecodeError as e:
        print(f"❌ Sanitized JSON still failed: {e}")
        return False
    
    return True

def test_step_normalization():
    """Test the step normalization function."""
    
    # Test old format
    old_format_steps = [
        {"step 1": "First step description"},
        {"step 2": "Second step description", "qa_info": {"some": "data"}},
        {"step 3": "Third step description"}
    ]
    
    print("\nTesting step normalization...")
    print("Converting from old format to new format")
    
    normalized = _normalize_steps_format(old_format_steps)
    
    # Check structure
    expected_keys = {"id", "description"}
    for i, step in enumerate(normalized):
        if not isinstance(step, dict):
            print(f"❌ Step {i+1} is not a dictionary")
            return False
        
        if "description" not in step:
            print(f"❌ Step {i+1} missing 'description' key")
            return False
            
        if "id" not in step:
            print(f"❌ Step {i+1} missing 'id' key")
            return False
        
        print(f"✅ Step {i+1}: id='{step['id']}', description='{step['description'][:50]}...'")
    
    # Test that new format passes through unchanged
    new_format_steps = [
        {"id": "step_1", "description": "First step"},
        {"id": "step_2", "description": "Second step"}
    ]
    
    normalized_new = _normalize_steps_format(new_format_steps)
    if normalized_new == new_format_steps:
        print("✅ New format steps pass through unchanged")
    else:
        print("❌ New format steps were incorrectly modified")
        return False
    
    return True

def main():
    """Run all tests."""
    print("Testing Gemini client JSON parsing fixes...")
    print("=" * 50)
    
    success = True
    
    success &= test_json_sanitization()
    success &= test_step_normalization()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All tests passed! The fixes should work.")
    else:
        print("❌ Some tests failed. Check the implementation.")
    
    return success

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
