#!/usr/bin/env python3
"""
JSON Parser for buggy_changes_with_buggy_line.json

This script provides utilities to parse and analyze the large JSON file
containing buggy changes with line-level information.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Iterator
import ijson  # For streaming JSON parsing of large files


class BuggyChangesParser:
    """Parser for the buggy changes JSON file."""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")
    
    def parse_full_file(self) -> Dict[str, Any]:
        """
        Parse the entire JSON file into memory.
        Warning: This may consume a lot of memory for large files.
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return {}
        except MemoryError:
            print("File too large to load into memory. Use streaming methods instead.")
            return {}
    
    def stream_parse_items(self) -> Iterator[Dict[str, Any]]:
        """
        Stream parse the JSON file, yielding items one by one.
        This is memory-efficient for large files.
        """
        try:
            with open(self.file_path, 'rb') as f:
                # Assuming the JSON structure is an object with multiple keys
                parser = ijson.parse(f)
                current_item = {}
                current_key = None
                
                for prefix, event, value in parser:
                    if event == 'start_map' and prefix == '':
                        current_item = {}
                    elif event == 'map_key':
                        current_key = value
                    elif event == 'string' or event == 'number' or event == 'boolean':
                        if current_key:
                            current_item[current_key] = value
                    elif event == 'end_map' and prefix == '':
                        if current_item:
                            yield current_item
                            current_item = {}
        except Exception as e:
            print(f"Error during streaming parse: {e}")
    
    def get_file_info(self) -> Dict[str, Any]:
        """Get basic information about the JSON file."""
        info = {
            'file_path': str(self.file_path),
            'file_size_mb': self.file_path.stat().st_size / (1024 * 1024),
            'exists': self.file_path.exists()
        }
        return info
    
    def sample_data(self, num_items: int = 5) -> List[Dict[str, Any]]:
        """
        Get a sample of items from the JSON file.
        """
        sample = []
        try:
            for i, item in enumerate(self.stream_parse_items()):
                if i >= num_items:
                    break
                sample.append(item)
        except Exception as e:
            print(f"Error sampling data: {e}")
        
        return sample
    
    def analyze_structure(self) -> Dict[str, Any]:
        """
        Analyze the structure of the JSON file by examining a few items.
        """
        sample = self.sample_data(3)
        if not sample:
            return {"error": "Could not sample data from file"}
        
        analysis = {
            'sample_count': len(sample),
            'keys': set(),
            'data_types': {},
            'sample_items': sample[:2]  # Show first 2 items
        }
        
        for item in sample:
            if isinstance(item, dict):
                analysis['keys'].update(item.keys())
                for key, value in item.items():
                    if key not in analysis['data_types']:
                        analysis['data_types'][key] = set()
                    analysis['data_types'][key].add(type(value).__name__)
        
        # Convert sets to lists for JSON serialization
        analysis['keys'] = list(analysis['keys'])
        analysis['data_types'] = {k: list(v) for k, v in analysis['data_types'].items()}
        
        return analysis
    
    def search_items(self, search_key: str, search_value: Any = None) -> List[Dict[str, Any]]:
        """
        Search for items containing a specific key or key-value pair.
        """
        results = []
        try:
            for item in self.stream_parse_items():
                if isinstance(item, dict):
                    if search_value is None:
                        # Just check if key exists
                        if search_key in item:
                            results.append(item)
                    else:
                        # Check for specific key-value pair
                        if item.get(search_key) == search_value:
                            results.append(item)
                        
                # Limit results to prevent memory issues
                if len(results) >= 100:
                    break
                    
        except Exception as e:
            print(f"Error during search: {e}")
        
        return results
    
    def count_items(self) -> int:
        """Count the total number of items in the JSON file."""
        count = 0
        try:
            for _ in self.stream_parse_items():
                count += 1
        except Exception as e:
            print(f"Error counting items: {e}")
        
        return count


def main():
    """Main function to demonstrate the parser usage."""
    json_file_path = r"c:\Users\Monique\Documents\UFPE\Doutorado\JIT-Fine\JIT-Fine\JITFine\labels for each line\buggy_changes_with_buggy_line.json"
    
    try:
        # Initialize parser
        parser = BuggyChangesParser(json_file_path)
        
        # Get file information
        print("=== FILE INFORMATION ===")
        file_info = parser.get_file_info()
        for key, value in file_info.items():
            print(f"{key}: {value}")
        
        print(f"\nFile size: {file_info['file_size_mb']:.2f} MB")
        
        # Analyze structure
        print("\n=== STRUCTURE ANALYSIS ===")
        structure = parser.analyze_structure()
        
        if 'error' in structure:
            print(f"Error: {structure['error']}")
            return
        
        print(f"Sample count: {structure['sample_count']}")
        print(f"Keys found: {structure['keys']}")
        print(f"Data types: {structure['data_types']}")
        
        # Show sample data
        print("\n=== SAMPLE DATA ===")
        for i, item in enumerate(structure['sample_items']):
            print(f"Item {i + 1}:")
            for key, value in item.items():
                # Truncate long values for display
                if isinstance(value, str) and len(value) > 100:
                    display_value = value[:100] + "..."
                else:
                    display_value = value
                print(f"  {key}: {display_value}")
            print()
        
        # Count total items (this might take a while for large files)
        print("=== COUNTING ITEMS ===")
        print("Counting total items... (this may take a moment)")
        total_count = parser.count_items()
        print(f"Total items in file: {total_count}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    # Install required dependency if not available
    try:
        import ijson
    except ImportError:
        print("Installing required dependency: ijson")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ijson"])
        import ijson
    
    main()