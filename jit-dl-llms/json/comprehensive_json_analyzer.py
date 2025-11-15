#!/usr/bin/env python3
"""
Comprehensive JSON Parser for buggy_changes_with_buggy_line.json

This script provides a complete toolkit for parsing and analyzing the buggy changes dataset.

File Structure:
{
    "repository_name": {
        "commit_id": {
            "added": {
                "file_path": ["line1", "line2", ...],
                ...
            },
            "deleted": {
                "file_path": ["line1", "line2", ...],
                ...
            },
            "modified": {
                "file_path": ["line1", "line2", ...],
                ...
            }
        },
        ...
    },
    ...
}
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Iterator, Optional, Tuple
from collections import defaultdict, Counter
import sys


class BuggyChangesAnalyzer:
    """Comprehensive analyzer for the buggy changes JSON dataset."""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.file_size_mb = self.file_path.stat().st_size / (1024 * 1024)
        self._data = None
        
    def load_data(self, force_load: bool = False) -> bool:
        """Load the JSON data. Only loads if file is small enough or force_load=True."""
        if self._data is not None:
            return True
            
        if self.file_size_mb > 100 and not force_load:
            print(f"File is {self.file_size_mb:.1f} MB. Set force_load=True to load anyway.")
            return False
            
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self._data = json.load(f)
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def get_basic_stats(self) -> Dict[str, Any]:
        """Get basic statistics about the dataset."""
        if not self.load_data():
            # If can't load full data, use streaming approach
            return self._get_stats_streaming()
        
        stats = {
            'file_size_mb': round(self.file_size_mb, 2),
            'total_repositories': len(self._data),
            'repository_names': list(self._data.keys())[:10],  # First 10 repos
            'total_commits': 0,
            'total_files_changed': 0,
            'total_lines_added': 0,
            'total_lines_deleted': 0,
            'total_lines_modified': 0
        }
        
        for repo_name, commits in self._data.items():
            stats['total_commits'] += len(commits)
            
            for commit_id, changes in commits.items():
                for change_type in ['added', 'deleted', 'modified']:
                    if change_type in changes:
                        files = changes[change_type]
                        stats[f'total_files_changed'] += len(files)
                        
                        for file_path, lines in files.items():
                            stats[f'total_lines_{change_type}'] += len(lines)
        
        return stats
    
    def _get_stats_streaming(self) -> Dict[str, Any]:
        """Get basic stats using streaming approach for large files."""
        stats = {
            'file_size_mb': round(self.file_size_mb, 2),
            'analysis_method': 'streaming (partial)',
            'sample_size': 1000
        }
        
        # Read first portion to estimate
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                sample = f.read(50000)  # 50KB sample
                
            # Count repositories in sample
            repo_count = sample.count('": {')
            commit_count = sample.count('"added"')  # Assuming each commit has added section
            
            stats.update({
                'estimated_repositories': f"~{repo_count * 10} (estimated)",
                'estimated_commits': f"~{commit_count * 10} (estimated)",
                'sample_analysis': f"Based on first {len(sample)} characters"
            })
            
        except Exception as e:
            stats['error'] = str(e)
            
        return stats
    
    def get_repository_stats(self, repo_name: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific repository."""
        if not self.load_data():
            return None
            
        if repo_name not in self._data:
            return {'error': f'Repository "{repo_name}" not found'}
        
        repo_data = self._data[repo_name]
        stats = {
            'repository': repo_name,
            'total_commits': len(repo_data),
            'commit_ids': list(repo_data.keys())[:5],  # First 5 commits
            'files_by_change_type': defaultdict(set),
            'lines_by_change_type': defaultdict(int),
            'file_extensions': Counter()
        }
        
        for commit_id, changes in repo_data.items():
            for change_type in ['added', 'deleted', 'modified']:
                if change_type in changes:
                    for file_path, lines in changes[change_type].items():
                        stats['files_by_change_type'][change_type].add(file_path)
                        stats['lines_by_change_type'][change_type] += len(lines)
                        
                        # Count file extensions
                        ext = Path(file_path).suffix.lower()
                        stats['file_extensions'][ext] += 1
        
        # Convert sets to counts
        stats['files_by_change_type'] = {
            k: len(v) for k, v in stats['files_by_change_type'].items()
        }
        
        return dict(stats)
    
    def search_by_repository(self, repo_name: str) -> Optional[Dict[str, Any]]:
        """Get all data for a specific repository."""
        if not self.load_data():
            return None
            
        return self._data.get(repo_name)
    
    def search_by_file_extension(self, extension: str) -> Dict[str, List[str]]:
        """Find all files with a specific extension across all repositories."""
        if not self.load_data():
            return {'error': 'Cannot load data for search'}
        
        results = defaultdict(list)
        extension = extension.lower()
        if not extension.startswith('.'):
            extension = '.' + extension
            
        for repo_name, commits in self._data.items():
            for commit_id, changes in commits.items():
                for change_type in ['added', 'deleted', 'modified']:
                    if change_type in changes:
                        for file_path in changes[change_type].keys():
                            if Path(file_path).suffix.lower() == extension:
                                results[repo_name].append(f"{commit_id}:{file_path}")
        
        return dict(results)
    
    def get_commit_details(self, repo_name: str, commit_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific commit."""
        if not self.load_data():
            return None
            
        if repo_name not in self._data or commit_id not in self._data[repo_name]:
            return {'error': f'Commit not found: {repo_name}/{commit_id}'}
        
        commit_data = self._data[repo_name][commit_id]
        details = {
            'repository': repo_name,
            'commit_id': commit_id,
            'summary': {},
            'files': {}
        }
        
        for change_type in ['added', 'deleted', 'modified']:
            if change_type in commit_data:
                files = commit_data[change_type]
                total_lines = sum(len(lines) for lines in files.values())
                details['summary'][change_type] = {
                    'file_count': len(files),
                    'line_count': total_lines
                }
                details['files'][change_type] = files
        
        return details
    
    def export_sample(self, output_file: str, max_repos: int = 3, max_commits_per_repo: int = 5) -> bool:
        """Export a small sample of the data to a new JSON file."""
        if not self.load_data():
            return False
        
        sample_data = {}
        
        for i, (repo_name, commits) in enumerate(self._data.items()):
            if i >= max_repos:
                break
                
            sample_data[repo_name] = {}
            
            for j, (commit_id, changes) in enumerate(commits.items()):
                if j >= max_commits_per_repo:
                    break
                    
                sample_data[repo_name][commit_id] = changes
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(sample_data, f, indent=2, ensure_ascii=False)
            
            sample_size_mb = os.path.getsize(output_file) / (1024 * 1024)
            print(f"Sample exported to {output_file} ({sample_size_mb:.2f} MB)")
            return True
            
        except Exception as e:
            print(f"Error exporting sample: {e}")
            return False
    
    def analyze_code_patterns(self, language_extensions: List[str] = None) -> Dict[str, Any]:
        """Analyze common code patterns in the dataset."""
        if language_extensions is None:
            language_extensions = ['.java', '.py', '.js', '.cpp', '.c']
        
        if not self.load_data():
            return {'error': 'Cannot load data for analysis'}
        
        patterns = {
            'common_keywords': Counter(),
            'import_statements': Counter(),
            'method_declarations': Counter(),
            'language_distribution': Counter()
        }
        
        for repo_name, commits in self._data.items():
            for commit_id, changes in commits.items():
                for change_type in ['added', 'deleted', 'modified']:
                    if change_type in changes:
                        for file_path, lines in changes[change_type].items():
                            ext = Path(file_path).suffix.lower()
                            if ext in language_extensions:
                                patterns['language_distribution'][ext] += len(lines)
                                
                                for line in lines:
                                    line = line.strip()
                                    if not line:
                                        continue
                                        
                                    # Count import statements
                                    if any(line.startswith(kw) for kw in ['import ', 'from ', '#include']):
                                        patterns['import_statements'][line[:50]] += 1
                                    
                                    # Count method/function declarations
                                    if any(kw in line for kw in ['def ', 'function ', 'public ', 'private ', 'protected ']):
                                        patterns['method_declarations'][line[:50]] += 1
                                    
                                    # Count common keywords
                                    for keyword in ['if', 'else', 'for', 'while', 'try', 'catch', 'return']:
                                        if f' {keyword} ' in line or line.startswith(keyword + ' '):
                                            patterns['common_keywords'][keyword] += 1
        
        # Get top items for each category
        result = {}
        for category, counter in patterns.items():
            result[category] = dict(counter.most_common(10))
        
        return result


def main():
    """Demonstrate the analyzer usage."""
    json_file_path = r"JITFine\labels for each line\buggy_changes_with_buggy_line.json"
    
    if not os.path.exists(json_file_path):
        print(f"Error: File not found: {json_file_path}")
        return
    
    analyzer = BuggyChangesAnalyzer(json_file_path)
    
    print("=== BUGGY CHANGES DATASET ANALYZER ===")
    print(f"File: {json_file_path}")
    
    # Basic statistics
    print("\n=== BASIC STATISTICS ===")
    stats = analyzer.get_basic_stats()
    for key, value in stats.items():
        if isinstance(value, list):
            if len(value) > 5:
                print(f"{key}: {value[:5]}... (showing first 5)")
            else:
                print(f"{key}: {value}")
        else:
            print(f"{key}: {value}")
    
    # Repository-specific analysis (if data is loaded)
    if analyzer._data:
        repo_names = list(analyzer._data.keys())[:3]  # First 3 repos
        
        for repo_name in repo_names:
            print(f"\n=== REPOSITORY: {repo_name} ===")
            repo_stats = analyzer.get_repository_stats(repo_name)
            if repo_stats and 'error' not in repo_stats:
                for key, value in repo_stats.items():
                    if key not in ['commit_ids']:  # Skip detailed lists
                        print(f"{key}: {value}")
        
        # Export sample
        print(f"\n=== EXPORTING SAMPLE ===")
        analyzer.export_sample("sample_buggy_changes.json", max_repos=2, max_commits_per_repo=3)
    
    print(f"\n=== ANALYSIS COMPLETE ===")


if __name__ == "__main__":
    main()