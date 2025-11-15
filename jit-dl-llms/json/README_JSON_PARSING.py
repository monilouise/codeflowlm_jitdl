#!/usr/bin/env python3
"""
Usage Guide for JSON Parsing Tools

This script demonstrates how to use all the JSON parsing tools created for the 
buggy_changes_with_buggy_line.json dataset.
"""

def print_summary():
    """Print a summary of the JSON dataset and available tools."""
    
    print("=" * 70)
    print("JSON DATASET ANALYSIS SUMMARY")
    print("=" * 70)
    
    print("\n📊 DATASET OVERVIEW:")
    print("   • File: buggy_changes_with_buggy_line.json")
    print("   • Size: 53.63 MB")
    print("   • Repositories: 26")
    print("   • Total Commits: 2,456")
    print("   • Files Changed: 39,698")
    print("   • Lines Added: 394,361")
    print("   • Lines Deleted: 166,769")
    
    print("\n🏗️  DATASET STRUCTURE:")
    print("   Repository Name")
    print("   ├── Commit ID (SHA)")
    print("   │   ├── added")
    print("   │   │   ├── file_path: [array of added lines]")
    print("   │   │   └── ...")
    print("   │   └── deleted")
    print("   │       ├── file_path: [array of deleted lines]")
    print("   │       └── ...")
    print("   └── ...")
    
    print("\n📁 SAMPLE REPOSITORIES:")
    repositories = ['ant-ivy', 'commons-math', 'opennlp', 'parquet-mr', 'archiva']
    for i, repo in enumerate(repositories, 1):
        print(f"   {i}. {repo}")
    
    print("\n💻 PROGRAMMING LANGUAGES:")
    print("   • Primarily Java (.java files)")
    print("   • Some Python, JavaScript, C++")
    
    print("\n🔧 AVAILABLE TOOLS:")
    print("   1. comprehensive_json_analyzer.py - Full analysis with statistics")
    print("   2. parse_buggy_changes.py - Streaming parser for large files")
    print("   3. simple_json_parser.py - Basic parser without dependencies")
    print("   4. json_parser_demo.ipynb - Interactive Jupyter notebook")
    print("   5. sample_buggy_changes.json - Small sample for testing")


def show_usage_examples():
    """Show practical usage examples."""
    
    print("\n" + "=" * 70)
    print("USAGE EXAMPLES")
    print("=" * 70)
    
    print("\n🚀 QUICK START:")
    print("   # Run comprehensive analysis")
    print("   python comprehensive_json_analyzer.py")
    
    print("\n📈 BASIC ANALYSIS:")
    print("""
   from comprehensive_json_analyzer import BuggyChangesAnalyzer
   
   analyzer = BuggyChangesAnalyzer('JITFine/labels for each line/buggy_changes_with_buggy_line.json')
   stats = analyzer.get_basic_stats()
   print(stats)
   """)
    
    print("\n🔍 REPOSITORY ANALYSIS:")
    print("""
   # Analyze a specific repository
   repo_stats = analyzer.get_repository_stats('ant-ivy')
   print(f"Total commits: {repo_stats['total_commits']}")
   print(f"Files changed: {repo_stats['files_by_change_type']}")
   """)
    
    print("\n📝 COMMIT DETAILS:")
    print("""
   # Get details of a specific commit
   commit = analyzer.get_commit_details('ant-ivy', 'commit_id_here')
   print(commit['summary'])
   """)
    
    print("\n🔎 SEARCH BY FILE TYPE:")
    print("""
   # Find all Java files
   java_files = analyzer.search_by_file_extension('.java')
   for repo, files in java_files.items():
       print(f"{repo}: {len(files)} Java files")
   """)
    
    print("\n📤 EXPORT SAMPLE:")
    print("""
   # Export a smaller sample for testing
   analyzer.export_sample('my_sample.json', max_repos=5, max_commits_per_repo=10)
   """)
    
    print("\n🧪 STREAMING FOR LARGE FILES:")
    print("""
   from parse_buggy_changes import BuggyChangesParser
   
   parser = BuggyChangesParser('large_file.json')
   
   # Process items one by one without loading entire file
   for item in parser.stream_parse_items():
       # Process each item
       print(item)
       break  # Process only first item for demo
   """)


def show_data_examples():
    """Show examples of what the data looks like."""
    
    print("\n" + "=" * 70)
    print("DATA EXAMPLES")
    print("=" * 70)
    
    print("\n📋 TYPICAL COMMIT STRUCTURE:")
    print("""
   {
       "ant-ivy": {
           "19b8dc8ed20a3f25d510fdfb983506b8001f072e": {
               "added": {
                   "src/java/org/apache/ivy/ant/IvyInfo.java": [
                       "getProject().setProperty(\\"ivy.revision\\", md.getModuleRevisionId().getRevision());"
                   ],
                   "src/java/org/apache/ivy/core/module/id/ModuleRevisionId.java": [
                       "return other.getRevision().equals(getRevision())",
                       "import org.apache.ivy.Ivy;",
                       "_revision = revision == null ? Ivy.getWorkingRevision() : revision;"
                   ]
               },
               "deleted": {
                   "some/file/path.java": [
                       "old line 1",
                       "old line 2"
                   ]
               }
           }
       }
   }
   """)
    
    print("\n🎯 USE CASES:")
    print("   • Bug prediction research")
    print("   • Code change analysis") 
    print("   • Software engineering metrics")
    print("   • Machine learning on code changes")
    print("   • Repository mining studies")
    
    print("\n⚠️  CONSIDERATIONS:")
    print("   • Large file size (53+ MB)")
    print("   • Use streaming for memory efficiency")
    print("   • Contains real commit data from open source projects")
    print("   • Focused on buggy changes (commits that introduced bugs)")


def main():
    """Main function."""
    print_summary()
    show_usage_examples()
    show_data_examples()
    
    print("\n" + "=" * 70)
    print("🎉 READY TO PARSE!")
    print("=" * 70)
    print("\n✅ All tools are ready to use")
    print("✅ Sample data exported for testing")
    print("✅ Multiple parsing approaches available")
    print("\n📚 Choose the right tool for your needs:")
    print("   • Small analysis: Use comprehensive_json_analyzer.py")
    print("   • Large-scale processing: Use streaming parser")
    print("   • Interactive exploration: Use Jupyter notebook")


if __name__ == "__main__":
    main()