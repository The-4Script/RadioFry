import os
import sys
import re
import argparse

# Fix for Windows console UnicodeEncodeError
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

def get_project_root():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, "..", ".."))

def parse_bank(bank_path):
    with open(bank_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    parts = re.split(r'(^## Entry\s+\d+.*?$)', content, flags=re.MULTILINE)
    
    entries = []
    for i in range(1, len(parts), 2):
        header = parts[i]
        body = parts[i+1]
        
        match = re.search(r'^## Entry\s+(\d+)', header)
        entry_id = match.group(1) if match else "???"
        
        entries.append({
            "id": entry_id,
            "header": header.strip(),
            "body": body.strip(),
            "full_text": header + body
        })
    return entries

def search_index_and_current(query, root):
    results = []
    for filename in ["research_memory/INDEX.md", "research_memory/CURRENT.md"]:
        filepath = os.path.join(root, filename)
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    if query.lower() in line.lower():
                        results.append(f"[{filename}] {line.strip()}")
    return results

def main():
    parser = argparse.ArgumentParser(description="Search RadioFry historical memory.")
    parser.add_argument("query", nargs="?", help="Topic or keyword to search for")
    parser.add_argument("--entry", help="Retrieve a specific entry by ID (e.g., '016' or '16')")
    parser.add_argument("--recent", type=int, help="Limit to N most recent matching entries")
    
    args = parser.parse_args()
    
    if not args.query and not args.entry:
        parser.print_help()
        sys.exit(1)
        
    root = get_project_root()
    bank_path = os.path.join(root, "BANK.md")
    
    if not os.path.exists(bank_path):
        print(f"Error: Could not find BANK.md at {bank_path}")
        sys.exit(1)
        
    entries = parse_bank(bank_path)
    
    if args.entry:
        target_id = str(int(args.entry)).zfill(3)
        for e in entries:
            if e["id"] == target_id:
                print(f"--- MATCHED ENTRY {e['id']} ---")
                print(e["full_text"].strip())
                print("-" * 40)
                sys.exit(0)
        print(f"Entry {args.entry} not found.")
        sys.exit(1)
        
    if args.query:
        print(f"Searching memory for: '{args.query}'\n")
        
        meta_matches = search_index_and_current(args.query, root)
        if meta_matches:
            print("--- Matches in Memory Index / Current State ---")
            for m in meta_matches:
                print(m)
            print()
            
        matches = []
        for e in entries:
            if args.query.lower() in e["full_text"].lower():
                matches.append(e)
                
        matches.sort(key=lambda x: x["id"], reverse=True)
        
        if args.recent:
            matches = matches[:args.recent]
            
        if not matches:
            print("No matching entries found in BANK.md.")
            sys.exit(0)
            
        print(f"--- Found {len(matches)} matching entries in BANK.md ---")
        for e in matches:
            print(f"\n>>> {e['header']} <<<")
            
            paragraphs = e["body"].split("\n\n")
            matched_paragraphs = [p for p in paragraphs if args.query.lower() in p.lower()]
            
            if len(matched_paragraphs) == len(paragraphs) or len(e["body"].splitlines()) < 30:
                print(e["body"])
            else:
                print("... [Context extracted] ...")
                for p in matched_paragraphs:
                    print(p.strip())
                    print("...")
            print("-" * 60)

if __name__ == "__main__":
    main()
