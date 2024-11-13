import re
import json

def parse_toc(filename):
    toc = []
    level_stack = []

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if not line:
                continue  # Skip empty lines

            # Match Level 1 headings (e.g., CHAPTER 4 - PLAN RISK MANAGEMENT .......... 19)
            match = re.match(r'^(CHAPTER|APPENDIX)\s+([A-Z0-9\-]+)\s*-\s*(.+?)\s*\.{3,}\s*(\d+)\s*$', line, re.IGNORECASE)
            if match:
                # Extract components
                chapter_type = match.group(1).capitalize()
                number = match.group(2).strip()
                title = match.group(3).strip()
                page = int(match.group(4).strip()) + 12  # Adjust page number

                # Create a new Level 1 item
                item = {
                    'type': chapter_type,
                    'number': number,
                    'title': title,
                    'page': page,
                    'children': []
                }
                toc.append(item)
                # Reset the level stack
                level_stack = [ (1, item) ]
                continue

            # Match numbered headings (e.g., 1.1 Title ........... 2)
            match = re.match(r'^(\d+(?:\.\d+)*)\s+(.+?)\s*\.{3,}\s*(\d+)\s*$', line)
            if match:
                number = match.group(1).strip()
                title = match.group(2).strip()
                page = int(match.group(3).strip()) + 12  # Adjust page number
                levels = number.split('.')
                level = len(levels)

                # Create a new item
                item = {
                    'number': number,
                    'title': title,
                    'page': page,
                    'children': []
                }

                # Adjust the level stack to the current level
                while level_stack and level_stack[-1][0] >= level:
                    level_stack.pop()

                if level_stack:
                    parent = level_stack[-1][1]
                    parent['children'].append(item)
                else:
                    toc.append(item)

                level_stack.append( (level, item) )
                continue

            # Optionally, handle other headings or text if needed

    return toc

def main():
    toc = parse_toc('toc_text.txt')
    with open('TO1.json', 'w', encoding='utf-8') as f:
        json.dump(toc, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    main()
