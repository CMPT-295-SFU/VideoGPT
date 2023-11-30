
# Read file line by line and parse lines like the following
# of type 2023-10-07 09:29:38.497 | Topic: parse S type instruction | | 1
# or 2023-10-07 09:40:32.197 | Q/A: can you explain how for load word, I can parse it into a instruction| | 2
# create tuple for each including timestamp, type, question, answer, and id
# write to csv file

import csv
import re
import sys
import os
import argparse


def is_new_entry(line):
    # Regular expression to match the timestamp pattern at the beginning of a line
    return re.match(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}', line)


def parse_file(filename):
    data = []
    with open(filename, 'r') as file:
        current_entry = None
        for line in file:
            if is_new_entry(line):
                if current_entry:
                    data.append(current_entry)
                parts = line.strip().split('|')
                current_entry = {
                    'timestamp': parts[0].strip(),
                    'type': parts[1].strip(),
                    'content': parts[2].strip() if len(parts) > 2 else '',
                    'optional': '',
                    'value': ''
                }
                if len(parts) > 3:
                    current_entry['optional'] = parts[3].strip()
                if len(parts) > 4:
                    current_entry['value'] = parts[-1].strip()
            else:
                # Append the line to the current entry's content
                current_entry['content'] += ' ' + line.strip()

        # Append the last entry
        if current_entry:
            data.append(current_entry)

    return data

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Parse log file')
    parser.add_argument('-l','--log', help='log file to parse', required=True)
    args = parser.parse_args()
    
    data = parse_file(args.log)    
    for d in data:
        print(d)
    

if __name__ == '__main__':
    main()