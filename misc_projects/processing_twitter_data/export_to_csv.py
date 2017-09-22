import re
import sys
import os

if len(sys.argv) < 3:
    exit("Error: please provide the path to twitter data followed by the path you want the result to be saved. \n"
         "\t Example: export_to_csv.py twitter_data.tsv twitter_result.csv")

input_file_path = sys.argv[1]
output_file_path = sys.argv[2]

# Check if the input file exists
if not os.path.isfile(input_file_path):
    exit("Error: could not find the input file. Please check the path.")

# Check if the output file exists, if so ask for permission to overwrite
if os.path.isfile(output_file_path):
    e = input("{0} already exists, do you want to overwrite? [y/n]: ".format(output_file_path))

    attempt_cnt = 0
    while e.lower() != 'y' and e.lower() != 'n':
        e = input("please use y/n to continue or exit! [y/n]: ")
        attempt_cnt += 1
        if attempt_cnt >= 3:
            exit()

    if e.lower() == 'y':
        os.remove(output_file_path)
    elif e.lower() == 'n':
        exit()

# Creating the output file and checking if the directory exists
try:
    open(output_file_path, 'a').close()
except:
    exit("Error: output file's directory does not exist.")

print("Starting to process...")

# Start reading the input file
with open(input_file_path, 'r', encoding='utf-8', errors='ignore') as _infile, open(output_file_path, 'w') as _outfile:
    string_buffer = ''
    buffer_cnt = 0

    for line in _infile:
        # Check if anyone is mentioned in the tweet at all, if not continue
        if line.find('@') == -1:
            continue

        match = line.split('\t')

        # Matching twitter handles in the tweet
        mentions = re.findall('(?<=^|(?<=[^a-zA-Z0-9-_\\.]))@([A-Za-z]+[A-Za-z0-9_]+)', match[1])

        # check if it is a retweet
        is_retweet = int(match[1].startswith('RT '))

        for twitter_handle in mentions:
            # csv format -> tweeted_by, tweeted_to, timestamp, tweet_id, is_retweet
            string_buffer += '{0},{1},{2},{3},{4}\n'.format(match[3][:-1].lower(), twitter_handle.lower(), match[2],
                                                            match[0], is_retweet)
            buffer_cnt += 1

        if buffer_cnt > 200:
            _outfile.write(string_buffer)
            string_buffer = ''
            buffer_cnt = 0

    if buffer_cnt > 0:
        _outfile.write(string_buffer)

print("Processing was done successfully!")
