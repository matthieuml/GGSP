import re


def extract_numbers_from_text(text):
    # Use regular expression to find integers and floats
    numbers = re.findall(r"\d+\.\d+|\d+", text)
    loop_feature = 1 if 'loop' in text else 0
    # Convert the extracted numbers to float
    return [float(num) for num in numbers] + [loop_feature]


def extract_features_from_file(file):
    stats = []
    fread = open(file, "r")
    line = fread.read()
    line = line.strip()
    stats = extract_numbers_from_text(line)
    fread.close()
    return stats
