import argparse

# Recive input parameters
parser = argparse.ArgumentParser()
parser.add_argument('arg1')
parser.add_argument('arg2')

args = parser.parse_args()
arg1 = args.arg1
arg2 = args.arg2

print arg1, arg2
