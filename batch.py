import os

files = os.listdir('input')
for file in files:
	os.system("python3 main.py input/"+file+" 0.75 0.0833")
