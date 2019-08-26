import os

files = os.listdir('input')
for file in files:
	os.system("python3 main.py input/"+file+" C0 C1 T0 T1 T2 0")
