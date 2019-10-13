import os

files = os.listdir('d:/input_I')
for file in files:
	os.system("python main.py d:/input_I/"+file+" 0.25 0 6 6  45 0 5.0")

files = os.listdir('d:/input_II')
for file in files:
	os.system("python main.py d:/input_II/"+file+" 0.5 0 6 6 40 0 5.0")

files = os.listdir('d:/input_360')
for file in files:
	os.system("python main.py d:/input_360/"+file+" 0.25 0.25 15 15 40 0 10.0")

files = os.listdir('d:/input_3D')
for file in files:
	os.system("python main.py d:/input_3D/"+file+"  0.5 0 6 6 85 0 10.0")

files = os.listdir('d:/input_CYGX')
for file in files:
	os.system("python main.py d:/input_CYGX/"+file+" 0.25 0.25 20 20 40 0 10.0")

files = os.listdir('d:/input_Deep')
for file in files:
	os.system("python main.py d:/input_Deep/"+file+" 0.5 0 20 20 55 0 10.0")

files = os.listdir('d:/input_SMOG')
for file in files:
	os.system("python main.py d:/input_SMOG/"+file+" 0.25 0.25 20 20 40 0 10.0")

files = os.listdir('d:/input_VELA')
for file in files:
	os.system("python main.py d:/input_VELA/"+file+" 0.25 0.25 25 25 55 0 10.0 ")
