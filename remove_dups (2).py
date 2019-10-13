import operator 
import sys
import math

arglist = sys.argv
infile = arglist[1]
outfile = arglist[2]
outregfile = arglist[3]
minsize = float(arglist[4])

r = open(outfile, "w+")
rreg = open(outregfile, "w+")

num_lines = sum(1 for line in open(infile))
print("starting number: ",num_lines)
l = open(infile, encoding="ascii", errors='ignore')

regs = []
lines = {}

for i in range(num_lines):
    ogline = l.readline()
    line = [i for i in (ogline.split())]
    key = []
    for j in range(len(line[0:8])):
        if(j==0):
            key.append(line[0:8][j])
        else:
            key.append(float(line[0:8][j]))
    regs.append(key)
    regs[i][3] = regs[i][3]/3600.0
    regs[i][4] = regs[i][4]/3600.0
    lines[tuple(key)] = ogline

regs.sort(key=operator.itemgetter(1), reverse = True)
start = len(regs)

i = 0
while(i < len(regs)):
    u = regs[i]
    if(u[7]< minsize):
        print(u[0],u[1],u[2],u[3],u[4],u[5],u[6],u[7],"removed: minsize")
        regs.remove(u)
    else:
        i += 1

print(start-len(regs)," removed from minsize")    
start = len(regs)
  
oregs = regs
  
i = 0    
while(i < len(regs)):
    temp = []
    c = 1
    left = False
    right = False
    while(True):
        if(left == True and right == True):
            break
        cxl = regs[i][1]+regs[i][3]*.5
        cxr = regs[i][1]-regs[i][3]*.5
        if(i-c <= 0):
            left = True
        if(i-c > 0):
            if(cxl < regs[i-c][1]):
                left = True
            if(cxl >= regs[i-c][1] and left == False):
                temp.append(regs[i-c])
        if(i+c >= len(regs)):
            right = True
        if(i+c < len(regs)):
            if(cxr > regs[i+c][1]):
                right = True
            if(cxr <= regs[i+c][1] and right == False):
                temp.append(regs[i+c])
        c += 1
    u = regs[i]
    for j in temp:
        jxl = j[1]+.5*j[3]
        jxr = j[1]-.5*j[3]
        jyt = j[2]+.5*j[4]
        jyb = j[2]-.5*j[4]
        ixl = u[1]+.5*u[3]+0.0028
        ixr = u[1]-.5*u[3]-0.0028
        iyt = u[2]+.5*u[4]+0.0028
        iyb = u[2]-.5*u[4]-0.0028
        if(jxl <=ixl and jxr >= ixr and jyb >= iyb and jyt <= iyt):
            print(j,"removed: inside box")
            oregs.remove(j)
    i += 1

regs = oregs

regs.sort(key=operator.itemgetter(0), reverse = False)
print(start-len(regs), "boxes removed from within boxes, ",len(regs)," remaining")

start = len(regs)
startloop = start
stoploop = 0

while(stoploop<startloop):
    i = 0
    startloop = len(regs)
    while(i < len(regs)-1):
        temp = []
        temp.append(regs[i])
        temp.append(regs[i+1])
        distance = math.sqrt( (temp[0][1]-temp[1][1])**2 + (temp[0][2]-temp[1][2])**2 )
        if (distance < 0.0028 or (round(temp[0][1],2)==round(temp[1][1],2) and round(temp[0][2],3)==round(temp[1][2],3))):
            if (temp[0][6]<temp[1][6]):
                regs.remove(temp[0])
            else:
                regs.remove(temp[1])
            print("removed: ",temp[0]," ",temp[1])
        i += 1

    stoploop=len(regs)

print(start-len(regs), "close centers removed, ",len(regs)," remaining")

for i in regs:
#   print(i)
    r.write(lines[tuple(i)])
    rreg.write('Galactic; box '+str(format(i[1],'.5f'))+' '+str(format(i[2],'.5f'))+' '+str(format(i[3],'.5f'))+' '+str(format(i[4],'.5f'))+'\n')
    