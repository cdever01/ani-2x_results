import os
import numpy as np

def readncdatall(file,N = 0):
    rdat = dict()
    if os.path.isfile(file):
        fd = open(file, 'r').read()
        fd = fd.split("\n",3)
        lot = fd[0]
        nda = int(fd[1])
        nat = int(fd[2].split(",")[0])
        spc = fd[2].split(",")[1:-1]

        elm = nat*3*3 + nat*4 + 1

        rdat.update({'lot':[lot],'species':spc})

        np.empty((nda,elm))

        Xi = []
        Ei = []
        Fi = []
        C1 = []
        C2 = []
        SD = []
        DP = []
        #print('File: ',file)
        for i,l in enumerate(fd[3].split("\n")[0:-1]):

            data = l.split(",")[0:-1]
            #if True:
            if len(data) == 3*3*nat+1+3*nat+3+3 or len(data) == 3*3*nat+1+3*nat+3+3+2:
               if len(data) == 3*3*nat+1+3*nat+3+3+2:
                  data = data[0:-2]
               #print(i,np.array(data).shape,3*3*nat+1+3*nat+3+3,l.count(","))
               Xi.append(np.array(data[0:nat * 3], dtype=np.float32).reshape(nat,3))
               Ei.append(data[nat*3])
               Fi.append(np.array(data[nat*3+1:nat*3+nat*3+1],dtype=np.float32).reshape(nat,3))
               C1.append(np.array(data[nat*3+nat*3+1:nat*3+nat*3+(nat+1)+1],dtype=np.float32))
               C2.append(np.array(data[nat*3+nat*3+(nat+1)+1:nat*3+nat*3+2*(nat+1)+1],dtype=np.float32))
               SD.append(np.array(data[nat*3+nat*3+2*(nat+1)+1:nat*3+nat*3+3*(nat+1)+1],dtype=np.float32))
               DP.append(np.array(data[nat*3+nat*3+3*(nat+1)+1:nat*3+nat*3+4*(nat+1)+(nat+1)*3+1],dtype=np.float32).reshape(nat+1,3))
               #DP.append(np.array(data[nat*3+nat*3+3*(nat+1)+1:nat*3+nat*3+4*(nat+1)+(nat+1)*3+1],dtype=np.float32).reshape(nat+1,3))
            else:
               print(i,np.array(data).shape,3*3*nat+1+3*nat+3+3,l.count(","))
               print('Line size does not match expected!')
               print('File:',file)
    else:
        exit(FileNotFoundError)

    if len(Xi) > 0:
        Xi = np.stack(Xi)
        Ei = np.array(Ei,dtype=np.float64)
        Fi = np.stack(Fi)
        C1 = np.stack(C1)
        C2 = np.stack(C2)
        SD = np.stack(SD)
        DP = np.stack(DP)
        rdat.update({'coordinates':Xi,
                     'energies':Ei,
                     'forces':Fi,
                     'hirshfeld':C1,
                     'cm5':C2,
                     'spindensities':SD,
                     'hirdipole':DP})

    return rdat
