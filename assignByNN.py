#!/usr/bin/python3.8
import glob, os, sys, re, math

import numpy as np
import tensorflow as tf
import tensorflow.keras.models

os.chdir(".")
NUM_THREADS = 12
config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=NUM_THREADS)
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

maxP2P = 8.0+ 0.000001 ##, 9.0  # the .asg file has p2p pairwise distance threshold of 9.0A

# 21 for maxP2p=7.0A; 27 for maxP2p=8.0A; 30 for maxP2p=9.0A:
# the maximum number of p2p distances in all the procssed PDBs
noOfp2pPoint_max = 27 
p2p_list = [0.0]*(noOfp2pPoint_max+2)
seq_list = [0.0]*noOfp2pPoint_max

noOfAngle = 5 # 5 angles 
startingAngleIndexFromEnd = noOfAngle # 5 angles 
angle_list=[1.0] * noOfAngle

deg2Radius = math.pi / 180.0

p2pTokenIndex = 2 # the starting index for p2p distance/index sequence is 2 
rowData = [] # * noOfColumn = 65

minNoOfToken = 7  # 2 tokens before the p2p distance/index statement, plus 5 (angles)
angleTokenIndex = 0 #
asgTokenIndex = 0

noOfp2pPoint = 0
noOfToken = 0
minProb = 0.001

p2p = 0.0
zeroList = [0.0, 0.0, 0.0, 0.0, 0.0]

def asgReader(fileName):
    sFile = open(fileName, 'r')
    content = sFile.readlines()
    noOfLine = len(content)
    noOfResidue = noOfLine - 2; ## since there are 2 extra lines in the asg file
    
    p2pList = []
    seqList = []
    angList = []
    rAngList = []
    residID = []
    
    lineCnt = 0
    while lineCnt < noOfLine:  ##if noOfLine > 1:
        line = content[lineCnt]
        txtLine = line.strip()
        tokens = txtLine.split() ##
        noOfToken = len(tokens)
        
        if  noOfToken > minNoOfToken and tokens[0].strip() != 'Agreement:':
            noOfp2pPoint = int((noOfToken - minNoOfToken) / 2)

            residID.append(tokens[0])
            
            p2p_list[0] = float(tokens[1].strip())
            if lineCnt > 0:                
                p2p_list[1] = p2pList[lineCnt-1][0]
            else:
                p2p_list[1] = 3.20  ## the mean value of uType

            seqTokenIndex = noOfp2pPoint + p2pTokenIndex

            for i in range(noOfp2pPoint_max):
                if i < noOfp2pPoint:
                    p2p = float(tokens[p2pTokenIndex + i].strip() )
                    if  p2p < maxP2P:
                        p2p_list[i+2] = float(tokens[p2pTokenIndex + i].strip())
                        seq_list[i] = float(tokens[seqTokenIndex + i].strip())
                    else:
                        p2p_list[i+2] = 0.0
                        seq_list[i] = 0.0
                else:
                   p2p_list[i+2] = 0.0
                   seq_list[i] =   0.0

            angleTokenIndex = noOfToken - startingAngleIndexFromEnd  #
            asgTokenIndex = noOfToken - 2         #should leave out if NO assignment
            for i in range(noOfAngle):
                angle_list[i]= deg2Radius * float(tokens[angleTokenIndex + i].strip())
                
            p2pList.append(list(p2p_list)) 
            seqList.append(list(seq_list))  
            angList.append(list(angle_list)) 

        lineCnt += 1

    sFile.close()

    ### Append anlges to each residue of the angles of the previous 5 residues 
    rAngleList = [0.0] * noOfAngle;  

    ## the 0th residues 
    rAngList.append(list(rAngleList))

    ## the 1st residue
    rAngleList[0] = angList[0][0]
    rAngList.append(list(rAngleList))

    ## the 2nd residue
    rAngleList[0] = angList[1][0]
    rAngleList[1] = angList[0][1]
    rAngList.append(list(rAngleList))

    ## the 3rd residue
    rAngleList[0] = angList[2][0]
    rAngleList[1] = angList[1][1]
    rAngleList[2] = angList[0][2]
    rAngList.append(list(rAngleList))

    ## the 4th residue
    rAngleList[0] = angList[3][0]
    rAngleList[1] = angList[2][1]
    rAngleList[2] = angList[1][2]
    rAngleList[3] = angList[0][3]
    rAngList.append(list(rAngleList))

    ## the 5th residue
    rAngleList[0] = angList[4][0]
    rAngleList[1] = angList[3][1]
    rAngleList[2] = angList[2][2]
    rAngleList[3] = angList[1][3]
    rAngleList[4] = angList[0][4]
    rAngList.append(list(rAngleList))

    for i in range (6, len(angList) ):
        rAngleList[0] = angList[i-1][0]
        rAngleList[1] = angList[i-2][1]
        rAngleList[2] = angList[i-3][2]
        rAngleList[3] = angList[i-4][3]
        rAngleList[4] = angList[i-5][4]
        rAngList.append(list(rAngleList))

    noOfResidue = len(p2pList)

    p2pData = []

    for i in range (noOfResidue):
        rowData = []
        rowData.extend( p2pList[i] )
        rowData.extend( zeroList )
        rowData.extend( seqList[i] )
        rowData.extend( zeroList )
        rowData.extend( angList[i] )
        rowData.extend( zeroList )
        rowData.extend( rAngList[i] )
        
        p2pData.append(list(rowData))  ## copy the row data

    return residID, p2pData

def assignSSE(model, residID, p2pDat, asgFile):

    X = np.array(p2pData)
    noOfRow = X.shape[0]
    noOfFeature = X.shape[1]
    X = X.reshape(noOfRow, noOfFeature, 1 )
    
    prediction = model.predict(X) #    print(type(prediction))
    rows = prediction.shape[0]

    p2pH = 0
    p2pE = 0
    p2pG = 0
    p2pT = 0
    p2pU = 0
    p2pB = 0

    for x in range(0, rows):

        predict = prediction[x]
        sorted_predict = np.argsort(predict)
        indexOfMaxProb = np.argmax(prediction[x])
        indexOf2ndMaxProb  = sorted_predict[-2]

        predtype = 'U'

        if indexOfMaxProb == 0:
            predtype = 'H'
            p2pH = p2pH + 1
        elif indexOfMaxProb == 1:
            predtype = 'E'
            p2pE = p2pE + 1
        elif indexOfMaxProb == 2:
            predtype = 'B'
            p2pB = p2pB + 1
        elif indexOfMaxProb == 3:
            predtype = 'G'
            p2pG = p2pG + 1
        elif indexOfMaxProb == 4:
            predtype = 'T'
            p2pT = p2pT + 1
        elif indexOfMaxProb == 5:
            predtype = 'U'
            p2pU = p2pU + 1

        predtype2 = 'u'
        if indexOf2ndMaxProb == 0:
            predtype2 = 'h'
        elif indexOf2ndMaxProb == 1:
            predtype2 = 'e'
        elif indexOf2ndMaxProb == 2:
            predtype2 = 'b'
        elif indexOf2ndMaxProb == 3:
            predtype2 = 'g'
        elif indexOf2ndMaxProb == 4:
            predtype2 = 't'
        elif indexOf2ndMaxProb == 5:
            predtype2 = 'u'

        asgFile.write("%8s    "  %(residID[x]))
        for items in prediction[x]: ##sorted_predict:
            if items > minProb:
                asgFile.write("%6.4f  " %(items))
            else:
                asgFile.write('0.0     ')

        asgFile.write('  '+predtype+'/'+predtype2+'\n')

    ## output the SSE content
    asgFile.write('-----\n')
    asgFile.write("{0} {1:4d} \n".format('H:', p2pH))
    asgFile.write("{0} {1:4d} \n".format('G:', p2pG))
    asgFile.write("{0} {1:4d} \n".format('E:', p2pE))
    asgFile.write("{0} {1:4d} \n".format('B:', p2pB))
    asgFile.write("{0} {1:4d} \n".format('T:', p2pT))
    asgFile.write("{0} {1:4d} \n".format('U:', p2pU))
    asgFile.write("{0} {1:5d} \n".format('No of Residues:', p2pH + p2pG + p2pE + p2pB + p2pT + p2pU ))     

if __name__=="__main__":

    model = tf.keras.models.load_model('best_asg0921_8A5z.h5')

    noOfResidue = 0

    p2pHt = 0
    p2pEt = 0
    p2pGt = 0
    p2pTt = 0
    p2pUt = 0
    p2pBt = 0

    total = len(sys.argv)
#    if total > 1:
#        fileName = sys.argv[1].strip() #
    for fileName in glob.glob("*.asg"):
        residID, p2pData = asgReader(fileName)
        asgFile = open(fileName[:-4]+'.lbl',"w+")
        assignSSE(model, residID, p2pData, asgFile)
        asgFile.close()
#    else:
#        print("\nPlease specify an input\n")



