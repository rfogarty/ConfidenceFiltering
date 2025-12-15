import numpy as np
import pandas as pd

def takePosition(s,position) :
    try :
        s = s.split('/')[position]
        return s
    except IndexError as e:
        print(e)
        print(f'Length of split: {len(s.split("/"))}')
        print(f'String {s} did not split properly on position={position}')
        exit()


def fixPatchNames(df,position) :
    tempStrings = np.array(df['PatchName'])
    tempStrings = np.array([takePosition(s,position) for s in tempStrings])
    tempStrings = np.array([s.removesuffix('.tiff') for s in tempStrings])
    tempStrings = np.array([s.removesuffix('.png') for s in tempStrings])
    tempStrings = np.array([s.replace(',',';') for s in tempStrings])
    df['PatchName'] = tempStrings
    return df


def assignTruth(s) :
    if 'GS3' in s :
        return 0
    else :
        return 1


def processTruth(df) :
    # TODO: ensure that this does the same as below!
    #df['Truth'] = df['PatchName'].str.contains('_GS4_').astype(int)
    tempStrings = np.array(df['PatchName'])
    tempTruth = np.array([assignTruth(s) for s in tempStrings])
    df['Outcome_truth'] = tempTruth
    return df


def openCSV(filename) :
    return pd.read_csv(filename)


#def loadLogs() :
#    logs.append(openCSV('log0.csv'))
#    logs.append(openCSV('log1.csv'))
#    logs.append(openCSV('log2.csv'))
#    logs.append(openCSV('log3.csv'))
#    logs.append(openCSV('log4.csv'))
#    logs.append(openCSV('log5.csv'))
#    logs.append(openCSV('log6.csv'))
#    logs.append(openCSV('log7.csv'))
#    logs.append(openCSV('log8.csv'))
#    logs.append(openCSV('log9.csv'))
#    if os.path.exists('log19.csv') :
#        logs.append(openCSV('log10.csv'))
#        logs.append(openCSV('log11.csv'))
#        logs.append(openCSV('log12.csv'))
#        logs.append(openCSV('log13.csv'))
#        logs.append(openCSV('log14.csv'))
#        logs.append(openCSV('log15.csv'))
#        logs.append(openCSV('log16.csv'))
#        logs.append(openCSV('log17.csv'))
#        logs.append(openCSV('log18.csv'))
#        logs.append(openCSV('log19.csv'))
#        logs3D=np.array([logs[0],logs[1],logs[2],logs[3],logs[4],logs[5],logs[6],logs[7],logs[8],logs[9],
#                         logs[10],logs[11],logs[12],logs[13],logs[14],logs[15],logs[16],logs[17],logs[18],logs[19]])
#    else :
#        logs3D=np.array([logs[0],logs[1],logs[2],logs[3],logs[4],logs[5],logs[6],logs[7],logs[8],logs[9]])
#    
#    logs2D=np.mean(logs3D,axis=0)
#    logsAll = pd.concat(logs)
#    logsAvg=pd.DataFrame(logs2D,columns=logs[0].columns.values)
#    logsParallel = pd.concat(logs,axis=1)
#    return (logsAll,logsParallel,logsAvg,logs)

def loadLogsList(indexList) :
    logs = []
    for idx in indexList :
        if os.path.exists(f'log{idx}.csv') :
            df = openCSV(f'log{idx}.csv')
            df['Fold'] = idx
            logs.append(df)
        else :
            print(f'WARNING: file log{idx}.csv does not exist')
    logsAll = pd.concat(logs)
    logsAll.reset_index(drop=True,inplace=True)
    return logsAll


def loadLogsRange(startIndex,endIndex) :
    assert(startIndex <= endIndex)
    return loadLogsList(range(startIndex,endIndex))


def loadResultList(indexList) :
    for idx in indexList :
        if os.path.exists(f'Results_{idx}.csv') :
            res = pd.read_csv(f'Results_{idx}.csv')
            df = pd.concat((df,res))
        else :
            print(f'WARNING: file Results_{idx}.csv does not exist')
    return df


def loadResultRange(startIndex,endIndex) :
    assert(startIndex <= endIndex)
    return loadResultList(range(startIndex,endIndex))

def parseMetric(files,pattern) :
    metric=subprocess.run(f'cat {" ".join(files)} | grep -A1 "{pattern}" | grep -v "{pattern}\\|--" | xargs -n1',
                          shell=True,stdout=subprocess.PIPE).stdout.decode('utf-8').replace('[','').replace(']','').split('\n')
    metrica=[]
    for a in metric :
        if a != '' :
            metrica.extend(str2np(a))
    return np.array(metrica)

def parseMetric(files,pattern) :
    #metric=subprocess.run(f'cat {" ".join(files)} | grep "{pattern}" | cut -d: -f2 | cut -d" " -f 2 | cut -d"(" -f1 | xargs -n1',
    metric=subprocess.run(f'cat {" ".join(files)} | grep "{pattern}" | cut -d: -f2 | cut -d" " -f 2 | xargs -n1',
                          shell=True,stdout=subprocess.PIPE).stdout.decode('utf-8').split('\n')
    metric=[float(a) for a in metric if a != '']
    return metric

#def str2np(strArray):
#   lItems = [] 
#   width = None
#   for line in strArray.split("\n"): 
#       lParts = line.split(',') 
#       n = len(lParts) 
#       if n==0: 
#           continue
#       if width is None: 
#           width = n 
#       else: 
#           assert n == width, "invalid array spec"
#       for str in lParts :
#           if str != '' :
#               lItems.append(float(str)) 
#   return lItems 


def str2np(strArray): 
   lItems = [] 
   width = None
   for line in strArray.split("\n"): 
       lParts = line.split() 
       n = len(lParts) 
       if n==0: 
           continue
       if width is None: 
           width = n 
       else: 
           assert n == width, "invalid array spec"
       for str in lParts :
           if str != '' :
               lItems.append(float(str)) 
       #lItems.append([float(str) for str in lParts]) 
   return np.array(lItems).astype(int) 


def parse4x4ConfusionMatrices(files,pattern) :

    confusionMatrix=np.zeros((4,4))
    for f in files :
        strmatrix=subprocess.run(f'grep "{pattern}" -A4 {f} | tail -n4',
                              shell=True,stdout=subprocess.PIPE).stdout.decode('utf-8').replace('[','').replace(']','')
        confMat=str2np(strmatrix)
        confusionMatrix+=confMat
    return confusionMatrix


def parse2x2ConfusionMatrices(files,pattern) :

    confusionMatrix=np.zeros((2,2))
    for f in files :
        #strmatrix=subprocess.run(f'grep "{pattern}" -A4 {f} | tail -n4',
        strmatrices=subprocess.run(f'grep "{pattern}" -A2 {f} | grep -v "{pattern}\\|--"',
                              shell=True,stdout=subprocess.PIPE).stdout.decode('utf-8').replace('[','').replace(']','')
        # now parse each 4 lines separately
        strmatrix=""
        count=0
        totalcount=0
        for line in strmatrices.split("\n") :
            strmatrix=f'{strmatrix}{line}\n'
            count+=1
            if count == 2 :
                count=0
                totalcount+=1
                if totalcount < 10 :
                   print(f'matrix is: {strmatrix}')
                confMat=str2np(strmatrix)
                confusionMatrix+=confMat
                strmatrix=""
    return confusionMatrix

