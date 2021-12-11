import pickle

def loadPickle(fileName):
    pickleFile = open('.\\..\\data\\processed\\%s' % (fileName), 'rb')
    data = pickle.load(pickleFile)
    pickleFile.close()
    return data

def dumpPickle(data, fileName):
    pickleFile = open('.\\..\\data\\processed\\%s' % (fileName), 'wb')
    pickle.dump(data, pickleFile)
    pickleFile.close()
