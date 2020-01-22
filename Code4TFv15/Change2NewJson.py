import os

def tflite2json(pathIn,pathDst):
    f = open(pathIn)  
    line = f.readline()  
    fout = open(pathDst, 'w')

    while line:
        dstline = 'aaa'
        if line.find(':') != -1:
            quoteIdx2 = line.find(':')
            linenew = line[:quoteIdx2] + '"' + line[quoteIdx2:]
            quoteIdx1 = linenew.rfind(' ', 0, quoteIdx2)
            dstline = linenew[:quoteIdx1 + 1] + '"' + linenew[quoteIdx1 + 1:]
            fout.write(dstline + os.linesep)
        else:
            dstline = line
            fout.write(line)
        line = f.readline()
    f.close()
    fout.close()
    print("Convert Done.")


pathIn = 'hand_landmark.json'
pathDst = 'hand_landmark_new.json'
# pathIn = 'palm_detection.json'
# pathDst = 'palm_detection_new.json'
tflite2json(pathIn,pathDst)

