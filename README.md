# LoadTFLite2TF
Load TFLite by using native TF operators

# Convert tflite to json 

* First, using google flatbuffer -- flatc，flatc can convert the tflite file to json format

* flatbuffer：https://github.com/google/flatbuffers 

* Install it：
    1. download the git 
    2. cmake -G "Unix Makefiles" //create the MakeFile
    3. make //create the flatc
    4. make install //install flatc

* Convert to json:
    1. copy the structure file 'schema.fbs' from tensorflow to the root of flatbuffer
    2. #./flatc -t schema.fbs -- xxxxx.tflite
    3. and you get the json
    4. using func tflite2json() change json to dictionary format json

# TODO 
1. Using TF1.x
