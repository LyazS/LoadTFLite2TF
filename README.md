# LoadTFLite2TF
Load TFLite by using native TF operators

# How to use
## Convert tflite to json 

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

## Load into TF native ops

* check your path (test_json) in Code4TFv13/TFLite2TFv13.py or Code4TFv15/TFLite2TFv15.py

* run it : **python TFLite2TFv15.py** and get a .pb file

* then your can load the .pb file to test
    1. construct a np.array for your network inputs
    2. **python loadpb.py**
    3. check your outputs

## Need to know
I just test the Project Mediapipe tflite,for your own using, check the ops definition
