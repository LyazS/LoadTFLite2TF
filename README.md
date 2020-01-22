# Convert tflite to pb 
1. Load TFLite by using native TF operators
2. create .pb file

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

* check your path (test_json) in Code4TFv15/TFLite2TFv15.py

* run **Change2NewJson.py** and get the new json

* run **TFLite2TFv15.py** and get a .pb file

* then your can test the speed using .pb/.tflite/OpenCV.DNN

## Need to know
1. I just test the Project Mediapipe tflite,for your own using, check the ops definition in **Operators.py**
