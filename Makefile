all:
	g++ Principal.cpp --std=c++17 -I/home/bryan/opencv/opencvi/include/opencv5/ -L/home/bryan/opencv/opencvi/lib/ -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_video -lopencv_videoio -lopencv_features2d -lopencv_objdetect -lopencv_dnn -lopencv_dnn_superres -lopencv_superres -lopencv_cudaimgproc -lopencv_cudafilters -lopencv_cudaarithm -lcudart -o vision.bin

saludo:
	echo "Hola C++"

clean:
	rm -f vision.bin

run:
	./vision.bin
