#!/bin/bash

images_folder=/home/dilee/Desktop/pico-cnn/accuracy_test/test_images

predict_arr=() #예측 결과값

#폴더 내 이미지파일들을 읽고 예측하는 for 루프
for image_path in $images_folder/*
do
	#예측할 이미지의 경로
	echo $image_path

	#모델 실행
	result=`/home/dilee/Desktop/pico-cnn/onnx_import/generated_code/vgg16/vgg16 /home/dilee/Desktop/pico-cnn/onnx_import/generated_code/vgg16/network.weights.bin /home/dilee/Desktop/pico-cnn/onnx_import/generated_code/vgg16/imagenet.means /home/dilee/Desktop/pico-cnn/data/imageNet_labels/LOC_synset_mapping.txt $image_path`

	predict_arr+=("$result")
	#echo $result >> predict.txt
done

#predict.txt에 저장
for predict in "${predict_arr[@]}"; do
		echo $predict >> predict.txt
done



#/home/dilee/Desktop/pico-cnn/onnx_import/generated_code/vgg16/vgg16 /home/dilee/Desktop/pico-cnn/onnx_import/generated_code/vgg16/network.weights.bin /home/dilee/Desktop/pico-cnn/onnx_import/generated_code/vgg16/imagenet.means /home/dilee/Desktop/pico-cnn/data/imageNet_labels/imagenet1000_clsidx_to_labels.txt /home/dilee/Desktop/pico-cnn/onnx_import/generated_code/vgg16/test_images/cat.jpg
