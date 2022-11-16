from sklearn import metrics

file = open("./result.txt", "a")

test_label_path = "/home/kjh/Desktop/LAB_pico/pico-cnn_research/accuracy_test/test_program/val5000_label.txt"

bitlist =  [32,24,16,12,11,10,9,8]

for i in bitlist:

	predict_label_path = "/home/kjh/Desktop/LAB_pico/pico-cnn_research/accuracy_test/test_program/result/predict_label_{}bit.txt".format(i)

	#실제 라벨을 리스트로 변환
	with open(test_label_path) as f:
	    test_label = f.read().splitlines()
	
	#예측 라벨을 리스트로 변환
	with open(predict_label_path) as f:
	    predict_label = f.read().splitlines()
	
	# print(test_label)
	# print(predict_label)
	
	confusion_matrix = metrics.confusion_matrix(test_label, predict_label)
	
	print(confusion_matrix)
	
	print()
	
	file.write('{}bits \n'.format(i))
	accuracy = metrics.accuracy_score(test_label, predict_label)
	print('Accuracy : ', accuracy)
	file.write('Accuracy : {}\n'.format(accuracy))
	
	precision = metrics.precision_score(test_label, predict_label, average='micro')
	print('Precision : ', precision)
	file.write('Precision : {}\n'.format(precision))
	
	recall = metrics.recall_score(test_label, predict_label, average='micro')
	print('Recall : ', recall)
	
	f1 = metrics.f1_score(test_label, predict_label, average='micro')
	print("f1 score : ", f1)
	


file.close()
