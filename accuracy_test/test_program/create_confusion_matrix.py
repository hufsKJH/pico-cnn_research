from sklearn import metrics

test_label_path = "/home/dilee/Desktop/pico-cnn_research/accuracy_test/test_label/test_label.txt"
predict_label_path = "/home/dilee/Desktop/pico-cnn_research/accuracy_test/test_program/predict_label.txt"

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

accuracy = metrics.accuracy_score(test_label, predict_label)
print('Accuracy : ', accuracy)

precision = metrics.precision_score(test_label, predict_label, average='micro')
print('Precision : ', precision)

recall = metrics.recall_score(test_label, predict_label, average='micro')
print('Recall : ', recall)

f1 = metrics.f1_score(test_label, predict_label, average='micro')
print("f1 score : ", f1)