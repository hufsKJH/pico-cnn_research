#define JPEG
#define IMAGENET

#define IMAGE_SIZE 224

#include <iostream>
#include <cstdlib>

#include <string>
#include <vector>
#include <sstream>

#include "pico-cnn/pico-cnn.h"
#include "network.h"
using namespace std;


void usage() {
    printf("./vgg[16|19] \\\n");
    printf("PATH_TO_EKUT_ES_ALEXNET_WEIGHTS.weights \\\n");
    printf("PATH_TO_MEANS_FILE.means \\\n");
    printf("PATH_TO_IMAGE_NET_LABELS.txt \\\n");
    printf("PATH_TO_IMAGE.jpg\n");
}

/**
 * @brief sorts the prediction and the labels (in place) of the network such that the label with the
 * highes prediction is at the front of the array (position 0)
 *
 * @param prediction (1 x length)
 * @param labels (1 x length)
 * @param length
 */
void sort_prediction(fp_t* prediction, uint16_t* labels_pos, const uint16_t length) {
    //예측값 확률 높은 순으로 정렬 함수
    // simple bubble sort
    uint16_t i,j;

    fp_t temp_prediction;
    uint16_t temp_label;

    for(i = 0; i < length-1; i++) {
        for(j = 0; j < length-1-i; j++) {
            if(prediction[j] < prediction[j+1]) {
                // swap
                temp_prediction = prediction[j];
                prediction[j] = prediction[j+1];
                prediction[j+1] = temp_prediction;

                temp_label = labels_pos[j];
                labels_pos[j] = labels_pos[j+1];
                labels_pos[j+1] = temp_label;
            }
        }
    }
}

vector<string> split(string str, char Delimiter) { //split 함수
    istringstream iss(str);             // istringstream에 str을 담는다.
    string buffer;                      // 구분자를 기준으로 절삭된 문자열이 담겨지는 버퍼
 
    vector<string> result;
 
    // istringstream은 istream을 상속받으므로 getline을 사용할 수 있다.
    while (getline(iss, buffer, Delimiter)) {
        result.push_back(buffer);               // 절삭된 문자열을 vector에 저장
    }
 
    return result;
}

int32_t main(int32_t argc, char** argv) {

    if(argc != 5) {
        PRINT_ERROR("Too few or to many arguments!\n")
        usage();
        return 1;
    }

    char weights_path[1024];
    char means_path[1024];
    char labels_path[1024];
    char jpeg_path[1024];

    //매개변수로 들어온 각각의 경로를 순서대로 각각의 변수에 복사해서 넣음 
    strcpy(weights_path, argv[1]);
    strcpy(means_path, argv[2]);
    strcpy(labels_path, argv[3]);
    strcpy(jpeg_path, argv[4]);

    uint32_t i;

    Network *net = new Network();

    //PRINT_INFO("Reading weights from " << weights_path)

    // read binary_weights
    //read_binary_weights.cpp 에서 불러온 함수
    if(read_binary_weights(weights_path, &net->kernels, &net->biases) != 0) {
        PRINT_ERROR("Could not read weights from " << weights_path)
        return 1;
    }

    // read means
    //PRINT_INFO("Reading means from " << means_path)

    fp_t* means = (fp_t*) malloc(3*sizeof(fp_t));

    if(read_means(means_path, means) != 0) {
        PRINT_ERROR("Could not read means file " << means_path)
        return 1;
    }

    // read labels
    //PRINT_INFO("Reading labels from " << labels_path)

    char** labels;
    int32_t num_labels;
    num_labels = read_imagenet_labels(labels_path, &labels, 1000); //imageNet label 의 개수 : 1000개

    if(num_labels != 1000) {
        PRINT_ERROR("Could not read imagenet labels " << labels_path)
        return 1;
    }

    // read input image
    //PRINT_INFO("Reading input image " << jpeg_path)

    fp_t** pre_mean_input;

    uint16_t height;
    uint16_t width;

    if(read_jpeg(&pre_mean_input, jpeg_path, 0.0, 255.0, &height, &width) != 0) {
        PRINT_ERROR("Could not read jpeg from " << jpeg_path)
        return 1;
    }

    // substract mean from each channel
    fp_t** input = (fp_t**) malloc(3*sizeof(fp_t*));
    input[0] = (fp_t*) malloc(IMAGE_SIZE*IMAGE_SIZE*sizeof(fp_t));
    input[1] = (fp_t*) malloc(IMAGE_SIZE*IMAGE_SIZE*sizeof(fp_t));
    input[2] = (fp_t*) malloc(IMAGE_SIZE*IMAGE_SIZE*sizeof(fp_t));

    uint16_t row;
    uint16_t column;

    for(i = 0; i < 3; i++) {
        for(row = 0; row < height; row++) {
            for(column = 0; column < height; column++) {
                input[i][row*width+column] = pre_mean_input[i][row*width+column] - means[i];
            }
        }
    }

    // free pre mean input image
    for(i = 0; i < 3; i++) {
        free(pre_mean_input[i]);
    }
    free(pre_mean_input);

    // free means
    free(means);

    fp_t *output  = (float*) malloc(1000*sizeof(float));

    pico_cnn::naive::Tensor *input_tensor = new pico_cnn::naive::Tensor(1, 3, 224, 224);
    pico_cnn::naive::Tensor *output_tensor = new pico_cnn::naive::Tensor(1, 1000);

    std::memcpy(input_tensor->data_+(0*224*224), input[0], 224*224*sizeof(fp_t));
    std::memcpy(input_tensor->data_+(1*224*224), input[1], 224*224*sizeof(fp_t));
    std::memcpy(input_tensor->data_+(2*224*224), input[2], 224*224*sizeof(fp_t));

    //PRINT_INFO("Starting CNN");

    net->run(input_tensor, output_tensor);

    //PRINT_INFO("After CNN");

    // free memory
    // input
    free(input[0]);
    free(input[1]);
    free(input[2]);
    free(input);

    // print prediction
    uint16_t* labels_pos;
    labels_pos = (uint16_t*) malloc(1000*sizeof(uint16_t));

    for(i = 0; i < 1000; i++) {
        labels_pos[i] = i;
    }

    std::memcpy(output, output_tensor->data_, 1000*sizeof(fp_t));
    sort_prediction(output, labels_pos, 1000);

    // PRINT_INFO("Prediction:");

    //std::cout << labels(labels_pos[0]);
    //결과

    //PRINT_INFO(labels[labels_pos[0]]);
    
    //예측값 추출하기
    string result_raw = labels[labels_pos[0]];
    
    /*
    cout << result_raw << endl;
    출력값 예시: n02120079 Arctic fox, white fox, Alopex lagopus
    */

    //클래스 숫자만 추출해야함, 공백기준으로 나누기 위해split 함수 사용
    vector<string> result = split(result_raw, ' ');

    //배열의 첫번째 원소가 클래스 번호값
    cout << result[0] << endl;    

    /* 라벨의 이름만 추출하는 코드(임의 작성/예전 라벨 기준)

     int start_idx = result_raw.find("\'");

    for (int i = start_idx+1; i < result_raw.size(); i++){
        if(result_raw[i]==39){//아스키코드표 참조 39번 :어퍼스트로피
                break;
            }
        result = result + result_raw[i];
        
        }
    */

    /* 예측 상위 10개 출력(기존 코드)
    for(i = 0; i < 10; i++) {
        PRINT_INFO(i+1 << ":\t" << output[i] << "\t" << labels[labels_pos[i]]);
    }
    */

    free(output);

    for(i = 0; i < 1000; i++) {
        free(labels[i]);
    }
    free(labels);

    free(labels_pos);

    delete net;

    return 0;
}



