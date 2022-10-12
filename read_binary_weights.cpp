#include "read_binary_weights.h"

//returns n bit number
// unsigned int pack_float(float x, int n){
//     unsigned int tmp;
//     std::memcpy(&tmp, &x, sizeof(tmp));
//     int k = 32 - n;
//     return tmp >> k;
// }

// float unpack_float(unsigned int x, int n){
//     int k = 32 - n;
//     x <<= k;
//     float tmp;
//     std::memcpy(&tmp, &x, sizeof(tmp));
//     return tmp;
// }
float bit_change(float x, int n){
    unsigned int tmp;
    std::memcpy(&tmp, &x, sizeof tmp);
    int k = 32 - n; //move bit as k
    tmp >>= k;
    tmp <<= k;
    float result;
    std::memcpy(&result, &tmp, sizeof result);
    return result;
}

int32_t read_binary_weights(const char* path_to_weights_file, pico_cnn::naive::Tensor ***kernels, pico_cnn::naive::Tensor ***biases) {

    FILE *binary_file;
    binary_file = fopen(path_to_weights_file, "r"); //weights_file open

    if(binary_file != 0) {

        // Read magic number : 파일 유형을 특정해주는 magic number 읽기
        char magic_number[4];
        if(fread((void*)&magic_number, 1, 3, binary_file) != 3) {
            PRINT_ERROR("ERROR reading magic number")
            fclose(binary_file);
            return 1;
        }
        magic_number[3] = '\0'; //'\0' : 문자열의 끝을 나타냄

        if(strcmp(magic_number,"FD\n") != 0) { //FD : File Descripter
            PRINT_ERROR("ERROR: Wrong magic number: " << magic_number)
            fclose(binary_file);
            return 1;
        }
        PRINT_DEBUG(magic_number)

        // Read name
        char buffer[100];
        char c = '\0';
        uint32_t i = 0;
        for(; c != '\n'; i++){ //'\n'이 나올때까지 (문장이 끝날때까지) for loop
            if(fread((void*)&c, sizeof(c), 1, binary_file) != 1) {
                PRINT_ERROR("ERROR while reading name")
                fclose(binary_file);
                return 1;
            }
            buffer[i] = c;
        }
        buffer[i-1] = '\0'; //terminate string correctly

        PRINT_DEBUG(buffer)

        // Read number of layers
        uint32_t num_layers; //layer의 개수
        if(fread((void*)&num_layers, sizeof(num_layers), 1, binary_file) != 1) {
            PRINT_ERROR("ERROR reading number of layers")
            fclose(binary_file);
            return 1;
        } else {
            PRINT_DEBUG("Number of layers: " << num_layers)
        }

        // With the support for BatchNormalization we need separate counters
        // for kernels and biases as the BatchNormalization layer only has
        // four bias like arrays of values.
        uint32_t layer, kernel_idx, bias_idx;
        kernel_idx = 0;
        bias_idx = 0;

        //layer 의 개수만큼 반복
        for(layer = 0; layer < num_layers; layer++) {

            // Read layer name
            c = '\0';
            i = 0;
            for(; c != '\n'; i++) {
                if(fread((void*)&c, sizeof(c), 1, binary_file) != 1) {
                    PRINT_ERROR("ERROR while reading layer name")
                    fclose(binary_file);
                    return 1;
                }
                buffer[i] = c;
            }
            buffer[i-1] = '\0'; //terminate string correctly

            // Read layer type
            char buffer_layer_type[100];
            c = '\0';
            i = 0;
            for(; c != '\n'; i++) {
                if(fread((void*)&c, sizeof(c), 1, binary_file) != 1) {
                    PRINT_ERROR("ERROR while reading layer type")
                    fclose(binary_file);
                    return 1;
                }
                buffer_layer_type[i] = c;
            }
            buffer_layer_type[i-1] = '\0'; //terminate string correctly

            PRINT_DEBUG("Layer " << layer << ": " << buffer << " of type: " << buffer_layer_type)
            //layerㅣ 타입이 'Conv' 일 때
            if(strcmp(buffer_layer_type, "Conv") == 0) {

                uint32_t num_output_channels = 0;
                uint32_t num_input_channels = 0;
                uint32_t kernel_height = 0;
                uint32_t kernel_width = 0;

                //channel 과 kernel 의 크기를 읽어들임
                if (fread((void *) &num_output_channels, sizeof(num_output_channels), 1, binary_file) != 1) {
                    PRINT_ERROR("ERROR while reading number of output channels")
                    fclose(binary_file);
                    return 1;
                }
                if (fread((void *) &num_input_channels, sizeof(num_input_channels), 1, binary_file) != 1) {
                    PRINT_ERROR("ERROR while reading number of input channels")
                    fclose(binary_file);
                    return 1;
                }
                if (fread((void *) &kernel_height, sizeof(kernel_height), 1, binary_file) != 1) {
                    PRINT_ERROR("ERROR while reading kernel height")
                    fclose(binary_file);
                    return 1;
                }
                if (fread((void *) &kernel_width, sizeof(kernel_width), 1, binary_file) != 1) {
                    PRINT_ERROR("ERROR while reading kernel width")
                    fclose(binary_file);
                    return 1;
                }
                PRINT_DEBUG("Num output channels: " << num_output_channels << ", num input channels: " <<
                            num_input_channels << ", height: " << kernel_height << ", width: " <<
                            kernel_width << ", kernel_idx: " << kernel_idx)

                if(kernel_height != 0 && kernel_width != 0 && num_output_channels != 0 && num_input_channels != 0) { //모두 잘 읽혔을 때
                    //fp_t 는 parameter.h 에 정의되어 있음 => float 형(typedef 기법)
                    //auto 는 자동으로 형을 정해주는 역할
                    //values 에 kernel의 크기만큼 fp_t 형의 배열을 생성 (values 에 배열의 주소를 넣음)
                    auto *values = new float[kernel_height*kernel_width](); 

                    for (uint32_t out_ch = 0; out_ch < num_output_channels; out_ch++) {
                        for (uint32_t in_ch = 0; in_ch < num_input_channels; in_ch++) {
                            //fread 함수를 통해 kernel 크기만큼 float 형 데이터로 이루어진 배열을 읽어들여 values 에 넣는다.
                            if(fread((void *) values, sizeof(float), kernel_height * kernel_width, binary_file) != (kernel_height*kernel_width)) {
                                PRINT_ERROR("ERROR while reading kernel values.")
                                free(values);
                                fclose(binary_file);
                                return 1;
                            }

                            // for(int i = 0;i<5;i++){
                            //     printf("weight %d:%f\n", i,values[i]);
                            // }
                            
                            /* weight 값들을 int16_t 로 바꾸는 코드(train 자체를 int16_t 로 시키지 않았기 때문에 말도 안되는 코드임)
                            for (uint32_t i=0; i<sizeof(values); i++){
                                values[i] = (int16_t)values[i];
                            }*/

                            //printf("before weight : %f\n", values[0]);
			    
                            for (uint32_t i=0; i<sizeof(values); i++){
                                values[i] = bit_change(values[i], 16);
                            }
                            
                            //printf("after weight : %f\n", values[0]);

                            //memcpy 함수로 커널 크기만큼 values 가 가리키는 배열의 데이터를
                            std::memcpy((*kernels)[kernel_idx]->get_ptr_to_channel(out_ch, in_ch),
                                        values, kernel_height*kernel_width*sizeof(float));
                        }
                    }

                    kernel_idx++;

                    delete[] values;
                }

                uint32_t num_biases = 0;
                if (fread((void *) &num_biases, sizeof(num_biases), 1, binary_file) != 1) {
                    PRINT_ERROR("ERROR while reading number of biases")
                    fclose(binary_file);
                    return 1;
                }
                PRINT_DEBUG("Number of biases: " << num_biases)

                if (num_biases) {
                    auto *bias_values = new float[num_biases]();

                    if(fread((void *) bias_values, sizeof(float), num_biases, binary_file) != num_biases) {
                        PRINT_ERROR("ERROR while reading bias values.")
                        free(bias_values);
                        fclose(binary_file);
                        return 1;
                    }
                    std::memcpy((*biases)[bias_idx]->get_ptr_to_channel(0, 0), bias_values, num_biases*sizeof(float));

                    bias_idx++;

                    delete[] bias_values;
                }

            } else if (strcmp(buffer_layer_type, "BatchNormalization") == 0) {
                // read gamma values
                uint32_t num_gamma = 0;
                if(fread((void *) &num_gamma, sizeof(num_gamma), 1, binary_file) != 1) {
                    PRINT_ERROR("ERROR while reading number of gamma values")
                    fclose(binary_file);
                    return 1;
                }
                PRINT_DEBUG("Number of gamma values: " << num_gamma)

                if(num_gamma) {
                    auto* gamma_values = new float[num_gamma]();

                    if(fread((void *) gamma_values, sizeof(float), num_gamma, binary_file) != num_gamma) {
                        PRINT_ERROR("ERROR while reading gamma values.")
                        free(gamma_values);
                        fclose(binary_file);
                        return 1;
                    }
                    std::memcpy((*biases)[bias_idx]->get_ptr_to_channel(0, 0), gamma_values, num_gamma*sizeof(float));

                    bias_idx++;

                    delete[] gamma_values;
                }

                // read beta values
                uint32_t num_beta = 0;
                if(fread((void *) &num_beta, sizeof(num_beta), 1, binary_file) != 1) {
                    PRINT_ERROR("ERROR while reading number of beta values\n")
                    fclose(binary_file);
                    return 1;
                }
                PRINT_DEBUG("Number of beta values: " << num_beta)

                if(num_beta) {
                    auto *beta_values = new float[num_beta]();

                    if(fread((void *) beta_values, sizeof(float), num_beta, binary_file) != num_beta) {
                        PRINT_ERROR("ERROR while reading beta values.")
                        free(beta_values);
                        fclose(binary_file);
                        return 1;
                    }
                    std::memcpy((*biases)[bias_idx]->get_ptr_to_channel(0, 0), beta_values, num_beta*sizeof(float));

                    bias_idx++;

                    delete[] beta_values;
                }

                // read mean values
                uint32_t num_mean = 0;
                if(fread((void *) &num_mean, sizeof(num_mean), 1, binary_file) != 1) {
                    PRINT_ERROR("ERROR while reading number of beta values")
                    fclose(binary_file);
                    return 1;
                }
                PRINT_DEBUG("Number of mean values: " << num_mean)

                if(num_mean) {
                    auto *mean_values = new float[num_mean]();

                    if(fread((void *) mean_values, sizeof(float), num_mean, binary_file) != num_mean) {
                        PRINT_ERROR("ERROR while reading mean values.")
                        free(mean_values);
                        fclose(binary_file);
                        return 1;
                    }
                    std::memcpy((*biases)[bias_idx]->get_ptr_to_channel(0, 0), mean_values, num_mean*sizeof(float));

                    bias_idx++;

                    delete[] mean_values;
                }

                // read variance values
                uint32_t num_variance = 0;
                if(fread((void *) &num_variance, sizeof(num_variance), 1, binary_file) != 1) {
                    PRINT_ERROR("ERROR while reading number of variance values\n")
                    fclose(binary_file);
                    return 1;
                }
                PRINT_DEBUG("Number of variance values: " << num_variance)

                if(num_variance) {
                    auto *variance_values = new float[num_variance]();

                    if(fread((void *) variance_values, sizeof(float), num_variance, binary_file) != num_variance) {
                        PRINT_ERROR("ERROR while reading variance values.")
                        free(variance_values);
                        fclose(binary_file);
                        return 1;
                    }
                    std::memcpy((*biases)[bias_idx]->get_ptr_to_channel(0, 0), variance_values, num_variance*sizeof(float));

                    bias_idx++;

                    delete[] variance_values;
                }

            } else if (strcmp(buffer_layer_type, "Gemm") == 0 ||
                       strcmp(buffer_layer_type, "MatMul") == 0 ||
                       strcmp(buffer_layer_type, "Transpose") == 0) {

                uint32_t num_kernels = 0;
                uint32_t kernel_height = 0;
                uint32_t kernel_width = 0;

                if(fread((void *) &num_kernels, sizeof(num_kernels), 1, binary_file) != 1) {
                    PRINT_ERROR("ERROR while reading number of kernels")
                    fclose(binary_file);
                    return 1;
                }
                if(fread((void *) &kernel_height, sizeof(kernel_height), 1, binary_file) != 1) {
                    PRINT_ERROR("ERROR while reading kernel height")
                    fclose(binary_file);
                    return 1;
                }
                if(fread((void *) &kernel_width, sizeof(kernel_width), 1, binary_file) != 1) {
                    PRINT_ERROR("ERROR while reading kernel width")
                    fclose(binary_file);
                    return 1;
                }

                PRINT_DEBUG("Num kernels: " << num_kernels << ", height: " << kernel_height << ", width: " << kernel_width << ", kernel_idx: " << kernel_idx)

                uint32_t kernel;
                auto *values = new float[kernel_height*kernel_width]();

                if(num_kernels != 1)
                    PRINT_ERROR_AND_DIE("Number of kernels != 1")

                for(kernel = 0; kernel < num_kernels; kernel++) {
                    if(fread((void *) values, sizeof(float), kernel_height * kernel_width, binary_file) != (kernel_height*kernel_width)) {
                        PRINT_ERROR("ERROR while reading kernel values.")
                        free(values);
                        fclose(binary_file);
                        return 1;
                    }
                    for (uint32_t i=0; i<sizeof(values); i++){
                        values[i] = (int16_t)values[i];
                    }
                    std::memcpy((*kernels)[kernel_idx]->get_ptr_to_channel(0, kernel), values, kernel_height*kernel_width*sizeof(float));
                }

                kernel_idx++;

                delete[] values;

                uint32_t num_biases = 0;
                if(fread((void *) &num_biases, sizeof(num_biases), 1, binary_file) != 1) {
                    PRINT_ERROR("ERROR while reading number of biases")
                    fclose(binary_file);
                    return 1;
                }
                PRINT_DEBUG("Number of biases: " << num_biases)

                if(num_biases) {
                    auto *bias_values = new float[num_biases]();

                    if(fread((void *) bias_values, sizeof(float), num_biases, binary_file) != num_biases) {
                        PRINT_ERROR("ERROR while reading bias values.")
                        free(bias_values);
                        fclose(binary_file);
                        return 1;
                    }
                    std::memcpy((*biases)[bias_idx]->get_ptr_to_channel(0, 0), bias_values, num_biases*sizeof(float));

                    bias_idx++;

                    delete[] bias_values;
                }
            } else if (strcmp(buffer_layer_type, "Add") == 0) {
                uint32_t num_biases = 0;
                if(fread((void *) &num_biases, sizeof(num_biases), 1, binary_file) != 1) {
                    PRINT_ERROR("ERROR while reading number of biases")
                    fclose(binary_file);
                    return 1;
                }
                PRINT_DEBUG("Number of biases: " << num_biases)

                if (num_biases) {
                    auto *bias_values = new float[num_biases]();

                    if(fread((void *) bias_values, sizeof(float), num_biases, binary_file) != num_biases) {
                        PRINT_ERROR("ERROR while reading bias values.")
                        free(bias_values);
                        fclose(binary_file);
                        return 1;
                    }
                    std::memcpy((*biases)[bias_idx]->get_ptr_to_channel(0, 0), bias_values, num_biases*sizeof(float));

                    bias_idx++;

                    delete[] bias_values;
                }
            } else {
                PRINT_ERROR("ERROR: Unknown layer type \"" << buffer_layer_type << "\" in weights file. Layer number: " << layer)
                fclose(binary_file);
                return 1;
            }

        }
        PRINT_DEBUG("Layer idx: " << layer << ", kernel idx: " << kernel_idx << ", bias idx: " << bias_idx)

        // Read end marker
        char end_marker[5];
        if(fread((void*)&end_marker, 1, 4, binary_file) != 4) {
            PRINT_ERROR("ERROR reading end marker")
            fclose(binary_file);
            return 1;
        }
        end_marker[4] = '\0';

        if(strcmp(end_marker,"end\n") != 0) {
            PRINT_ERROR("ERROR: Wrong end marker read: " << end_marker)
            fclose(binary_file);
            return 1;
        }
        PRINT_DEBUG(end_marker)

    }
    fclose(binary_file);
    return 0;

}
