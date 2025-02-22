#include <Arduino.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

// Include the quantized sine prediction model
#include "sin_predictor_model2.h"

#define INPUT_BUFFER_SIZE 64
#define OUTPUT_BUFFER_SIZE 64
#define INT_ARRAY_SIZE 8

// Function declarations
int string_to_array(char *in_str, int *int_array);
void print_int_array(int *int_array, int array_len);
int sum_array(int *int_array, int array_len);
// void test_auto_input();

// Serial communication buffers
char received_char = (char)NULL;
int chars_avail = 0;                    
char out_str_buff[OUTPUT_BUFFER_SIZE];  
char in_str_buff[INPUT_BUFFER_SIZE];    
int input_array[INT_ARRAY_SIZE];        

int in_buff_idx = 0;
int array_length = 0;
int array_sum = 0;

// Declare global variables for quantization parameters (from Python)
float input_scale = 0.15294118f;
int input_zero_point = 3;
float output_scale = 0.5166817f;
int output_zero_point = -63;


// TensorFlow Lite Globals
namespace {
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  constexpr int kTensorArenaSize = 10 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
}



void setup() {
  delay(5000);
  Serial.println("Test Project waking up");
  memset(in_str_buff, (char)0, INPUT_BUFFER_SIZE * sizeof(char)); 

  // Load the quantized sine predictor model
  model = tflite::GetModel(sin_predictor_quantized_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
      Serial.println("Model schema version mismatch.");
      return;
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
      Serial.println("AllocateTensors() failed");
      return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Setup complete.");

  // delay(1000);
  // test_auto_input();
}

void loop() {
  chars_avail = Serial.available(); 
  if (chars_avail > 0) {
    received_char = Serial.read();
    Serial.print(received_char);
    
    in_str_buff[in_buff_idx++] = received_char;
    
    if (received_char == 13) {  // 'Enter' key pressed
      in_str_buff[in_buff_idx] = '\0';

      Serial.print("About to process line: ");
      Serial.println(in_str_buff);

      array_length = string_to_array(in_str_buff, input_array);
      
      if (array_length != 7) {  // Ensure exactly 7 numbers
        Serial.println("Error: Please enter exactly 7 numbers.");
      } else {
        sprintf(out_str_buff, "Read in %d integers: ", array_length);
        Serial.print(out_str_buff);
        print_int_array(input_array, array_length);
        array_sum = sum_array(input_array, array_length);
        sprintf(out_str_buff, "Sums to %d\r\n", array_sum);
        Serial.print(out_str_buff);

        // Measure print time
        unsigned long t0 = micros();
        Serial.println("Processing input...");
        unsigned long t1 = micros();

        // Quantize the input (real value -> int8)
        if (input->type == kTfLiteInt8) {
          for (int i = 0; i < 7; i++) {
            input->data.int8[i] = static_cast<int8_t>(
              round((input_array[i] / input_scale) + input_zero_point)
            );
        }
        } else {
            Serial.println("Unsupported input type!");
            return;
        }


        // Run inference
        if (interpreter->Invoke() != kTfLiteOk) {
            Serial.println("Error running inference.");
        } else {
            unsigned long t2 = micros();

            // Extract the prediction result
            int prediction;
            if (output->type == kTfLiteInt8) {
              prediction = round((output->data.int8[0] - output_zero_point) * output_scale);
            }

            // Print model prediction and timing results
            Serial.print("Model Prediction: ");
            Serial.println(prediction);
            Serial.print("Printing time = ");
            Serial.print(t1 - t0);
            Serial.print(" µs. Inference time = ");
            Serial.print(t2 - t1);
            Serial.println(" µs.");
        }
      }

      memset(in_str_buff, (char)0, INPUT_BUFFER_SIZE * sizeof(char)); 
      in_buff_idx = 0;
    }
    else if (in_buff_idx >= INPUT_BUFFER_SIZE) {
      memset(in_str_buff, (char)0, INPUT_BUFFER_SIZE * sizeof(char)); 
      in_buff_idx = 0;
    }    
  }
}

int string_to_array(char *in_str, int *int_array) {
  int num_integers = 0;
  char *token = strtok(in_str, ",");

  while (token != NULL) {
    int_array[num_integers++] = atoi(token);
    token = strtok(NULL, ",");
    if (num_integers >= INT_ARRAY_SIZE) {
      break;
    }
  }

  return num_integers;
}

// int string_to_array(char *in_str, int *int_array) {
//   int num_integers = 0;
//   char *token = strtok(in_str, ",");  // Split by commas

//   while (token != NULL) {
//       for (int i = 0; token[i] != '\0'; i++) {
//           if (!isdigit(token[i]) && token[i] != '-' && token[i] != ' ', token[i] != '\n', token[i] != '\r', token[i] != '\t') {
//             Serial.print("Error: Invalid character detected: ");
//             Serial.println(token[i]);
//             return -1;
//           }
//       }

//       int_array[num_integers++] = atoi(token);

//       if (num_integers > 7) {
//           Serial.println("Error: Too many numbers entered.");
//           return -1;
//       }

//       token = strtok(NULL, ",");
//   }

//   if (num_integers != 7) {
//       Serial.println("Error: Please enter exactly 7 numbers.");
//       return -1;
//   }

//   return num_integers;
// }


// void print_int_array(int *int_array, int array_len) {
//   int curr_pos = 0;

//   sprintf(out_str_buff, "Integers: [");
//   curr_pos = strlen(out_str_buff);
//   for(int i = 0; i < array_len; i++) {
//     curr_pos += sprintf(out_str_buff + curr_pos, "%d, ", int_array[i]);
//   }
//   sprintf(out_str_buff + curr_pos, "]\r\n");
//   Serial.print(out_str_buff);
// }

void print_int_array(int *int_array, int array_len) {
  int curr_pos = 0;

  sprintf(out_str_buff, "Integers: [");
  curr_pos = strlen(out_str_buff);
  for(int i = 0; i < array_len; i++) {
    curr_pos += sprintf(out_str_buff + curr_pos, "%d, ", int_array[i]);
  }
  sprintf(out_str_buff + curr_pos, "]\r\n");
  Serial.print(out_str_buff);
}

int sum_array(int *int_array, int array_len) {
  int curr_sum = 0;
  for(int i = 0; i < array_len; i++) {
    curr_sum += int_array[i];
  }
  return curr_sum;
}



// void test_auto_input() {
//   // Define 5 test input arrays (you can adjust the values)
//   int test_input_arrays[5][7] = {
//       {3, 7, 12, 5, 10, 1, 4},
//       {2, 4, 6, 8, 10, 12, 14},
//       {1, 1, 1, 0, -1, -1, -1}, 
//       {9, 8, 7, 6, 5, 6, 7},
//       {-15, -14, -13, -12, -11, -10, -9}
//   };

//   // Loop through the 5 predefined arrays
//   for (int j = 0; j < 5; j++) {
//       // Print the current test array
//       Serial.print("Running automated test with the following input: ");
//       print_int_array(test_input_arrays[j], 7);

//       // Quantize the input (real value -> int8)
//       if (input->type == kTfLiteInt8) {
//         for (int i = 0; i < 7; i++) {
//           input->data.int8[i] = static_cast<int8_t>(
//             round((test_input_arrays[j][i] / input_scale) + input_zero_point)
//           );
//       }
//       } else {
//           Serial.println("Unsupported input type!");
//           return;
//       }

//       // Run inference
//       unsigned long t0 = micros();
//       if (interpreter->Invoke() != kTfLiteOk) {
//           Serial.println("Error running inference.");
//           return;
//       }
//       unsigned long t1 = micros();

//       // Extract the prediction result
//       int prediction;
//       if (output->type == kTfLiteInt8) {
//         prediction = round((output->data.int8[0] - output_zero_point) * output_scale);
//       }

//       // Print the prediction and timing results
//       Serial.print("Automated Model Prediction for Test ");
//       Serial.print(j + 1);
//       Serial.print(": ");
//       Serial.println(prediction);
//       Serial.print("Inference time = ");
//       Serial.print(t1 - t0);
//       Serial.println(" µs.");
//       delay(500);
//   }
// }