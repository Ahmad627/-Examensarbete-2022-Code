Author : Ahmad Al Halabi

// Import TensorFlow stuff
#include "TensorFlowLite.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/version.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"

// Vår modell
#include "sine_model.h"

#define DEBUG 1

// inställningar
constexpr int led_pin = 2;
constexpr float pi = 3.14159265;                  // Some pi
constexpr float freq = 0.5;                       // Frequency (Hz) of sinewave
constexpr float period = (1 / freq) * (1000000);  // Period (microseconds)
float sampleSum = 0;
float sqDevSum = 0.0;
int SAMPLES = 500;

// // TFLite globals, används för kompatibilitet med Arduino-skisser
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* model_input = nullptr;
  TfLiteTensor* model_output = nullptr;

  
   
  //Skapa ett minnesområde att använda för input, output och andra TensorFlow-arrayer. 
  constexpr int kTensorArenaSize = 5 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
} // namespace

bool fHasLooped  = false;

void setup() {

 // Vänta tills Serial ansluts
#if DEBUG
  while(!Serial);
#endif

  pinMode(led_pin, OUTPUT);
 
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Mappa modellen till en användbar datastruktur
  model = tflite::GetModel(sine_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model version does not match Schema");
    while(1);
  }

   
  static tflite::AllOpsResolver resolver;

  // Bygg en tolk (interpreter) för att köra modellen 
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize,
    error_reporter);
  interpreter = &static_interpreter;

  // Tilldela minne från tensor_arenan för modellens tensorer
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    while(1);
  }

  // Tilldela modellin- och utgångsbuffertar (tensorer) till pekare
  model_input = interpreter->input(0);
  model_output = interpreter->output(0);

  // Få information om minnesområdet som ska användas för modellens inmatning.
#if DEBUG
  Serial.print("Number of dimensions: ");
  Serial.println(model_input->dims->size);
  Serial.print("Dim 1 size: ");
  Serial.println(model_input->dims->data[0]);
  Serial.print("Dim 2 size: ");
  Serial.println(model_input->dims->data[1]);
  Serial.print("Input type: ");
  Serial.println(model_input->type);
#endif
}

void loop() {
  if ( fHasLooped == false )
   {
     for(int count = 0; count <=500; count++)
      {

        #if DEBUG
          unsigned long start_timestamp = millis();
        #endif
        
          // Få aktuell tidsstämpel och modulo med period
          unsigned long timestamp = millis();
          timestamp = timestamp % (unsigned long)period;
        
          // Beräkna x-värdet för att mata till modellen
          float x_val = ((float)timestamp * 2 * pi) / period;
        
          // Kopiera värde till ingångsbuffert (tensor)
          model_input->data.f[0] = x_val;
        
          // Kör slutledning
          TfLiteStatus invoke_status = interpreter->Invoke();
          if (invoke_status != kTfLiteOk) {
            error_reporter->Report("Invoke failed on input: %f\n", x_val);
          }
        
          // Läs förutsagt y-värde från utgångsbuffert (tensor)
          float y_val = model_output->data.f[0];
            
          for(int i = 0; i < SAMPLES; i++) 
          {
             sampleSum += y_val ;
             //delay(20); // set this to whatever you want
           }

                float meanSample = sampleSum/float(SAMPLES);

          for(int i = 0; i < SAMPLES; i++) 
          {
            // pow(x, 2) is x squared.
              sqDevSum += pow((meanSample - float(y_val)), 2);
          }
          
            float stDev = sqrt(sqDevSum/(float(SAMPLES) -1));
            Serial.println("Standard dviation");
            Serial.println(stDev);
            Serial.println("  ");
            
          int brightness = (int)(255 * y_val);
          analogWrite(led_pin, brightness);
       
          // Skriv ut värdena
          Serial.println("y: predicted values");
          Serial.println(y_val);
          Serial.println("  ");
          
        #if DEBUG
          Serial.print("Time for inference (ms): ");
          Serial.println(millis() - start_timestamp);
        #endif
        
        //delay(4000);
     }

      fHasLooped = true;
   }
}
