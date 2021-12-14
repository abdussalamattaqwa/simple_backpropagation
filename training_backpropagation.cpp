#include <iostream>
#include <list>
#include <cstdlib>
#include <math.h>


// Eksternal file
#include "data/data_training.txt"
#include "data/data_target.txt"
// function for sigmoid activation
float sigmoid(float x) { return 1 / (1 + exp(-x)); }
// function for sigmoid derivative
float dSigmoid(float x) { return x * (1 - x); }

// function untuk inisialisasi data random weight
float init_weight() { return ((float)rand())/((float)RAND_MAX); }


int  main(){
    static const char huruf[7] = {'A', 'B', 'C', 'D', 'E', 'J', 'K'};

    //  number of input  = 63
    static const int numInputs = 63;    

    //  number of hidden layer = 4
    static const int numHiddenNodes = 32;    
    float z[numHiddenNodes];

    //  number of output = 7
    static const int numOutputs = 7;
    float y[numOutputs];

    static const float learning_rate = 0.1f;

    // STEP 0============================================
    // Initialize weights (Set to small random values).
    // weight hidden 
    float weight_v[numInputs][numHiddenNodes];
    for (int i = 0; i < numInputs; i++)
    {
        for (int j = 0; j < numHiddenNodes; j++)
        {
            weight_v[i][j] = init_weight();
        }
    }
    // bias hidden
    float bias_v[numHiddenNodes];
    for (int i = 0; i < numHiddenNodes; i++)
    {
        bias_v[i] = init_weight();
    }


    // weight output
    float weight_w[numHiddenNodes][numOutputs];
    for (int j = 0; j < numHiddenNodes; j++)
    {
        for (int k = 0; k < numOutputs; k++)
        {
            weight_w[j][k] = init_weight();
        }
    }
    // bias output
    float bias_w[7];
    for (int k = 0; k < numOutputs; k++)
    {
        bias_w[k] = init_weight();
    }


    // STEP 1============================================
    // While stopping condition is false, do Steps 2-9.
    bool step_1 = false;
    int epoch = 0, max_epoch = 5000;

    while (step_1 == false)
    {
        // STEP 2============================================
        // For each training pair, do Steps 3-8.
        int array_of_target = 0;
        
        // Feedforward:
        // STEP 3============================================
        // Each input unit (Xi, i = 1, ... , n) receives
        // input signal Xi and broadcasts this signal to all
        // units in the layer above (the hidden units).
        for (size_t pola = 0; pola < 21; pola++)
        {
            
            // STEP 4============================================
            // Each hidden unit (Zj,j = 1, ... ,p) 
            for (size_t j = 0; j < numHiddenNodes; j++)
            {
                // sums its weighted input signals,
                float z_in = bias_v[j];
                for (size_t i = 0; i < 63; i++)
                {
                    z_in = z_in + (x[pola][i] * weight_v[i][j] );
                }

                // applies its activation function to compute its output signal
                z[j] = sigmoid(z_in);

            }
            
            // STEP 5============================================
            // Each output unit (Yk , k = 1, ... , m) sums its weighted input signals,
            for (size_t k = 0; k < numOutputs; k++)
            {
                // sums its weighted input signals,
                float y_in = bias_w[k];
                for (size_t j = 0; j < numHiddenNodes; j++)
                {
                    y_in = y_in + (z[j] * weight_w[j][k]);
                }

                // applies its activation function to compute its output signal
                y[k] = sigmoid(y_in);
            }

            // Log training
            // printf("\n========================\n\nPattern ke - %d (%c)\t \noutput: \n\t( ", pola+1, huruf[array_of_target]);
            // for (size_t k = 0; k < 7; k++)
            // {
            //     printf(" %f, ", y[k]);
            // }
            // printf(" )\n Expected Output: \n\t(");
            // for (size_t k = 0; k < 7; k++)
            // {
            //     printf("      %d,  ", target[array_of_target][k]);
            // }

            // Backpropagation of error:
            
            // STEP 6============================================
            // Each output unit (Yk , k = 1, ... ,m) receives
            // a target pattern corresponding to the input
            // training pattern, computes its error information term,
            float errorOutputs[numOutputs];
            for (int k=0; k<numOutputs; k++) {
                float error = (target[array_of_target][k] - y[k]);
                errorOutputs[k] = error * dSigmoid(y[k]);
            }

            // calculates its weight correction term (used to update Wjk later),
            float deltaWeightOutput[numHiddenNodes][numOutputs];
            float deltaBiasOutput[numOutputs];
            for (size_t k = 0; k < numOutputs; k++)
            {
                // calculates its weight correction term (used to update Wjk later)
                for (size_t j = 0; j < numHiddenNodes; j++)
                {
                    deltaWeightOutput[j][k] = learning_rate * errorOutputs[k] * z[j];
                }
                
                // calculates its bias correction term (used to update WOk later)
                deltaBiasOutput[k] = learning_rate * errorOutputs[k];
            }

            // STEP 7============================================
            // Each hidden unit (Zjo j = 1, ... ,p) sums its delta inputs (from units in the layer above)
            float errorHiddenUnits[numHiddenNodes];
            for (int j=0; j<numHiddenNodes; j++) {
                float errorHidden = 0;
                for (size_t k = 0; k < numOutputs; k++)
                {
                    errorHidden += (errorOutputs[k] * weight_w[j][k]);
                }

                // multiplies by the derivative of its activation function to calculate its error information term
                errorHiddenUnits[j] = errorHidden * dSigmoid(z[j]);  
            }

            float deltaWeightHiddenUnits[numInputs][numHiddenNodes];
            float deltaBiasHiddenUnits[numHiddenNodes];
            for (size_t j = 0; j < numHiddenNodes; j++)
            {
                // calculates its weight correction term (used to update vij later)
                for (size_t i = 0; i < 63; i++)
                {
                    deltaWeightHiddenUnits[i][j] = learning_rate * errorHiddenUnits[j] * x[pola][i];
                }
                
                deltaBiasHiddenUnits[numHiddenNodes] = learning_rate * errorHiddenUnits[j];
            }
            
            // Update weights and biases:
            // STEP 8============================================
            // Each output unit (Yk, k = I, , m) updates
            // its bias and weights (j = 0, , p):
            for (size_t k = 0; k < numOutputs; k++)
            {
                for (size_t j = 0; j < numHiddenNodes; j++)
                {
                    weight_w[j][k] += deltaWeightOutput[j][k]; 
                }
                bias_w[k] += deltaBiasOutput[k];
            }

            // Each hidden unit (Z], j == 1, ,p) updates
            // its bias and weights (i = 0, , n)
            for (size_t j = 0; j < numHiddenNodes; j++)
            {
                for (size_t i = 0; i < 63; i++)
                {
                    weight_v[i][j] += deltaWeightHiddenUnits[i][j];
                }
                bias_v[j] += deltaBiasHiddenUnits[j];     
            }
            
            array_of_target++;
            if (pola == 6 || pola == 13 || pola == 20)
            {
                array_of_target = 0;
            }

        }
        // STEP 9============================================
        // Test stopping condition.
        printf("\n\nTraining, Epoch %d", epoch);
        printf("\n==================", epoch);
        if (epoch == max_epoch){
            step_1 = true;
        }
        epoch++;
        
    }



    // ==============================================================================
    // Final Result
    FILE *fp;
    fp = fopen("data/data_wb_backpropagation.txt", "w");

    fprintf(fp, "#include <math.h>\n");
    fprintf(fp, "// function untuk aktivasi sigmoid\n");
    fprintf(fp, "float sigmoid(float x) { return 1 / (1 + exp(-x)); }\n");

    fprintf(fp, "// function untuk aktivasi bipolar\n");
    fprintf(fp, "int reluCostum(float x) { return (x < 0.5) ? -1.0 : 1.0;}\n\n");

    fprintf(fp, "//  number of input  = %d\n", numInputs);
    fprintf(fp, "static const int numInputs = %d;  \n\n", numInputs);  

    fprintf(fp, "//  number of hidden layer = %d\n", numHiddenNodes);
    fprintf(fp, "static const int numHiddenNodes = %d;\n", numHiddenNodes);    
    fprintf(fp, "float z[numHiddenNodes];\n\n"); 

    fprintf(fp, "//  number of output = %d\n", numOutputs);
    fprintf(fp, "static const int numOutputs = %d;\n",  numOutputs);
    fprintf(fp, "int y[numOutputs];\n\n"); 

    fprintf(fp, "static const float learning_rate = 0.1;\n\n");


    
    fprintf(fp, "//Weight dan bias yang didapat\n");
    fprintf(fp, "// weight hidden \n");
    fprintf(fp, "float weight_v[numInputs][numHiddenNodes] = \n");
    fprintf(fp, "{\n");
    for (size_t i = 0; i < numInputs; i++)
    {
        fprintf(fp, "\t{ ");
        for (size_t j = 0; j < numHiddenNodes; j++)
        {
            fprintf(fp, "\t%f,", weight_v[i][j]);
        }
        fprintf(fp, "\t},\n");
        
    }
    fprintf(fp, "};\n\n");
   
    fprintf(fp, "// bias hidden \n");
    fprintf(fp, "float bias_v[numHiddenNodes] = { ");
    for (size_t j = 0; j < numHiddenNodes; j++)
    {
        fprintf(fp, "\t%f,", bias_v[j]);
    }
    fprintf(fp, "\t};\n\n");


    fprintf(fp, "// weight output \n");
    fprintf(fp, "float weight_w[numHiddenNodes][numOutputs] = \n");
    fprintf(fp, "{\n");
    for (size_t j = 0; j < numHiddenNodes; j++)
    {
        fprintf(fp, "\t{ ");
        for (size_t k = 0; k < numOutputs; k++)
        {
            fprintf(fp, "\t%f,", weight_w[j][k]);
        }
        
        fprintf(fp, "\t},\n");
    }    
    fprintf(fp, "};\n\n");

    fprintf(fp, "// wbias output \n");
    fprintf(fp, "float bias_w[numOutputs] = { ");
    for (size_t k = 0; k < numOutputs; k++)
    {
        fprintf(fp, "\t%f,", bias_w[k]);
    }
    fprintf(fp, "\t};\n\n");
    
    return 0;
} 