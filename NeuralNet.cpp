#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
#include <math.h>

using namespace std;

//class for NN
class NeuralNetwork 
{
private:
    class Node 
    {
    public:
        double input;
        double activation;
        double biasWeight;
        vector<double> inputWeights;
    };
    int numInputNodes;
    vector<Node> inputLayer;
    int numHiddenNodes;
    vector<Node> hiddenLayer;
    int numOutputNodes;
    vector<Node> outputLayer;
    
public:
    NeuralNetwork(string inputFile);
    void train();
    void test();
    void printWeights(string outputFile);
    //sigmopid activation function
    double sigmoid(double value)
    {
        return (1 / (1 + exp(value * (-1))));
    }
    //derivative
    double sigmoidPrime(double value)
    {
        return (sigmoid(value) * (1 - sigmoid(value)));
    }
};

//open NN file and initialize
NeuralNetwork::NeuralNetwork(string inputFile)
{
    ifstream fin;
    fin.open(inputFile);
    fin >> numInputNodes >> numHiddenNodes >> numOutputNodes;
    Node emptyNode;
    double weight;
    for(int i = 0; i < numInputNodes; i++)
    {
        inputLayer.push_back(emptyNode);
    }
    //hidden layer
    for(int i = 0; i < numHiddenNodes; i++)
    {
        hiddenLayer.push_back(emptyNode);
        fin >> hiddenLayer.at(i).biasWeight;
        
        for(int j = 0; j < numInputNodes; j++)
        {
            fin >> weight;
            hiddenLayer.at(i).inputWeights.push_back(weight);
        }
    }
    //output layer
    for(int i = 0; i < numOutputNodes; i++)
    {
        outputLayer.push_back(emptyNode);
        fin >> outputLayer.at(i).biasWeight;
        for(int j = 0; j < numHiddenNodes; j++)
        {
            fin >> weight;
            outputLayer.at(i).inputWeights.push_back(weight);
        }
    }
    fin.close();
}
//function to train NN
void NeuralNetwork::train()
{
    string trainingFile;
    string outFile;
    int numEpochs;
    double learningRate;
    cout << "Please enter the name of the file containing the training set." << endl;
    cout << "Training Set File: ";
    cin >> trainingFile;
    cout << endl;
    ifstream fin;
    fin.open(trainingFile);
    while(!fin.is_open())
    {
        cout << "The file you have entered cannot be found. Enter a new file name." << endl;
        cout << "Training Set File: ";
        cin >> trainingFile;
        cout << endl;
        fin.open(trainingFile);
    }
    cout << "Please enter the name of the output file." << endl;
    cout << "Output File Name: ";
    cin >> outFile;
    cout << endl;
    cout << "Please enter the number of number of epochs." << endl;
    cout << "Number of Epochs: ";
    cin >> numEpochs;
    cout << endl;
    cout << "Please enter the learning rate." << endl;
    cout << "Learning rate: ";
    cin >> learningRate;
    cout << endl;
    int numExamples;
    fin >> numExamples;
    fin.ignore(256, '\n');
    vector<vector<vector<double>>> examples;
    vector<vector<double>> example;
    vector<double> eInputs;
    vector<double> eOutputs;
    double value;
    for(int i = 0; i < numExamples; i++)
    {
        example.clear();
        eInputs.clear();
        eOutputs.clear();
        
        for(int j = 0; j < numInputNodes; j++)
        {
            fin >> value;
            eInputs.push_back(value);
        }
        for(int j = 0; j < numOutputNodes; j++)
        {
            fin >> value;
            eOutputs.push_back(value);
        }
        example.push_back(eInputs);
        example.push_back(eOutputs);
        examples.push_back(example);
    }
    fin.close();
    for(int i = 0; i < numEpochs; i++)
    {
        for(int j = 0; j < numExamples; j++)
        {
            for(int k = 0; k < numInputNodes; k++)
            {
                inputLayer.at(k).input = inputLayer.at(k).activation = examples.at(j).at(0).at(k);
            }
            double sum;
            for(int k = 0; k < numHiddenNodes; k++)
            {
                sum = hiddenLayer.at(k).biasWeight * (-1);
                for(int l = 0; l < numInputNodes; l++)
                {
                    sum += hiddenLayer.at(k).inputWeights.at(l) * inputLayer.at(l).activation;
                }
                hiddenLayer.at(k).input = sum;
                hiddenLayer.at(k).activation = sigmoid(sum);
            }
            for(int k = 0; k < numOutputNodes; k++)
            {
                sum = outputLayer.at(k).biasWeight * (-1);
                for(int l = 0; l < numHiddenNodes; l++)
                {
                    sum += outputLayer.at(k).inputWeights.at(l) * hiddenLayer.at(l).activation;
                }
                outputLayer.at(k).input = sum;
                outputLayer.at(k).activation = sigmoid(sum);
            }
            
            vector<double> outputErrors;
            vector<double> hiddenErrors;
            for(int k = 0; k < numOutputNodes; k++)
            {
                value = sigmoidPrime(outputLayer.at(k).input)
                    * (examples.at(j).at(1).at(k) - outputLayer.at(k).activation);
                outputErrors.push_back(value);
            }
            for(int k = 0; k < numHiddenNodes; k++)
            {
                sum = 0;
                for(int l = 0; l < numOutputNodes; l++)
                {
                    sum += outputLayer.at(l).inputWeights.at(k) * outputErrors.at(l);
                }
                value = sigmoidPrime(hiddenLayer.at(k).input) * sum;
                hiddenErrors.push_back(value);
            }
            
            //update weight according to deltas
            for(int k = 0; k < numHiddenNodes; k++)
            {
                for(int l = 0; l < numInputNodes; l++)
                {
                    hiddenLayer.at(k).inputWeights.at(l) += (learningRate * inputLayer.at(l).activation * hiddenErrors.at(k));
                }
                hiddenLayer.at(k).biasWeight += (learningRate * (-1) * hiddenErrors.at(k));
            }
            for(int k = 0; k < numOutputNodes; k++)
            {
                for(int l = 0; l < numHiddenNodes; l++)
                {
                    outputLayer.at(k).inputWeights.at(l) += (learningRate * hiddenLayer.at(l).activation * outputErrors.at(k));
                }
                outputLayer.at(k).biasWeight += (learningRate * (-1) * outputErrors.at(k));
            }
        }
    }
    printWeights(outFile);
}

//function to test NN
void NeuralNetwork::test()
{
    string testFile;
    string outFile;
    cout << "Please enter the name of the file containing the test set." << endl;
    cout << "Test Set File: ";
    cin >> testFile;
    cout << endl;
    ifstream fin;
    fin.open(testFile);
    // Checks for valid file name
    while(!fin.is_open())
    {
        cout << "The file you have entered cannot be found. Enter a new file name." << endl;
        cout << "Testing Set File: ";
        cin >> testFile;
        cout << endl;
        fin.open(testFile);
    }
    cout << "Please enter the name of the output file." << endl;
    cout << "Output File Name: ";
    cin >> outFile;
    cout << endl;
    int numExamples;
    fin >> numExamples;
    fin.ignore(256, '\n');
    vector<vector<vector<double>>> examples;
    vector<vector<double>> example;
    vector<double> eInputs;
    vector<double> eOutputs;
    double value;
    for(int i = 0; i < numExamples; i++)
    {
        example.clear();
        eInputs.clear();
        eOutputs.clear();
        
        for(int j = 0; j < numInputNodes; j++)
        {
            fin >> value;
            eInputs.push_back(value);
        }
        for(int j = 0; j < numOutputNodes; j++)
        {
            fin >> value;
            eOutputs.push_back(value);
        }
        example.push_back(eInputs);
        example.push_back(eOutputs);
        examples.push_back(example);
    }
    fin.close();
    //confusion matrix, 4 element vector {A, B, C, D}
    vector<double> confusionMatrix = {0, 0, 0, 0};
    vector<vector<double>> confusionMatrices;
    // Adds a matrix for each output node
    for(int i = 0; i < numOutputNodes; i++)
    {
        confusionMatrices.push_back(confusionMatrix);
    }
    //global variables for microaveraging
    double globalA = 0, globalB = 0, globalC = 0, globalD = 0;
    for(int i = 0; i < numExamples; i++)
    {
        for(int j = 0; j < numInputNodes; j++)
        {
            inputLayer.at(j).input = inputLayer.at(j).activation = examples.at(i).at(0).at(j);
        }
        
        //propagate forward to hidden
        double sum;
        for(int j = 0; j < numHiddenNodes; j++)
        {
            sum = hiddenLayer.at(j).biasWeight * (-1);
            for(int k = 0; k < numInputNodes; k++)
            {
                sum += hiddenLayer.at(j).inputWeights.at(k) * inputLayer.at(k).activation;
            }
            hiddenLayer.at(j).input = sum;
            hiddenLayer.at(j).activation = sigmoid(sum);
        }
        //propagate forward to input
        for(int j = 0; j < numOutputNodes; j++)
        {
            sum = outputLayer.at(j).biasWeight * (-1);
            for(int k = 0; k < numHiddenNodes; k++)
            {
                sum += outputLayer.at(j).inputWeights.at(k) * hiddenLayer.at(k).activation;
            }
            outputLayer.at(j).input = sum;
            outputLayer.at(j).activation = sigmoid(sum);
            if(outputLayer.at(j).activation >= 0.5 && examples.at(i).at(1).at(j) == 1)
            {
                confusionMatrices.at(j).at(0)++;
                globalA++;
            } 
            else if (outputLayer.at(j).activation >= 0.5 && examples.at(i).at(1).at(j) == 0)
            {
                confusionMatrices.at(j).at(1)++;
                globalB++;
            } 
            else if(outputLayer.at(j).activation < 0.5 && examples.at(i).at(1).at(j) == 1)
            {
                confusionMatrices.at(j).at(2)++;
                globalC++;
            }
            else 
            {
                confusionMatrices.at(j).at(3)++;
                globalD++;
            }
        }
    }
    //metric calculations, output to output file
    ofstream fout;
    fout.open(outFile);
    double accuracy, precision, recall, f1;
    double macroAccuracy = 0,
            macroPrecision = 0,
            macroRecall = 0,
            macroF1 = 0;
    double A, B, C, D;
    for(int i = 0; i < numOutputNodes; i++)
    {
        A = confusionMatrices.at(i).at(0);
        B = confusionMatrices.at(i).at(1);
        C = confusionMatrices.at(i).at(2);
        D = confusionMatrices.at(i).at(3);
        
        accuracy = (A + D) / (A + B + C + D);
        precision = A / (A + B);
        recall = A / (A + C);
        f1 = (2 * precision * recall) / (precision + recall);
        
        fout << fixed << setprecision(0) << A << " " << B << " " << C << " " << D << " " << fixed << setprecision(3) << accuracy << " " << precision << " " << recall << " " << f1 << '\n';
        
        macroAccuracy += accuracy;
        macroPrecision += precision;
        macroRecall += recall;
    }
    macroAccuracy /= numOutputNodes;
    macroPrecision /= numOutputNodes;
    macroRecall /= numOutputNodes;
    macroF1 = (2 * macroPrecision * macroRecall) / (macroPrecision + macroRecall);   
    double microAccuracy = (globalA + globalD) / (globalA + globalB + globalC + globalD);
    double microPrecision = globalA / (globalA + globalB);
    double microRecall = globalA / (globalA + globalC);
    double microF1 = (2 * microPrecision * microRecall) / (microPrecision + microRecall);  
    fout << fixed << setprecision(3) << microAccuracy << " " << microPrecision << " " << microRecall << " " << microF1 << '\n';
    fout << fixed << setprecision(3) << macroAccuracy << " " << macroPrecision << " " << macroRecall << " " << macroF1 << '\n';  
    fout.close();
    
}

//function to display weights
void NeuralNetwork::printWeights(string outputFile)
{
    ofstream fout;
    fout.open(outputFile);
    fout << numInputNodes << " " << numHiddenNodes << " " << numOutputNodes << '\n';
    for(int i = 0; i < numHiddenNodes; i++)
    {
        fout << fixed << setprecision(3) << hiddenLayer.at(i).biasWeight;
        for(int j = 0; j < numInputNodes; j++)
        {
            fout << " " << fixed << setprecision(3) << hiddenLayer.at(i).inputWeights.at(j);
        }
        fout << '\n';
    }
    for(int i = 0; i < numOutputNodes; i++)
    {
        fout << fixed << setprecision(3) << outputLayer.at(i).biasWeight;
        for(int j = 0; j < numHiddenNodes; j++)
        {
            fout << " " << fixed << setprecision(3) << outputLayer.at(i).inputWeights.at(j);
        }
        fout << '\n';
    }
    fout.close();
}

//prompts user for testing or training, NN file
int main(int argc, const char * argv[]) 
{
    double choice;
    cout << "Would you like to [1]train or [2]test your neural network?" << endl;
    cout << "Please enter the number corresponding with your choice: ";
    cin >> choice;
    cout << endl;
    while(choice != 1 && choice != 2)
    {
        cout << "The number you entered was invalid. Please try again" << endl;
        cout << "Numbered Choice: ";
        cin >> choice;
        cout << endl;
    }
    string inputFile;
    cout << "Please enter the name of the file containing the initial neural network" << endl;
    cout << "Initial Neural Network File Name: ";
    cin >> inputFile;
    cout << endl;
    ifstream fin;
    fin.open(inputFile);
    while(!fin.is_open())
    {
        cout << "The file you have entered cannot be found. Please enter a new file name" << endl;
        cout << "Initial Neural Network File Name: ";
        cin >> inputFile;
        cout << endl;
        fin.open(inputFile);
    }
    fin.close();
    NeuralNetwork net = NeuralNetwork(inputFile);
    if(choice == 1)
    {
        net.train();
    } 
    else 
    {
        net.test();
    }
    
    return 0;
}