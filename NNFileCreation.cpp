#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
#include <math.h>
#include <time.h>
using namespace std;

int main()
{
    ofstream outfile;
	outfile.open("tictactoe.NN.txt");
    string r;
    int inputNode, hiddenNode, outputNode;
    cout << "How many input nodes: ";
    cin >> inputNode;
    cout << endl;
    cout << "How many hidden nodes: ";
    cin >> hiddenNode;
    cout << endl;
    cout << "How many output nodes: ";
    cin >> outputNode;
    cout << endl;
    outfile << to_string(inputNode) + " " + to_string(hiddenNode) + " " + to_string(outputNode) + " " + "\n";
    srand((unsigned)time(NULL));
    for (int j = 0; j < hiddenNode; j++)
    {
        for (int i = 0; i < inputNode+1; i++) 
        {
            r = to_string((float)rand()/RAND_MAX);
            r = r.substr(0,5);
            outfile << r + " ";
        }
    outfile << "\n";
    }
    for (int k = 0; k < outputNode; k++)
    {
        for (int a = 0; a < hiddenNode + 1; a++)
        {
            r = to_string((float)rand()/RAND_MAX);
            r = r.substr(0,5);
            outfile << r + " ";
        }
    outfile << "\n";
    }
    outfile.close();
    return 0;
}
