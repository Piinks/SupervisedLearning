/* Kate Lovett
* id3.cpp
* November 2017
* Supervised Learning Project for classifying irises.
* Command Line Arguments: Number of real-valued features in the data-set,
*     file name for training input,
*     file name for testing input.
* Compile with Makefile or: g++ -std=c++11 -O2 -o id3 id3.cpp
* This program will take the training data, build a decision tree for classifying
* data, and then test that tree with the testing data.
* Output will be the number of correct tests from the testing data set.
*/

#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <sstream>
#include <fstream>
#include <set>
#include <math.h>
using namespace std;

struct decisionNode{
	bool terminal;
	int classification, splitColumn;
	double splitVal;
	decisionNode *left;
	decisionNode *right;
};

int numClasses;
int numFeats;

vector<vector<int> > sort_attributes(vector<vector<double> > data);
double getInfo(int start, int end, int column, vector<vector<double> > data, vector<vector<int> > indices);
decisionNode* buildTree(vector<vector<double> > data);
void deleteNodes(decisionNode* &currentNode);

int main(int argc, char* argv[]) {

	vector<vector<double> > data;
	string line;
	double value;
	ifstream training, testing;
	set<int> classes;
	decisionNode *head;

	numFeats = atoi(argv[1]);
	training.open(argv[2]);
	testing.open(argv[3]);

	// Prepping and sorting of vectors provided by Dr. Phillips, sort.cpp
	getline(training,line);
	stringstream parsed(line);

	while (!parsed.eof()) {
		parsed >> value;
		data.push_back(vector<double>());
	}

	while (!training.eof()) {
		stringstream parsed(line);
		for (int i = 0; i < data.size(); i++) {
			parsed >> value;
			data[i].push_back(value);
			if(i == data.size()-1){
				classes.insert(value);
			}
		}
		getline(training,line);
	}
	// The classes set is intended to check for the # of unique classifications
	numClasses = classes.size();

	// Building the Decision Tree --------------------------------------
	head = buildTree(data);

	// Testing the Decision Tree ---------------------------------------

	double testData[numFeats];
	int testClass, correctTests = 0;
	decisionNode* currentNode;

	// Read Initial Data Point
	for(int i = 0; i < numFeats; i ++){
		testing >> testData[i];
	}
	testing >> testClass;

	while(!testing.eof()){

		currentNode = head;
		bool endSearch = false;
		while(!endSearch){
			// Successful Classification
			if(currentNode->terminal && (currentNode->classification == testClass)){
				correctTests += 1;
				endSearch = true;
			}
			else if(testData[currentNode->splitColumn] < currentNode->splitVal && currentNode->left != NULL){
				currentNode = currentNode->left;
			}
			else if(currentNode->right != NULL){
				currentNode = currentNode->right;
			}
			// Failed classification
			else{
				endSearch = true;
			}
		}
		// Read Next Data Point
		for(int i = 0; i < numFeats; i ++){
			testing >> testData[i];
		}
		testing >> testClass;
	}

	// Output Correct # of Tests
	cout << correctTests << endl;

	// Close-up & Clean-up
	training.close();
	testing.close();
	deleteNodes(head);

	return 0;
}

/* Function: sort_attributes
 * Funtion provided by Dr. Phillips for purpose of sorting data.
 * Instead of re-arranging the data, a 2-D array of sorted indices
 * is returned so the data can be accessed in order by any given column
 */
vector<vector<int> > sort_attributes(vector<vector<double> > data) {
	vector<vector<int> > indices;
	vector<double> *ptr;
	indices.resize(data.size());
	for (int x = 0; x < indices.size(); x++) {
		indices[x].resize(data[x].size());
		iota(indices[x].begin(),indices[x].end(),0);
		ptr = &(data[x]);
		sort(indices[x].begin(),indices[x].end(),
		[&](size_t i, size_t j){ return (*ptr)[i] < (*ptr)[j]; });
	}
	return indices;
}

/* Function: getInfo
 * This function serves TWO purposes.
 *
 * (1) The first is to calculate the given info for the data set I(x)
 *  which is the summation of -p(x)log2(p(x)), for every class x.
 *  When the function is called in this fashion, start is always zero and
 *  end is the size of the data set.
 *  The column # does not matter.
 *
 * (2) The second is to calculate a portion of E, I(x|y)
 *  which is the summation of -p(x|y)log2(p(x|y)), for every class x.
 *  if E is being calculated, p(y)(I(x|y)) will be handled in buildTree.
 *  When the function is called for this purpose, start and end reference
 *  a section of data that is being evaluated for potential split.
 *  In this case, the column number does matter, as the potential split
 *  is evaluating columns individuals in addition to start/end indexes.
 */
double getInfo(int start, int end, int column, vector<vector<double> > data, vector<vector<int> > indices){
	double p, sEntropy, info = 0;
	int sum, range = end - start;

	// Loop i checks distribution for each class
	for(int i = 0; i < numClasses; i++){
		sum = 0;
		for(int j = start; j < end; j++){
			if (data[numFeats][indices[column][j]] == i){
				sum += 1;
			}
		}
		// p(i) = # class(i) / total data points
		p = (double)sum / (double)range;


		// Catch for 0log2(0)
		if(p != 0){
			sEntropy = -1*p*log2(p);
		}
		else{
			sEntropy = 0;
		}
		info += sEntropy;
	}
	return info;
}

/* Function: buildTree
 * This function builds the decision tree!
 * Upon reciving the current data set, it will sort the attricutes by index,
 * and then evaluate the data to find the best way to divide the data for
 * the greatest gain of information.
 * Once the best gain has been found, the data is split, a tree node is generated,
 * and the function is called again for both sets of partitioned data.
 * When a terminal node has been reached, it will return the recursive calls.
 */
decisionNode* buildTree(vector<vector<double>> data){

	vector<vector<int> > indices;
	int start = 0;
	int end = data[0].size();
	indices = sort_attributes(data);

	double info = getInfo(start, end, 0, data, indices);

	// When information returns 0, this indicates a terminal node has been reached
	if(!info){
		decisionNode *newNode = new decisionNode;
		newNode->terminal = true;
		newNode->classification = data[numFeats][indices[0][0]];
		newNode->splitColumn = -1;
		newNode->splitVal = -1;
		newNode->left = NULL;
		newNode->right = NULL;
		// This condition ends the recursive call and returns the terminal node
		return newNode;
	}

	// If this function does not return as a result of finding a terminal node,
	// begin looking for potential splits
	double splitAvg, E, minEval = info;
	int split, splitCol;

	for(int k = 0; k < numFeats; k++){
		for(int i = start; i < end-1; i++){
			// Detection of potential split
			if(data[k][indices[k][i]] != data[k][indices[k][i+1]]){

				E = ((double)(i+1)/(double)end)*getInfo(start, i+1, k, data, indices) + ((double)(end-i+1)/(double)end)*getInfo(i+1, end, k, data, indices);

				// I chose to accept the lowest E value as the indicator of best split.
				// Since Gain = I - E, the lowest value E would yield the greatest gain.
				if(E < minEval){
					minEval = E;
					split = i;
					splitAvg = (data[k][indices[k][i]] + data[k][indices[k][i+1]]) / 2;
					splitCol = k;
				}
			}
		}
	}
	// Now that the best split has been determined, the data is divided up.
	vector<vector<double> > leftData;
	vector<vector<double> > rightData;
	leftData.resize(numFeats+1);
	rightData.resize(numFeats+1);

	for(int i = 0; i < end; i++){
		for(int j = 0; j <  numFeats+1; j++){
			if(data[splitCol][indices[splitCol][i]] < splitAvg){
				leftData[j].push_back(data[j][indices[splitCol][i]]);
			}
			else{
				rightData[j].push_back(data[j][indices[splitCol][i]]);
			}
		}
	}

	// Create new node in the decision tree
	decisionNode *newNode = new decisionNode;
	newNode->terminal = false;
	newNode->classification = -1;
	newNode->splitColumn = splitCol;
	newNode->splitVal = splitAvg;

	// Recursive call continues to build the tree
	newNode->left = buildTree(leftData);
	newNode->right = buildTree(rightData);

	return newNode;
}

/* Function: deleteNodes
 * Cleaning up memory allocation before program exits.
 */
void deleteNodes(decisionNode* &currentNode){
	if(currentNode == NULL){
		return;
	}

	deleteNodes(currentNode->left);
	deleteNodes(currentNode->right);

	delete currentNode;
	currentNode = NULL;
}
