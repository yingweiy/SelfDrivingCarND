//
// Created by yingweiy on 10/21/17.
//

#include <iostream>
#include "Dense"
#include <vector>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

VectorXd CalculateRMSE(const vector<VectorXd> &estimations,
                       const vector<VectorXd> &ground_truth);

int main() {
    /*
     * Compute RMSE
     */
    vector<VectorXd> estimations;
    vector<VectorXd> ground_truth;

    //the input list of estimations
    VectorXd e(4);
    e << 1, 1, 0.2, 0.1;
    estimations.push_back(e);
    e << 2, 2, 0.3, 0.2;
    estimations.push_back(e);
    e << 3, 3, 0.4, 0.3;
    estimations.push_back(e);

    //the corresponding list of ground truth values
    VectorXd g(4);
    g << 1.1, 1.1, 0.3, 0.2;
    ground_truth.push_back(g);
    g << 2.1, 2.1, 0.4, 0.3;
    ground_truth.push_back(g);
    g << 3.1, 3.1, 0.5, 0.4;
    ground_truth.push_back(g);

    //call the CalculateRMSE and print out the result
    cout << CalculateRMSE(estimations, ground_truth) << endl;


    return 0;
}

VectorXd CalculateRMSE(const vector<VectorXd> &estimations,
                       const vector<VectorXd> &ground_truth){

    VectorXd rmse(4);
    rmse << 0,0,0,0;

    //accumulate squared residuals
    for(int i=0; i < estimations.size(); ++i){
        // ... your code here
        VectorXd err = estimations[i] - ground_truth[i];
        err = err.array() * err.array();
        rmse += err;
    }

    //calculate the mean
    // ... your code here
    rmse = rmse / estimations.size();
    //calculate the squared root
    // ... your code here
    rmse = rmse.array().sqrt();
    //return the result
    return rmse;
}
