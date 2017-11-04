//
// Created by yingweiy on 10/21/17.
//
#include <iostream>
#include "Dense"
#include <vector>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

MatrixXd CalculateJacobian(const VectorXd& x_state);

int main() {

    /*
     * Compute the Jacobian Matrix
     */

    //predicted state  example
    //px = 1, py = 2, vx = 0.2, vy = 0.4
    VectorXd x_predicted(4);
    x_predicted << 1, 2, 0.2, 0.4;

    MatrixXd Hj = CalculateJacobian(x_predicted);

    cout << "Hj:" << endl << Hj << endl;

    return 0;
}

MatrixXd CalculateJacobian(const VectorXd& x_state) {

    MatrixXd Hj(3,4);
    //recover state parameters
    float px = x_state(0);
    float py = x_state(1);
    float vx = x_state(2);
    float vy = x_state(3);

    //TODO: YOUR CODE HERE

    //check division by zero
    float pxy = px*px + py*py;
    float spxy = sqrt(pxy);

    if (pxy<0.000001) {
        cout << "Dividing by zero in Jacobian calculation!" << endl;
        return Hj;
    }

    //compute the Jacobian matrix

    Hj << px/spxy, py/spxy, 0, 0,
            -py/pxy, px/pxy, 0, 0,
            py*(vx*py - vy*px)/pow(pxy, 1.5), px*(vy*px - vx*py)/pow(pxy, 1.5), px/spxy, py/spxy;


    return Hj;
}
