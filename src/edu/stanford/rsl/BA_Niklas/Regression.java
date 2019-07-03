package edu.stanford.rsl.BA_Niklas;

import org.apache.commons.math3.stat.regression.SimpleRegression;

public class Regression {

    public static void deg1(double[][] points) {

        // creating regression object, passing true to have intercept term
        SimpleRegression simpleRegression = new SimpleRegression(true);

        // passing data to the model
        // model will be fitted automatically by the class 
        simpleRegression.addData(points);

        // querying for model parameters
        System.out.println("slope = " + simpleRegression.getSlope());
        System.out.println("intercept = " + simpleRegression.getIntercept());

        // trying to run model for unknown data
        System.out.println("prediction for 1.5 = " + simpleRegression.predict(1.5));

    }

}