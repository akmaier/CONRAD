package edu.stanford.rsl.BA_Niklas;

import org.math.plot.utils.Array;

/*************************************************************************
 *  Compilation:  javac -cp .:jama.jar PolynomialRegression.java
 *  Execution:    java  -cp .:jama.jar PolynomialRegression
 *  Dependencies: jama.jar StdOut.java
 * 
 *  % java -cp .:jama.jar PolynomialRegression
 *  0.01 N^3 + -1.64 N^2 + 168.92 N + -2113.73 (R^2 = 0.997)
 *
 *************************************************************************/

import Jama.Matrix;
import Jama.QRDecomposition;


public class PolynomialRegression {
    private final int N;         // number of observations
    private final int degree;    // degree of the polynomial regression
    private final Matrix beta;   // the polynomial regression coefficients
    private double SSE;          // sum of squares due to error
    private double SST;          // total sum of squares

  /**
     * Performs a polynomial reggression on the data points <tt>(y[i], x[i])</tt>.
     * @param x the values of the predictor variable
     * @param y the corresponding values of the response variable
     * @param degree the degree of the polynomial to fit
     * @throws java.lang.IllegalArgumentException if the lengths of the two arrays are not equal
     */
    public PolynomialRegression(double[] x, double[] y, int degree) {
        this.degree = degree;
        N = x.length;

        // build Vandermonde matrix
        double[][] vandermonde = new double[N][degree+1];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j <= degree; j++) {
                vandermonde[i][j] = Math.pow(x[i], j);
            }
        }
        Matrix X = new Matrix(vandermonde);

        // create matrix from vector
        Matrix Y = new Matrix(y, N);

        // find least squares solution
        QRDecomposition qr = new QRDecomposition(X);
        beta = qr.solve(Y);


        // mean of y[] values
        double sum = 0.0;
        for (int i = 0; i < N; i++)
            sum += y[i];
        double mean = sum / N;

        // total variation to be accounted for
        for (int i = 0; i < N; i++) {
            double dev = y[i] - mean;
            SST += dev*dev;
        }

        // variation not accounted for
        Matrix residuals = X.times(beta).minus(Y);
        SSE = residuals.norm2() * residuals.norm2();
    }

   /**
     * Returns the <tt>j</tt>th regression coefficient
     * @return the <tt>j</tt>th regression coefficient
     */
    public double beta(int j) {
        return beta.get(j, 0);
    }

   /**
     * Returns the degree of the polynomial to fit
     * @return the degree of the polynomial to fit
     */
    public int degree() {
        return degree;
    }

   /**
     * Returns the coefficient of determination <em>R</em><sup>2</sup>.
     * @return the coefficient of determination <em>R</em><sup>2</sup>, which is a real number between 0 and 1
     */
    public double R2() {
        if (SST == 0.0) return 1.0;   // constant function
        return 1.0 - SSE/SST;
    }

   /**
     * Returns the expected response <tt>y</tt> given the value of the predictor
     *    variable <tt>x</tt>.
     * @param x the value of the predictor variable
     * @return the expected response <tt>y</tt> given the value of the predictor
     *    variable <tt>x</tt>
     */
    public double predict(double x) {
        // horner's method
        double y = 0.0;
        for (int j = degree; j >= 0; j--)
            y = beta(j) + (x * y);
        return y;
    }

   /**
     * Returns a string representation of the polynomial regression model.
     * @return a string representation of the polynomial regression model,
     *   including the best-fit polynomial and the coefficient of determination <em>R</em><sup>2</sup>
     */
    public String toString() {
        String s = "";
        int j = degree;

        // ignoring leading zero coefficients
        while (j >= 0 && Math.abs(beta(j)) < 1E-5)
            j--;

        // create remaining terms
        for (j = j; j >= 0; j--) {
            if      (j == 0) s += String.format("%.2f ", beta(j));
            else if (j == 1) s += String.format("%.2f N + ", beta(j));
            else             s += String.format("%.2f N^%d + ", beta(j), j);
        }
        return s + "  (R^2 = " + String.format("%.3f", R2()) + ")";
    }

    public static PolynomialRegression calc_regression(double[][] points, int deg) {
        double[] x = new double[points.length];
        double[] y = new double[points.length];
        for(int i = 0; i < points.length; i++){
        	x[i] = points[i][0];
        	y[i] = points[i][1];
        }
       
        PolynomialRegression regression = new PolynomialRegression(y, x, deg);
//        System.out.println(regression);
//        System.out.println(regression.beta(1));
        System.out.println("Polynomial Regression done");
        System.out.println(regression);
        return regression;
    }
}
