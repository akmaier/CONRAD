package edu.stanford.rsl.conrad.fitting;


public class LogarithmicFunction extends Function {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3584939480038216387L;
	private double a;
	private double b;
	private double c;
	private double d;


	public LogarithmicFunction(){
		numberOfParameters = 4;
	}

	private static double bFunction(double b, double x1, double x2, double x3, double y1, double y2, double y3){
		double z = (y1-y2) / (y1-y3);
		return (Math.pow(x1 - b, z-1) * (x2 - b)) - Math.pow(x3 - b, z);
	}

	private static double estimateA(double b, double x1, double x2, double y1, double y2){
		double nominator = (y1 - y2);
		double denominator = Math.log((x1 - b) / (x2 -b));
		return nominator / denominator;
	}

	private static double estimateC(double b, double x1, double x2, double y1, double y2){
		double z = y2 / y1;
		double nominator = x2 - b;
		double denominator = Math.pow(x1 - b, z);
		return Math.pow(nominator / denominator, 1 / (1 -z));
	}

	private static double estimateD(double a, double b, double c, double x, double y){
		return y - (a * Math.log((x - b)/ c));
	}

	@Override
	public double evaluate(double x){
		return (a * Math.log((x - b) / Math.abs(c))) + d;
	}

	private static double estimateB(double x1, double x2, double x3, double y1, double y2, double y3){
		int scale = 20000;
		double bestB = 0;
		double minDist = Double.MAX_VALUE;
		for (int i=0;i<scale;i++){
			double x = ((i + 0.0) / scale * 10) - 5;
			double b = bFunction(x, x1, x2, x3, y1, y2, y3);
			if(Math.abs(b) < minDist){
				minDist = Math.abs(b);
				bestB = x;
			}
		}
		return bestB;
	}

	public String toString(){
		if (fittingDone){
			return "y = (" + a + " * log((x - " +
					b + ") / |" +
					c + "|)) + " +
					d;
		} else {
			return "y = (a * log((x - b) / c)) + d";
		}
	}

	public void fitToThreePoints(double x1, double x2, double x3, double y1, double y2, double y3){
		b = estimateB(x1, x2, x3, y1, y2, y3);
		a = estimateA(b, x1, x2, y1, y2);
		c = estimateC(b, x1, x2, y1, y2);
		d = estimateD(a, b, c, x1, y1);
		fittingDone = true;
	}

	protected boolean isValidEstimate(){
		boolean revan = true;
		if (Double.isInfinite(a) || Double.isNaN(a)) revan = false;
		if (Double.isInfinite(b) || Double.isNaN(b)) revan = false;
		if (Double.isInfinite(c) || Double.isNaN(c)) revan = false;
		if (Double.isInfinite(d) || Double.isNaN(d)) revan = false;
		return revan;
	}

	public double getA() {
		return a;
	}

	public void setA(double a) {
		this.a = a;
	}

	public double getB() {
		return b;
	}

	public void setB(double b) {
		this.b = b;
	}

	public double getC() {
		return c;
	}

	public void setC(double c) {
		this.c = c;
	}

	public double getD() {
		return d;
	}

	public void setD(double d) {
		this.d = d;
		fittingDone = true;
	}

	@Override
	protected double [] getInitialX(){
		return new double []{1,1,1,0};
	}


	@Override
	public void fitToPoints(double[] x, double [] y) {
		if (x.length == 3) {
			fitToThreePoints(x[0], x[1], x[2], y[0], y[1], y[2]);
		} else {
			super.fitToPoints(x, y);
		}
	}

	@Override
	public int getMinimumNumberOfCorrespondences() {
		return 3;
	}

	@Override
	public double[] getParametersAsDoubleArray() {
		return new double[]{a,b,c,d};
	}

	public static void main(String [] args){
		LogarithmicFunction func = new LogarithmicFunction();
		double [] x = {10000, 50000, 100000, 2000000, 600000000};
		double [] y = {10, 20, 30, 500, 3000};
		func.fitToPoints(x, y);
		System.out.println(func 
				+ "\n" + func.evaluate(10000)
				+ "\n" + func.evaluate(50000)
				+ "\n" + func.evaluate(2000000)
				+ "\n" + func.evaluate(600000000)
				+ "\n" + func.evaluate(83000000*1000));
	}

	@Override
	public void setParametersFromDoubleArray(double[] param) {
		this.a = param[0];
		this.b = param[1];
		this.c = param[2];
		this.d = param[3];
	}

}
