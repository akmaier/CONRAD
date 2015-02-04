package edu.stanford.rsl.conrad.utils;

import java.util.Random;


public abstract class StatisticsUtil {

	private static Random rand = new Random(System.currentTimeMillis());

	/**
	 * Generates a Poisson distributed random number. The Poisson distribution is definded by the number lambda. 
	 * The distribution has mean lambda and standard deviation of Math.sqrt(lambda).<BR>
	 * If lambda is smaller than 200 the value is determined by cumulative statistics, otherwise an approximation is used. Time measurements showed that the approximation used here outperforms the implementation in Matlab 2008.
	 * <br><BR>
	 * <img alt="poisson distribution" src="http://upload.wikimedia.org/wikipedia/commons/thumb/1/16/Poisson_pmf.svg/360px-Poisson_pmf.svg.png">
	 * @param lambda
	 * @return a random number drawn from the Poisson distribution
	 * @see #poissonRandomNumberSmall(double)
	 * @see #poissonRandomNumberBig(double)
	 */
	public static int poissonRandomNumber(double lambda){
		if (lambda < 200){
			return poissonRandomNumberSmall(lambda);
		} else {
			return poissonRandomNumberBig(lambda);
		}
	}
	
	/**
	 * Generates a Poisson distributed random number. The Poisson distribution is definded by the number lambda. 
	 * The distribution has mean lambda and standard deviation of Math.sqrt(lambda).<BR>
	 * Computation is based on cumulative statistics and gets <b>slow</b> with greater values of lambda.<BR><BR>
	 * <img alt="poisson distribution" src="http://upload.wikimedia.org/wikipedia/commons/thumb/1/16/Poisson_pmf.svg/360px-Poisson_pmf.svg.png">
	 * @param lambda
	 * @return a random number from the Poisson distribution
	 */
	public static int poissonRandomNumberSmall(double lambda)
	{
		int x = 0;
		double t = 0.0;
		while (true)
		{
			t -= Math.log(rand.nextDouble()) / lambda;
			if (t > 1.0)
			{
				break;
			}
			++x;
		}
		return x;
	}

	private static double stirlingConst =  Math.log(Math.sqrt(Math.PI*2));
	
	/**
	 * Computes ln(k!) using the Stirling approximation. <BR>(cf. <a href="http://portal.acm.org/citation.cfm?id=355997">J.H. Ahrens, U. Dieter. Computer Generation of Poission Deviates from Modified Normal Distributions. ACM Transactions on Mathematical Software (TOMS). 8(2):163-79. 1982.</a>)
	 * @param k the k
	 * @return ln(k!)
	 */
	public static double logfactorial(int k){
		double k3= k*k*k;
		double k5= k3*k*k;
		return (k+0.5) * Math.log(k) - k +  stirlingConst + (1.0 / (12*k)) - (1.0 / (360*k3)) + (1.0 / (1260*k5));
	}
	
	/**
	 * Generates a Poisson distributed random number. The Poisson distribution is definded by the number lambda. 
	 * The distribution has mean lambda and standard deviation of Math.sqrt(lambda).<BR>
	 * Uses the rejection algorithm after Atkinson (cf. <a href="http://www.jstor.org/stable/2346807">A.C. Atkinson. The Computer Generation of Poisson Random Variables. 
	 * Journal of the Royal Statistical Society. Series C (Applied Statistics)
	 * Vol. 28, No. 1 (1979), pp. 29-35</a>)
	 * <br><BR>
	 * 
	 * <img alt="poisson distribution" src="http://upload.wikimedia.org/wikipedia/commons/thumb/1/16/Poisson_pmf.svg/360px-Poisson_pmf.svg.png">
	 * @param lambda
	 * @return a random number from the Poisson distribution
	 */
	public static int poissonRandomNumberBig(double lambda){
		double beta = Math.PI * (1.0 / Math.sqrt(3.0 * lambda));
		double alpha = beta * lambda;
		double k = Math.log(0.8065) - lambda - Math.log(beta);
		double x = 0;
		//System.out.println(alpha + " " + beta + " " + k);
		int n = 0;
		while(true) {
			while (true){
				double u_one = rand.nextDouble();
				x = (alpha - Math.log((1.0-u_one)/u_one)) / beta;
				if (x > -.5) break;
			}
			n = (int)(x+0.5);
			double u_two = rand.nextDouble();
			if (alpha - (beta * x) + Math.log(u_two/Math.pow((1 + Math.exp(alpha-(beta * x))), 2)) <= k + n*Math.log(lambda) - logfactorial(n)){
				break;
			}
		}
		//System.out.println(n);
		return n;
	}


	public static void main(String args []){
		long start = System.currentTimeMillis();
		int number = 20000000;
		double [] array = new double [number];
		for (int i=0; i< number; i++){
			array[i] = poissonRandomNumber(75000);
		}
		double mean = DoubleArrayUtil.computeMean(array);
		double stddev = DoubleArrayUtil.computeStddev(array, mean);
		long end = System.currentTimeMillis();
		System.out.println(mean + " " + stddev + " " +(end - start) /1000 );
	}

}
/*
 * Copyright (C) 2010-2014  Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/