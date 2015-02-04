package edu.stanford.rsl.conrad.phantom;


//TODO: Use our own matrices instead of Jama.Matrix


/*
 * SheppLogan3D.java
 *
 * Created on January 7, 2007, 8:46 PM
 */

/**
 *
 * Three-dimensional Shepp-Logan Phantom in both the Fourier and image domains.
 *
 * This is a class called SheppLogan3D. It can be used to generate Fourier domain 
 * signal (or k-space) as well as image domain signal based on a 3-dimensional 
 * analytical phantom in both domains. Please refer to the source code or 
 * the article referenced below for further information.
 *
 * <br>
 *
 * 
 * <br>
 * Please refer to 
 * <br>
 * Koay CG, Sarlls JE, &#214zarslan E. 
 * Three Dimensional Analytical Magnetic Resonance Imaging Phantom in the Fourier Domain. Magn Reson Med. 58: 430-436 (2007)
 * <br> 
 * for further information.
 * 
 * <br>
 * @see <a href=http://dx.doi.org/10.1002/mrm.21292>Ref</a>
 * @author  Cheng Guan Koay
 * @since 07/25/2007
 *
 * @version &#8722&#8734.
 * 
 * 
 * Rewritten to be usable with JAMA.
 * 
 * @author Andreas Maier
 * @since 03/09/2010
 * 
 */
public class SheppLogan3D {
    
		int NE = 0; // number of ellipsoids
        
        double[][][] RT; //rotation matrices. T denotes matrix transpose.
     
        double[][] d; //displacement vectors.
        
        double[][] abc; //the length of the principal axes.
        
        double[] rho; // signal intensity.
        
    
    
    
    /** Creates a new instance of SheppLogan3D */
    public SheppLogan3D() {
    
        
        // number of ellipsoids
        NE = SheppLogan3D.ellipsoids.length;
        
        RT = new double[NE][3][3]; //transposed rotation matrices
     
        d = new double[NE][3]; //displacement vectors
        
        abc = new double[NE][3]; //The length of the principal axes
        
        rho = new double[NE]; // signal intensity
        
        for(int i=0; i<NE; i++){
         
        d[i][0] = ellipsoids[i][0]; // delta_x
        d[i][1] = ellipsoids[i][1]; // delta_y
        d[i][2] = ellipsoids[i][2]; // delta_z

        
        abc[i][0] = ellipsoids[i][3]; // a
        abc[i][1] = ellipsoids[i][4]; // b
        abc[i][2] = ellipsoids[i][5]; // c
        
        RT[i] = transpose(
                    dot(
                    dot( SheppLogan3D.Rz(ellipsoids[i][6]), 
                             SheppLogan3D.Ry(ellipsoids[i][7])
                           ),SheppLogan3D.Rz(ellipsoids[i][8])
                           )
                           );
         
        rho[i]  = ellipsoids[i][9];
        
        }//end for
        
        
    }

        
    private double [] [] transpose(double [] [] in){
    	return new Jama.Matrix(in).transpose().getArray();
    }
    
    private double [] [] dot(double [][] one, double [][] two){
    	return new Jama.Matrix(one).times(new Jama.Matrix(two)).getArray();
    }
    
    private double [] dot(double [][] one, double [] two){
    	double [] vector = new double [one.length];
    	for (int i= 0; i < one.length; i++){
    		double sum = 0;
    		for(int j=0;j < two.length; j++){
    			sum += one[i][j] * two[j];
    		}
    		vector[i]=sum;
    	}
    	return vector;
    }
    
    private double dot(double [] one, double [] two){
    	double sum = 0;
    	for(int j=0;j < two.length; j++){
    		sum += one[j] * two[j];
    	}
    	return sum;
    }
    
    
    /**
     *  User may add new ellipsoids and change their properties with this constructor.
     *
     *  @param ellipsoids is a two dimensional matrix arranged according to the following convention:
     *  <PRE> 
          delta_x, delta_y, delta_z,        a,       b,       c,            phi,  theta,  psi,     rho 
       {{       0,       0,       0,     0.69,    0.92,     0.9,              0,      0,    0,      2. },
        {       0,       0,       0,   0.6624,   0.874,    0.88,              0,      0,    0,    -0.8 },
        {   -0.22,      0.,   -0.25,     0.41,    0.16,    0.21, (3*Math.PI)/5.,      0,    0,    -0.2 },
        {    0.22,      0.,   -0.25,     0.31,    0.11,    0.22, (2*Math.PI)/5.,      0,    0,    -0.2 },
        {       0,    0.35,   -0.25,     0.21,    0.25,     0.5,              0,      0,    0,     0.2 },
        {       0,     0.1,   -0.25,    0.046,   0.046,   0.046,              0,      0,    0,     0.2 },
        {   -0.08,   -0.65,   -0.25,    0.046,   0.023,    0.02,              0,      0,    0,     0.1 },
        {    0.06,   -0.65,   -0.25,    0.046,   0.023,    0.02,              0,      0,    0,     0.1 },
        {    0.06,  -0.105,   0.625,    0.056,    0.04,     0.1,     Math.PI/2.,      0,    0,     0.2 },
        {      0.,     0.1,   0.625,    0.056,   0.056,     0.1,     Math.PI/2.,      0,    0,    -0.2 }};
       </PRE>
     *  <br>
     *
     *  Please refer to the paper mentioned above for further information on the notations.
     * 
     */
    public SheppLogan3D(double[][] ellipsoids) {
    
        
        // number of ellipsoids
        NE = ellipsoids.length;
        
        RT = new double[NE][3][3]; //transposed rotation matrices
     
        d = new double[NE][3]; //displacement vectors
        
        abc = new double[NE][3]; //The length of the principal axes
        
        rho = new double[NE]; // signal intensity
        
        for(int i=0; i<NE; i++){
         
        d[i][0] = ellipsoids[i][0]; // delta_x
        d[i][1] = ellipsoids[i][1]; // delta_y
        d[i][2] = ellipsoids[i][2]; // delta_z

        
        abc[i][0] = ellipsoids[i][3]; // a
        abc[i][1] = ellipsoids[i][4]; // b
        abc[i][2] = ellipsoids[i][5]; // c
        
        RT[i] = transpose(
                    dot(
                    dot( SheppLogan3D.Rz(ellipsoids[i][6]), 
                             SheppLogan3D.Ry(ellipsoids[i][7])
                           ),SheppLogan3D.Rz(ellipsoids[i][8])
                           )
                           );
         
        rho[i]  = ellipsoids[i][9];
        
        }//end for
        
        
        
        
        
    }
    
    
    
    
    
    
    /**
      *  Given a list of position vectors, i.e. {{x1,y1,z1},{x2,y2,z2},...}, 
      *  the image domain signals at those locations are returned.
      * 
      */    
     public double[] ImageDomainSignal(double[][] rList){

         int LEN      = rList.length;
         double[] s = new double[LEN];
      
         for(int i=0; i<LEN; i++){
         
             s[i]=ImageDomainSignal(rList[i][0],rList[i][1],rList[i][2]);
             
         }
      
         return s;
    }
    
     
    
     
    
    /**
     * returning real value of the image intensity at (x,y,z).
     *
     */
    public double ImageDomainSignal(double x, double y, double z){
    
        double[] r = {x,y,z};
        
        double signal = 0.0; 
        
        double[] p = new double[3];
        
        double sum = 0.0;
        
        for(int i=0; i<this.NE; i++){ // loop through each of the ellipsoids
        
                 
             p = dot(RT[i],new double[] {r[0]-d[i][0],r[1]-d[i][1],r[2]-d[i][2]});
            
             sum = Math.pow(p[0]/abc[i][0],2) + Math.pow(p[1]/abc[i][1],2) + Math.pow(p[2]/abc[i][2],2);
        
             signal += (sum<=1.0)?rho[i]:0;
        
        }
        
        
        return signal;
    }
    
    

     /**
      *  Given a list of (kx,ky,kz), the k-space signals at those locations are returned.
      *  The return array is of dimension kList.length by 2.
      *  The first column of the array is the real part of the complex signal and the second is 
      *  the imaginary part of the complex signal.
      */    
     public double[][] FourierDomainSignal(double[][] kList){

         int LEN      = kList.length;
         double[][] s = new double[LEN][2];
      
         for(int i=0; i<LEN; i++){
         
             s[i]=FourierDomainSignal(kList[i][0],kList[i][1],kList[i][2]);
             
         }
      
         return s;
    }
    
    protected double norm(double [] [] in){
    	return new Jama.Matrix(in).normF();
    }
    
    private double norm(double [] one){
    	double sum = 0;
    	for(int j=0;j < one.length; j++){
    		sum += one[j] * one[j];
    	}
    	return Math.sqrt(sum);
    }
    
    protected double [][] multiply(double [] [] one, double [] [] two){
    	double [] [] out = new double [one.length][one[0].length];
    	for (int i= 0; i < one.length; i++){
    		for(int j=0;j < one[0].length; j++){
    			out[i][j] = one[i][j] * two[i][j];
    		}
    	}
    	return out;
    }
    
    private double [] multiply(double []  one, double [] two){
    	double [] out = new double [one.length];
    	for (int i= 0; i < one.length; i++){
    		
    			out[i] = one[i] * two[i];
    		
    	}
    	return out;
    }
    
    /**
     * returning the complex signal evaluated at ( kx, ky, kz) in an array of length 2, i.e. {Re, Im}.
     */
    public double[] FourierDomainSignal(double kx, double ky, double kz){
    
        double[] k = {kx,ky,kz};
        
        double signal[] = new double[2]; // {Re, Im} , real and imaginary signals
        
        double K = 0.0;
        double arg = 0.0;
        
        for(int i=0; i<this.NE; i++){
        
             K = norm( multiply(dot(RT[i],k), abc[i]) );
            
             arg = 2.0 * Math.PI * K;
        
             if(K==0.0){ // if K = 0
             
                 if( norm(d[i])==0.0 ){ // if displacement vector is zero
                 
                     signal[0] +=(4./3.)*Math.PI* rho[i]*abc[i][0]*abc[i][1]*abc[i][2];
                 
                 }else{ // displacement vector is not zero
                     double kd = dot(k,d[i]);
                     double temp = (4./3.)*Math.PI* rho[i]*abc[i][0]*abc[i][1]*abc[i][2];
                     signal[0] += temp * Math.cos(2.0 * Math.PI * kd);
                     signal[1] -= temp * Math.sin(2.0 * Math.PI * kd);
                 }
                 
             }else if (K<=0.002){  // if K<=0.002
             
                 
                 if( norm(d[i])==0.0 ){ // if displacement vector is zero
                 
                     double temp = 4.1887902047863905 - 16.5366808961599*Math.pow(K,2) + 23.315785507450016*Math.pow(K,4);
                     signal[0] += rho[i]*abc[i][0]*abc[i][1]*abc[i][2]*temp;
                 
                 }else{  // if displacement vector is not zero
                     double kd = dot(k,d[i]);
                     double temp1 = 4.1887902047863905 - 16.5366808961599*Math.pow(K,2) + 23.315785507450016*Math.pow(K,4);
                     double temp2 = rho[i]*abc[i][0]*abc[i][1]*abc[i][2]*temp1;
                     
                     signal[0] += temp2 * Math.cos(2.0 * Math.PI * kd);
                     signal[1] -= temp2 * Math.sin(2.0 * Math.PI * kd);
                 }
             
                 
             }else{ // K>0.002
             
                 
             
                 
                 if( norm(d[i])==0.0 ){ // if displacement vector is zero
                 
                     double temp = Math.sin(arg)-arg*Math.cos(arg);
                            temp /= (2.0*Math.pow(Math.PI,2)*Math.pow(K,3));
                            
                     signal[0] += rho[i]*abc[i][0]*abc[i][1]*abc[i][2]*temp;
                 
                 }else{  // displacement vector is not zero
                     double kd = dot(k,d[i]);
                     double temp = Math.sin(arg)-arg*Math.cos(arg);
                            temp /= (2.0*Math.pow(Math.PI,2)*Math.pow(K,3));
                            
                            temp *= rho[i]*abc[i][0]*abc[i][1]*abc[i][2];
                 
                     signal[0] += temp * Math.cos(2.0 * Math.PI * kd);
                     signal[1] -= temp * Math.sin(2.0 * Math.PI * kd);
                 }
                 
                 
             }//end
             
             
             
        }
        
        
        return signal;
    }
    
    
    
    
    private static double[][] ellipsoids =
            // delta_x, delta_y, delta_z,        a,       b,       c,            phi,  theta,  psi,     rho 
            {{       0,       0,       0,     0.69,    0.92,     0.9,              0,      0,    0,      2. },
             {       0,       0,       0,   0.6624,   0.874,    0.88,              0,      0,    0,    -0.8 },
             {   -0.22,      0.,   -0.25,     0.41,    0.16,    0.21, (3*Math.PI)/5.,      0,    0,    -0.2 },
             {    0.22,      0.,   -0.25,     0.31,    0.11,    0.22, (2*Math.PI)/5.,      0,    0,    -0.2 },
             {       0,    0.35,   -0.25,     0.21,    0.25,     0.5,              0,      0,    0,     0.2 },
             {       0,     0.1,   -0.25,    0.046,   0.046,   0.046,              0,      0,    0,     0.2 },
             {   -0.08,   -0.65,   -0.25,    0.046,   0.023,    0.02,              0,      0,    0,     0.1 },
             {    0.06,   -0.65,   -0.25,    0.046,   0.023,    0.02,              0,      0,    0,     0.1 },
             {    0.06,  -0.105,   0.625,    0.056,    0.04,     0.1,     Math.PI/2.,      0,    0,     0.2 },
             {      0.,     0.1,   0.625,    0.056,   0.056,     0.1,     Math.PI/2.,      0,    0,    -0.2 }};
             
             
     
     protected static double[][] Rx(double t){

         return new double[][] {{1, 0, 0}, {0, Math.cos(t), -Math.sin(t)}, {0, Math.sin(t), Math.cos(t)}};
         
     }
     
     private static double[][] Ry(double t){

         return new double[][] {{Math.cos(t), 0, Math.sin(t)}, {0, 1, 0}, {-Math.sin(t), 0, Math.cos(t)}};
         
     }
     
     private static double[][] Rz(double t){

         return new double[][]{{Math.cos(t), -Math.sin(t), 0}, {Math.sin(t), Math.cos(t), 0}, {0, 0, 1}};
         
     }
     
     
     
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/