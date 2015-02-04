package edu.stanford.rsl.conrad.filtering.rampfilters;

public class IRRFilter {
	// IIR filter design code based on a Pascal program
	// listed in "Digital Signal Processing with Computer Applications"
	// by P. Lynn and W. Fuerst (Prentice Hall)

	  public static final int LP = 1;
	  public static final int HP = 2;
	  public static final int BP = 3;

	  public static final int BUTTERWORTH = 4;
	  public static final int CHEBYSHEV = 5;

	  private int order;
	  private int prototype, filterType, freqPoints;
	  private float fp1, fp2, fN, ripple, rate;
	  private double[] pReal;
	  private double[] pImag;
	  private double[] z;
	  private double[] aCoeff;
	  private double[] bCoeff;

	  public void IIRFilter() {
	    // initial (default) settings
	    prototype = BUTTERWORTH;
	    filterType = LP;
	    order = 1;
	    ripple = 1.0f;
	    rate = 8000.0f;
	    fN = 0.5f*rate;
	    fp1 = 0.0f;
	    fp2 = 1000.0f;
	  }

	  public void setPrototype(String p) {
	    if (p.equals("Butterworth")) setPrototype(BUTTERWORTH);
	    if (p.equals("Chebyshev"))   setPrototype(CHEBYSHEV);
	  }

	  public void setPrototype(int p) {
	    prototype = p;
	  }

	  public int getPrototype() {
	    return prototype;
	  }

	  public void setFilterType(String ft) {
	    if (ft.equals("LP")) setFilterType(LP);
	    if (ft.equals("BP")) setFilterType(BP);
	    if (ft.equals("HP")) setFilterType(HP);
	  }

	  public int getFilterType() {
	    return filterType;
	  }

	  public void setFilterType(int ft) {
	    filterType = ft;
	    if ((filterType == BP) && odd(order)) order++;
	  }

	  public void setOrder(int n) {
	    order = Math.abs(n);
	    if ((filterType == BP) && odd(order)) order++;
	  }

	  public int getOrder() {
	    return order;
	  }

	  public void setRate(float r) {
	    rate = r;
	    fN = 0.5f * rate;
	  }

	  public float getRate() {
	    return rate;
	  }

	  public void setFreq1(float fp1) {
	    this.fp1 = fp1;
	  }

	  public float getFreq1() {
	    return fp1;
	  }

	  public void setFreq2(float fp2) {
	    this.fp2 = fp2;
	  }

	  public float getFreq2() {
	    return fp2;
	  }

	  public void setRipple(float r) {
	    ripple = r;
	  }

	  public float getRipple() {
	    return ripple;
	  }

	  public void setFreqPoints(int f) {
	    freqPoints = f;
	  }

	  public int getFreqPoints() {
	    return freqPoints;
	  }

	  public float getPReal(int i) {
	    // returns real part of filter pole with index i
	    return (float)pReal[i];
	  }

	  public float getPImag(int i) {
	    // returns imaginary part of filter pole with index i
	    return (float)pImag[i];
	  }

	  public float getZero(int i) {
	    // returns filter zero with index i
	    return (float)z[i];
	  }

	  public float getACoeff(int i) {
	    // returns IIR filter numerator coefficient with index i
	    return (float)aCoeff[i];
	  }

	  public float getBCoeff(int i) {
	    // returns IIR filter denominator coefficient with index i
	    return (float)bCoeff[i];
	  }

	  float sqr(float x) {
	    return x * x;
	  }

	  double sqr(double x) {
	    return x * x;
	  }

	  boolean odd(int n) {
	    // returns true if n odd
	    return (n % 2) != 0;
	  }

	  private void locatePolesAndZeros() {
	    // determines poles and zeros of IIR filter
	    // based on bilinear transform method
	    pReal  = new double[order + 1];
	    pImag  = new double[order + 1];
	    z      = new double[order + 1];
	    double ln10 = Math.log(10.0);
	    for(int k = 1; k <= order; k++) {
	      pReal[k] = 0.0;
	      pImag[k] = 0.0;
	    }
	    // Butterworth, Chebyshev parameters
	    int n = order;
	    if (filterType == BP) n = n/2;
	    int ir = n % 2;
	    int n1 = n + ir;
	    int n2 = (3*n + ir)/2 - 1;
	    double f1;
	    switch (filterType) {
	      case LP: f1 = fp2;       break;
	      case HP: f1 = fN - fp1;  break;
	      case BP: f1 = fp2 - fp1; break;
	      default: f1 = 0.0;
	    }
	    double tanw1 = Math.tan(0.5*Math.PI*f1/fN);
	    double tansqw1 = sqr(tanw1);
	    // Real and Imaginary parts of low-pass poles
	    double t, a = 1.0, r = 1.0, i = 1.0;
	    for (int k = n1; k <= n2; k++) {
	      t = 0.5*(2*k + 1 - ir)*Math.PI/(double)n;
	      switch (prototype) {
	        case BUTTERWORTH:
	          double b3 = 1.0 - 2.0*tanw1*Math.cos(t) + tansqw1;
	          r = (1.0 - tansqw1)/b3;
	          i = 2.0*tanw1*Math.sin(t)/b3;
	          break;
	        case CHEBYSHEV:
	          double d = 1.0 - Math.exp(-0.05*ripple*ln10);
	          double e = 1.0 / Math.sqrt(1.0 / sqr(1.0 - d) - 1.0);
	          double x = Math.pow(Math.sqrt(e*e + 1.0) + e, 1.0/(double)n);
	          a = 0.5*(x - 1.0/x);
	          double b = 0.5*(x + 1.0/x);
	          double c3 = a*tanw1*Math.cos(t);
	          double c4 = b*tanw1*Math.sin(t);
	          double c5 = sqr(1.0 - c3) + sqr(c4);
	          r = 2.0*(1.0 - c3)/c5 - 1.0;
	          i = 2.0*c4/c5;
	          break;
	      }
	      int m = 2*(n2 - k) + 1;
	      pReal[m + ir]     = r;
	      pImag[m + ir]     = Math.abs(i);
	      pReal[m + ir + 1] = r;
	      pImag[m + ir + 1] = - Math.abs(i);
	    }
	    if (odd(n)) {
	      if (prototype == BUTTERWORTH) r = (1.0 - tansqw1)/(1.0 + 2.0*tanw1+tansqw1);
	      if (prototype == CHEBYSHEV)   r = 2.0/(1.0 + a*tanw1) - 1.0;
	      pReal[1] = r;
	      pImag[1] = 0.0;
	    }
	    switch (filterType) {
	      case LP:
	        for (int m = 1; m <= n; m++)
	          z[m]= -1.0;
	        break;
	      case HP:
	        // low-pass to high-pass transformation
	        for (int m = 1; m <= n; m++) {
	          pReal[m] = -pReal[m];
	          z[m]     = 1.0;
	        }
	        break;
	      case BP:
	        // low-pass to bandpass transformation
	        for (int m = 1; m <= n; m++) {
	          z[m]  =  1.0;
	          z[m+n]= -1.0;
	        }
	        double f4 = 0.5*Math.PI*fp1/fN;
	        double f5 = 0.5*Math.PI*fp2/fN;
	        /*
	        check this bit ... needs value for gp to adjust critical freqs
	        if (prototype == BUTTERWORTH) {
	          f4 = f4/Math.exp(0.5*Math.log(gp)/n);
	          f5 = fN - (fN - f5)/Math.exp(0.5*Math.log(gp)/n);
	        }
	        */
	        double aa = Math.cos(f4 + f5)/Math.cos(f5 - f4);
	        double aR, aI, h1, h2, p1R, p2R, p1I, p2I;
	        for (int m1 = 0; m1 <= (order - 1)/2; m1++) {
	          int m = 1 + 2*m1;
	          aR = pReal[m];
	          aI = pImag[m];
	          if (Math.abs(aI) < 0.0001) {
	            h1 = 0.5*aa*(1.0 + aR);
	            h2 = sqr(h1) - aR;
	            if (h2 > 0.0) {
	              p1R = h1 + Math.sqrt(h2);
	              p2R = h1 - Math.sqrt(h2);
	              p1I = 0.0;
	              p2I = 0.0;
	            }
	            else {
	              p1R = h1;
	              p2R = h1;
	              p1I = Math.sqrt(Math.abs(h2));
	              p2I = -p1I;
	            }
	          }
	          else {
	            double fR = aa*0.5*(1.0 + aR);
	            double fI = aa*0.5*aI;
	            double gR = sqr(fR) - sqr(fI) - aR;
	            double gI = 2*fR*fI - aI;
	            double sR = Math.sqrt(0.5*Math.abs(gR + Math.sqrt(sqr(gR) + sqr(gI))));
	            double sI = gI/(2.0*sR);
	            p1R = fR + sR;
	            p1I = fI + sI;
	            p2R = fR - sR;
	            p2I = fI - sI;
	          }
	          pReal[m]   = p1R;
	          pReal[m+1] = p2R;
	          pImag[m]   = p1I;
	          pImag[m+1] = p2I;
	        } // end of m1 for-loop
	        if (odd(n)) {
	          pReal[2] = pReal[n+1];
	          pImag[2] = pImag[n+1];
	        }
	        for (int k = n; k >= 1; k--) {
	          int m = 2*k - 1;
	          pReal[m]   =   pReal[k];
	          pReal[m+1] =   pReal[k];
	          pImag[m]   =   Math.abs(pImag[k]);
	          pImag[m+1] = - Math.abs(pImag[k]);
	        }
	        break;
	      default:
	    }
	  }

	  public void design() {
	    aCoeff = new double[order + 1];
	    bCoeff = new double[order + 1];
	    double[] newA = new double[order + 1];
	    double[] newB = new double[order + 1];
	    locatePolesAndZeros(); // find filter poles and zeros
	    // compute filter coefficients from pole/zero values
	    aCoeff[0]= 1.0;
	    bCoeff[0]= 1.0;
	    for (int i = 1; i <= order; i++) {
	      aCoeff[i] = 0.0;
	      bCoeff[i] = 0.0;
	    }
	    int k = 0;
	    int n = order;
	    int pairs = n/2;
	    if (odd(order)) {
	     // first subfilter is first order
	      aCoeff[1] = - z[1];
	      bCoeff[1] = - pReal[1];
	      k = 1;
	    }
	    for (int p = 1; p <= pairs; p++) {
	      int m = 2*p - 1 + k;
	      double alpha1 = - (z[m] + z[m+1]);
	      double alpha2 = z[m]*z[m+1];
	      double beta1  = - 2.0*pReal[m];
	      double beta2  = sqr(pReal[m]) + sqr(pImag[m]);
	      newA[1] = aCoeff[1] + alpha1*aCoeff[0];
	      newB[1] = bCoeff[1] + beta1 *bCoeff[0];
	      for (int i = 2; i <= n; i++) {
	        newA[i] = aCoeff[i] + alpha1*aCoeff[i-1] + alpha2*aCoeff[i-2];
	        newB[i] = bCoeff[i] + beta1 *bCoeff[i-1] + beta2 *bCoeff[i-2];
	      }
	      for (int i = 1; i <= n; i++) {
	        aCoeff[i] = newA[i];
	        bCoeff[i] = newB[i];
	      }
	    }
	  }

	  public float[] filterGain() {

	    // filter gain at uniform frequency intervals
	    float[] g = new float[freqPoints+1];
	    double theta, s, c, sac, sas, sbc, sbs;
	    float gMax = -100.0f;
	    float sc = 10.0f/(float)Math.log(10.0f);
	    double t = Math.PI / freqPoints;
	    for (int i = 0; i <= freqPoints; i++) {
	      theta = i*t;
	      if (i == 0) theta = Math.PI*0.0001;
	      if (i == freqPoints) theta = Math.PI*0.9999;
	      sac = 0.0f;
	      sas = 0.0f;
	      sbc = 0.0f;
	      sbs = 0.0f;
	      for (int k = 0; k <= order; k++) {
	        c = Math.cos(k*theta);
	        s = Math.sin(k*theta);
	        sac += c*aCoeff[k];
	        sas += s*aCoeff[k];
	        sbc += c*bCoeff[k];
	        sbs += s*bCoeff[k];
	      }
	      g[i] = sc*(float)Math.log((sqr(sac) + sqr(sas))/(sqr(sbc) + sqr(sbs)));
	      gMax = Math.max(gMax, g[i]);
	    }
	    // normalise to 0 dB maximum gain
	    for (int i=0; i<=freqPoints; i++) g[i] -= gMax;
	    // normalise numerator (a) coefficients
	    float normFactor = (float)Math.pow(10.0, -0.05*gMax);
	    for (int i=0; i<=order; i++) aCoeff[i] *= normFactor;
	    return g;
	  }
}
/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
