package edu.stanford.rsl.conrad.physics;

import java.text.NumberFormat;

import java.util.Arrays;

import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.DoubleArrayUtil;
import edu.stanford.rsl.conrad.utils.LinearInterpolatingDoubleArray;
import static edu.stanford.rsl.conrad.physics.Constants.*;
import static edu.stanford.rsl.conrad.utils.DoubleArrayUtil.*;

public class XRaySpectrum {

	/**
	 * Creates a spectrum that is similar to a C-arm spectrum with default parameters.
	 * @param E the energies in [keV]
	 * @param kVp the peak voltage in [kV]
	 * @param target the target material ("W" or "Mo")
	 * @param mAs the acceleration time times the current 
	 * @return the photon flux in [photons/mm2/bin] in an array matching the energies E
	 * @throws Exception
	 */
	public static double [] generateXRaySpectrum(double [] E, double kVp, String target, double mAs) throws Exception{
		double degrees, mmpyrex, mmoil, mmlexan, mmAl, mdis;
		if (target.equals("W")) {
				degrees = 10;
				mmpyrex = 2.38;
				mmoil = 3.06;
				mmlexan = 2.66;
				mmAl = 1.50;
				mdis = 1;
		}else if (target.equals("Mo")) {
				degrees = 13;
				mmpyrex = 1.11;
				mmoil = 3.06;
				mmlexan = 2.42;
				mmAl = 0;
				mdis = 1;
		} else {
			throw new Exception("Undefined target material");
		}
		return generateXRaySpectrum(E, kVp, target, mAs, mdis, degrees, mmpyrex, mmoil, mmlexan, mmAl);
	}
	
	/**
	 * 
	 * @param E the energies in [keV]
	 * @param kVp the peak voltage in [kV]
	 * @param target the target material ("W" or "Mo")
	 * @param mAs the acceleration time times the current 
	 * @param mdis amount of air in [m]
	 * @param degrees tube angle in [deg]
	 * @param mmpyrex amount of pyrex filtration in [mm]
	 * @param mmoil amount of oil filtration in [mm]
	 * @param mmlexan amount of lexan filtration in [mm]
	 * @param mmAl amount of Al filtration in [mm]
	 * @return the photon flux in [photons/mm2/bin] in an array matching the energies E
	 * @throws Exception
	 */
	public static double [] generateXRaySpectrum(double [] E, double kVp, String target, double mAs, double mdis, double degrees, double mmpyrex, double mmoil, double mmlexan, double mmAl) throws Exception{
		double Emin, Emax;
		double Emin_W	=	10.0;		/* lower keV cutoff for W spectra */
		double Emin_Mo	=	3.0;		/* lower keV cutoff for Mo spectra */
		double Emax_W	=	301.;		/* upper keV cutoff for W spectra */
		double Emax_Mo	=	101.;		/* upper keV cutoff for Mo spectra */
		if (target.equals("W")) {
			Emin = Emin_W;
			Emax = Emax_W;
		}else if (target.equals("Mo")) {
			Emin = Emin_Mo;
			Emax = Emax_Mo;
		} else {
			throw new Exception("Undefined target material");
		}
		double radians = degrees / 180.0 * Math.PI;
		double dE = E[1]-E[0];
		if (kVp+dE > Emax)
			throw new Exception("Desired kVp greater than Emax");
		if (E[0]<Emin)
			throw new Exception("E contains element less than Emin for this target material");
		double [] phi = new double [E.length];
		NumberFormat nf = NumberFormat.getInstance();
		nf.setMaximumFractionDigits(10);
		nf.setMaximumIntegerDigits(10);
		DoubleArrayUtil.add(phi, characteristic(E, target, kVp, radians));
		
		DoubleArrayUtil.add(phi, bremsstrahlung(E, target, kVp, radians));
		
		/* Calculated spectrum transmitted through the inherent filter materials */
		double [] inherentFilter = DoubleArrayUtil.multiply(mu(E, "pyrex"), PYREX_mdensity*mmpyrex/10.0);
		DoubleArrayUtil.add(inherentFilter, DoubleArrayUtil.multiply(mu(E, "lexan") , LEXAN_mdensity*mmlexan/10.0));
		
		DoubleArrayUtil.add(inherentFilter, DoubleArrayUtil.multiply(mu(E, "Al") , Al_mdensity*mmAl/10.0));
		DoubleArrayUtil.add(inherentFilter, DoubleArrayUtil.multiply(mu(E, "air"), AIR_mdensity*mdis*100.0));
		
		DoubleArrayUtil.multiply(inherentFilter, -1);
		DoubleArrayUtil.exp(inherentFilter);
		DoubleArrayUtil.multiply(phi, inherentFilter);
		
		/* Now scale to photons/mm2/bin.*/
		double f = mAs * EpMAS / (4.0 * Math.PI * mdis * mdis * 1.0E6);
		DoubleArrayUtil.multiply(phi, f);
		return phi;
	}

	/**
	 * Returns the mass attenuation coef for the selected material at
	 * energy E.  The coefs are fit to Howerton data by N. Cardinal,
	 * and are thought to be significantly better than the coefs used by Tucker.
	 * Except for Re, which gives nonsense results, so use Tucker for Re
	 * @param E the array of energies
	 * @param material the material ("W", "Re", "Mo", "Al", "oil", "lexan", "pyrex", or "air")
	 * @return the array of absorption coefficients.
	 * @throws Exception may occur
	 */
	public static double [] mu(double [] E, String material) throws Exception{
		double [] uT = Arrays.copyOf(E, E.length);
		DoubleArrayUtil.divide(uT, 100);
		double [] mu = new double[E.length];
		int nEbin = E.length;
		double [] a = null;
		for (int j=0; j<nEbin; j++){
			mu[j]=0;
			if (material.equals("Re")){
				if (E[j]>= Re_K){
					a = new double [] {-5.803e-1, 3.336e0, 3.292e0, 1.502e0, 0};
				} else {
					a= new double [] {-2.987e-1, 5.815e-1, 6.230e-1, 7.727e-2, -6.155e-3};
				}
				mu[j] += a[0];
				mu[j] += a[1] / Math.pow(uT[j], 1.6);
				mu[j] += a[2] / Math.pow(uT[j], 2.7);
				mu[j] += a[3] / Math.pow(uT[j], 3.5);
				mu[j] += a[4] / Math.pow(uT[j], 4.5);
			} else {
				if (material.equals("W")){
					if (E[j]> W_K){
						a = aWh;
					} else {
						a = aWl;
					}
				} else if (material.equals("Re")){
					if (E[j]> Re_K){
						a = aReh;
					} else {
						a = aRel;
					}
				} else if (material.equals("Mo")){
					if (E[j]> Mo_K){
						a = aMoh;
					} else {
						a = aMol;
					}
				} else if (material.equals("Al")){
					a = aAl;
				} else if (material.equals("oil")){
					a = aOIL;
				} else if (material.equals("lexan")){
					a = aLEX;
				} else if (material.equals("pyrex")){
					a = aPYR;
				} else if (material.equals("air")){
					a = aAIR;
				} else {
					throw new Exception("Undefined material in mu()");
				}
				
				double u = Math.pow(100.0/E[j], 0.5);
				mu[j] = a[0] * u;
				
				for(int n = 1; n < a.length; n++){
					mu[j] +=  (a[n]) * Math.pow(u, n+1);
					
				}
				//print("a", a);
				//System.out.println(j+ " " + mu[j]);
			}
		}
		
		return mu;
	}

	/**
	 * Generates characteristic lines and adds them to the fluence spectrum.  The
	 * characteristic peaks are added to the nearest two bins, weighted by the
	 * position of the peak between the bins
	 * @param E the array of energies
	 * @param tubeTarget the target material
	 * @param To Peak kilo volage
	 * @param theta angle
	 * @return the array of characteristics.
	 * @throws Exception may occur
	 */
	public static double [] characteristic (double [] E, String tubeTarget, double To, double theta) throws Exception{
		double [] phi = new double [E.length];
		if (tubeTarget.equals("W")){
			if (To > Re_K) { /* add Rhenium characteristic lines*/
				DoubleArrayUtil.add(phi, characteristicLine(E, tubeTarget, To, Re_K, Re_Ka1_f, Re_Ka1, theta));
				DoubleArrayUtil.add(phi, characteristicLine(E, tubeTarget, To, Re_K, Re_Ka2_f, Re_Ka2, theta));
				DoubleArrayUtil.add(phi, characteristicLine(E, tubeTarget, To, Re_K, Re_Kb1_f, Re_Kb1, theta));
				DoubleArrayUtil.add(phi, characteristicLine(E, tubeTarget, To, Re_K, Re_Kb2_f, Re_Kb2, theta));
			}
			if (To > W_K) { /* Tungsten characteristic lines */
				DoubleArrayUtil.add(phi, characteristicLine(E, tubeTarget, To, W_K, W_Ka1_f, W_Ka1, theta));
				DoubleArrayUtil.add(phi, characteristicLine(E, tubeTarget, To, W_K, W_Ka2_f, W_Ka2, theta));
				DoubleArrayUtil.add(phi, characteristicLine(E, tubeTarget, To, W_K, W_Kb1_f, W_Kb1, theta));
				DoubleArrayUtil.add(phi, characteristicLine(E, tubeTarget, To, W_K, W_Kb2_f, W_Kb2, theta));
			}
		} else if (tubeTarget.equals("Mo")){ /* Molybdenum characteristic line */
			if (To > Mo_K){
				DoubleArrayUtil.add(phi, characteristicLine(E, tubeTarget, To, Mo_K, Mo_Ka1_f, Mo_Ka1, theta));
				DoubleArrayUtil.add(phi, characteristicLine(E, tubeTarget, To, Mo_K, Mo_Ka2_f, Mo_Ka2, theta));
				DoubleArrayUtil.add(phi, characteristicLine(E, tubeTarget, To, Mo_K, Mo_Kb1_f, Mo_Kb1, theta));
				DoubleArrayUtil.add(phi, characteristicLine(E, tubeTarget, To, Mo_K, Mo_Kb2_f, Mo_Kb2, theta));
			}
		} else {
			throw new Exception("Illegal target material in characteristic()");
		}
		return phi;
	}

	/** 
	 * Return the vector spectrum phi that contains only the specified 
	 * characteristic line with the appropriate split between the
	 * nearest two bins.
	 * 
	 * @param E the energies
	 * @param tubeTarget the material
	 * @param To Peak kilo volage
	 * @param Ek K-edge
	 * @param char_line_f the flurescent yield
	 * @param char_line_E the energy of the line
	 * @param theta the angle
	 * @return the array which contains the characteristic line.
	 * @throws Exception may happen
	 */
	public static double [] characteristicLine(double [] E, String tubeTarget, double To, double Ek, double char_line_f, double char_line_E, double theta) throws Exception{
		double [] phi = new double [E.length];
		double Emin = E[0];
		double dE = E[1] - E[0];
		double fbin = ((char_line_E - Emin) / dE);
		int il = (int) Math.floor(fbin);
		int ih = il +1;
		double n = N(tubeTarget, To, Ek, char_line_f, char_line_E, theta);
		if (il < phi.length) {
			phi[il] = (ih - fbin) * n;
		}
		if (ih < phi.length) {
			phi[ih] = (fbin - il) * n;
		}
		return phi;
	}

	/**
	 * Ref 1 Eq 22
	 * Returns number of characteristic x rays per electron from target.
	 * The coefficients Ak and nk are empirical values to fit the spectra
	 * exposures to experimental values (see Tucker et al).
	 * Micheal Moreau showed that there is an analytic solution to the integral
	 * in this equation.  His solution is used
	 * @param tubeTarget 'W' or 'Mo'
	 * @param To
	 * @param Ek K absorption energy (keV)
	 * @param fi fractional emission of char line
	 * @param Ei energy of char line
	 * @param theta
	 * @return number of x-rays
	 * @throws Exception
	 */
	public static double N(String tubeTarget, double To, double Ek, double fi, double Ei, double theta) throws Exception{
		/**
		 * Ak		empir photons/electron
		 * nk		empir exponent governing intensity
		 * R         depth (cm) at which average kinetic energy of electrons 
		 *           equals Ek, from T-W relation
		 *
		 * Evaluate integral over distance into target.  Use Gaussian
		 * integration.  The constants Ak, nk, uf and R are first determined
		 * appropriate for the target used
		 */
		double N = 1;
		double Ak, nk, R;
		double [] uf;
		if (tubeTarget.equals("W")){
			Ak = Ak_W;
			nk = nk_W;
			uf = DoubleArrayUtil.divide(DoubleArrayUtil.multiply(mu(new double [] {Ei}, tubeTarget), W_mdensity), Math.sin(theta));
			R = (Math.pow(To,2) - Math.pow(Ek,2)) / (W_mdensity * c(To));
		} else if (tubeTarget.equals("Mo")){
			Ak = Ak_Mo;
			nk = nk_Mo;
			uf = DoubleArrayUtil.divide(DoubleArrayUtil.multiply(mu(new double [] {Ei}, tubeTarget), Mo_mdensity), Math.sin(theta));
			R = (Math.pow(To,2) - Math.pow(Ek,2)) / (Mo_mdensity * c(To));
		} else {
			throw new Exception("Undefined target material in N(n)");
		}
		/* Evaluate the integral in Eq (22) using Micheal Moreau's solution.*/
		double a = -uf[0];
		double eaR = Math.exp(a*R);
		double R2 = R*R;
		double a2 = a*a;
		double integral = 1.5/(R*a) * (eaR - 1. - 1./(R2)*(eaR*(R2 - 2.*R/a + 2./a2) - 2./a2));
		N = Ak*Math.pow((To/Ek-1.0),nk)*fi*integral;	/*x rays/electro*/	
		return N;
	}

	/**
	 * Returns the Thomson-Whiddington constant using a rational approximation
	 * (keV2 cm2 / g).  This routine fits the tabulated values in Birch and
	 * Marshall's paper to within +/- 0.0025 x 1E6 and minimizes the rms error
	 * subject to this restriction.  Maximum error less than +/- 0.4%
	 * @param To incident electron energy (kVp)
	 * @return  Thomson-Whiddington constant
	 */
	public static double c (double To){
		double t0 = 0.20;
		double t1 = 0.47;
		double t2 = 0.21;
		double t3 = 1.08;
		double s0 = 1.00;
		double s1 = -1.19;
		double s2 = 3.00;
		double x = 0.01 * To;
		double c = ((t0 + x*(t1 + x*(t2 + x*t3))) / (s0 + x*(s1 + x*s2)) * 1.E06);
		return c;
	}

	/**
	 * Generates bremsstrahlung x-ray photon spectrum (photons/electron/bin)
	 * using energies in vector E
	 * @param E the energy
	 * @param target the target 
	 * @param To incident electron energy (kVp)
	 * @param theta the angle
	 * @return bremsstrahlung
	 * @throws Exception 
	 */
	public static double [] bremsstrahlung(double [] E, String target, double To, double theta) throws Exception{
		/**
		 * These coefficients are used in the gaussian integration routines.
		 * Use of 8 terms results in an integration accuracy of typically 0.01%.
		 * The greater the number of terms, the longer is the computation time.
		 */
		NumberFormat nf = NumberFormat.getInstance();
		nf.setMaximumFractionDigits(10);
		nf.setMaximumIntegerDigits(10);
		double NGAUSS = 4;
		double [] Gx = {0.183434642495650, 0.525532409916329, 0.796666477413627, 0.960289856497536};
		double [] Gw = {0.362683783378362, 0.313706645877887, 0.222381034453374, 0.101228536290376};
		double dE = E[1] - E[0];
		double f = 0; /* in cm2/g */
		if (target.equals("W")){
			f = dE*1000*ALPHA*Math.pow(RE,2)*Av * (TF_W*(Math.pow(W_Z,2)/W_A) + TF_Re*(Math.pow(Re_Z,2)/Re_A));	
		} else if (target.equals("Mo")){
			f = dE*1000*ALPHA*Math.pow(RE,2)*Av * Math.pow(Mo_Z,2)/Mo_A;
		} else {
			throw new Exception("Undefined target material in bremsstrahlung()");
		}
		/**
		 * Evaluate integral over electron energies T from E to To.  Use
		 * Gaussian integration
		 */
		double [] integral = new double [E.length];
		double [] Tlolim = DoubleArrayUtil.min(Arrays.copyOf(E, E.length),To);
		double Tuplim = To;
		for (int j = 0; j < NGAUSS; j++) {
			double [] T = multiply(add(multiply(Arrays.copyOf(Tlolim, Tlolim.length),(1. - Gx[j])), Tuplim *(1. + Gx[j])), 0.5);
			DoubleArrayUtil.add(integral, DoubleArrayUtil.multiply(Integ(E,target,T,To,theta) , Gw[j]));
			//DoubleArrayUtil.print("Integ", Integ(E,target,T,To,theta), nf);
			T = multiply(add(multiply(Arrays.copyOf(Tlolim, Tlolim.length),(1. + Gx[j])), Tuplim *(1. - Gx[j])),0.5);
			DoubleArrayUtil.add(integral, DoubleArrayUtil.multiply(Integ(E,target,T,To,theta), Gw[j]));
			
		}
		/* g/(keV cm2) */
		DoubleArrayUtil.multiply(integral, multiply(add(multiply(Arrays.copyOf(Tlolim,Tlolim.length),-1),Tuplim),0.5));  
		double [] phi = DoubleArrayUtil.multiply(integral, f);
		DoubleArrayUtil.divide(phi, E);
		return phi;
	}

	public static double [] Integ(double [] Ei, String target, double [] T, double To, double theta) throws Exception {
		double [] B = B(Ei,target,T,To);
		NumberFormat nf = NumberFormat.getInstance();
		nf.setMaximumFractionDigits(10);
		nf.setMaximumIntegerDigits(10);
		
		DoubleArrayUtil.multiply(B, divide(add(Arrays.copyOf(T, T.length),MoC2),T));
		
		DoubleArrayUtil.multiply(B, F(Ei,target,T,To,theta));
		//print("B", F(Ei,target,T,To,theta), nf);
		DoubleArrayUtil.divide(B, MSP(target,T));
		return B;
	}

	/**
	 * 
	 * Ref 1 Eq 18
	 * Proportional to x-ray photons/electron (unitless).
	 * The coefficients A0 and A1 are empirical values to fit the spectra
	 * exposures to experimental values (see Tucker et al).  The values
	 * of B_W and B_Mo are spectral shape parameters, and are not changed
	 * from Tucker's empirically derived values
	 * @param E
	 * @param target
	 * @param T
	 * @param To
	 * @return B
	 * @throws Exception
	 */
	public static double [] B(double [] E, String target, double [] T, double To) throws Exception{
		double [] B_W =  {-5.049, 10.847, -10.516, 3.842};
		double [] B_Mo = {-4.238, 7.799, -6.739, 2.313};
		double [] C = Arrays.copyOf(E, E.length);
		DoubleArrayUtil.divide(C, T);
		double A0, A1, B1, B2, B3, B4;
		if (target.equals("W")){
			A0 = A0_W;
			A1 = A1_W;
			B1 = B_W[0];
			B2 = B_W[1];
			B3 = B_W[2];
			B4 = B_W[3];
		} else if (target.equals("Mo")){
			A0 = A0_Mo;
			A1 = A1_Mo;
			B1 = B_Mo[0];
			B2 = B_Mo[1];
			B3 = B_Mo[2];
			B4 = B_Mo[3];
		} else {
			throw new Exception("Undefined target material in B()");
		}
		double factor = (A0 + A1*To);
		double [] br = new double [C.length];
		for (int i = 0; i < br.length; i++){
			br[i] = factor *(1.0 + (B1*C[i]) + (B2*Math.pow(C[i],2)) + (B3*Math.pow(C[i],3)) + (B4*Math.pow(C[i],4)));
		}
		return br;
	}

	/**
	 * Ref 1 Eq 16
	 * Returns mass stopping power term for specified material and 
	 * electron energy T
	 * @param material
	 * @param T
	 * @return MSP
	 * @throws Exception
	 */
	public static double [] MSP(String material, double [] T) throws Exception{
		double Amsp_W = 2024.1;	    /* keV cm2 / g */
		double Bmsp_W = 10361.0;	/* keV cm2 / g */
		double Cmsp_W = 0.04695;	/* 1/keV */
		double Amsp_Re = 2014.4;	/* keV cm2 / g */
		double Bmsp_Re = 10276.0;	/* keV cm2 / g */
		double Cmsp_Re = 0.04688;	/* 1/keV */
		double Amsp_Mo = 2458.08;	/* keV cm2 / g */
		double Bmsp_Mo = 14155.1;	/* keV cm2 / g */
		double Cmsp_Mo = 0.04983;	/* 1/ke */
		double [] msp = null;
		if (material.equals("W")){
			double [] temp = Arrays.copyOf(T, T.length);
			multiply(temp, -Cmsp_W);
			exp(temp);
			multiply(temp, Bmsp_W);
			multiply(add(temp, Amsp_W), TF_W);
			msp = temp;
			temp = Arrays.copyOf(T, T.length);
			multiply(temp, -Cmsp_Re);
			exp(temp);
			multiply(temp, Bmsp_Re);
			add(temp, Amsp_Re);
			multiply(temp, TF_Re);
			add(msp, temp);
		} else if (material.equals("Mo")){
			msp = new double[T.length];
			for (int i = 0; i < T.length; i++) {
				msp[i] = Amsp_Mo + Bmsp_Mo * Math.exp(-T[i] * Cmsp_Mo);
			}
		} else {
			throw new Exception ("Undefined target material in MSP()");
		}
		return msp;
	}

	/**
	 * Ref 1 Eq 11
	 * Returns fraction of x-ray photons of energy E escaping target material.
	 * Note: mu() returns mass attenuation data, so we can drop the density 
	 * term in Tucker's paper
	 * @param E
	 * @param target
	 * @param T
	 * @param To
	 * @param theta
	 * @return F
	 * @throws Exception
	 */
	public static double [] F(double [] E, String target, double [] T, double To, double theta) throws Exception{
		double [] u;
		if (target.equals("W")){
			u = DoubleArrayUtil.multiply(mu(E, "W"), TF_W);
			DoubleArrayUtil.add(u, DoubleArrayUtil.multiply(mu(E,"Re"),TF_Re));
		} else if (target.equals("Mo")){
			u = mu(E, "Mo");
		} else {
			throw new Exception ("Undefined target material in F()");
		}
		double factor [] = divide(add( multiply(pow(Arrays.copyOf(T, T.length),2), -1),Math.pow(To,2)), (c(To)*Math.sin(theta)));
		NumberFormat nf = NumberFormat.getInstance();
		nf.setMaximumFractionDigits(10);
		nf.setMaximumIntegerDigits(10);
		//print("factor",mu(E, "W"), nf);
		DoubleArrayUtil.multiply(u, -1);
		DoubleArrayUtil.multiply(u, factor);
		DoubleArrayUtil.exp(u);
		return u;
	}
	
	/**
	 * Returns the number of quanta per R of exposure at the energies
	 * specified by vector E (keV).
	 * @param airAbsorption
	 * @return number of quanta
	 * @throws Exception
	 */
	public static double xrqpRe(double airAbsorption) throws Exception{
		
		double E_CHG = 1.6021892E-19;   // charge on an electron (C)
		double C_kg_R = 2.580E-4;       // Coul/kg/R for air
		double W_air = 33.97;           // work function for air (eV)

		// Calculate quanta/R values		
		double f = W_air * C_kg_R * 10.0 / (E_CHG * 1E9);
		
		return f / airAbsorption;		
	}

	public static void main(String [] argv){	
		double [] energy = {10, 20, 30, 40, 50, 60, 70, 80};
		try {
			double [] F = generateXRaySpectrum(energy, 120, "W", 1);
			DoubleArrayUtil.print("E", energy);
			NumberFormat nf = NumberFormat.getInstance();
			nf.setMaximumFractionDigits(10);
			nf.setMaximumIntegerDigits(10);
			DoubleArrayUtil.print("F", F, nf);
						
			/**
			 * Half value layer (HVL)
			 * 1. Generate x-ray spectrum
			 * 2. Optimize x thickness for HVL using optimizer
			 */
			double min = 10.0; // Minimum [keV]
			double max = 150.0; // Maximum [keV]
			double delta = 0.5; // Delta [keV]
			double peakVoltage = 125.0; // Peak Voltage [kVp]			
			double timeCurrentProduct = 1.0; // Time Current Product [mAs]
			
			int steps = (int) ((max - min) / delta);
			double [] energies = new double [steps];
			for (int i = 0; i < steps; i ++){
				energies[i] = min + (i*delta);
			}
			double [] spectrum = XRaySpectrum.generateXRaySpectrum(energies, peakVoltage, "W", timeCurrentProduct, 1, 12, 2.38, 3.06, 2.66, 1.2); // C-arm: 1.2mm, CT: 10.5mm			
			//* PCXMC 1.5, Al 1.2mm, 125KVp 
			//double [] spectrum = {2.339734153, 4.041388504, 27.38163849, 15.39283896, 37.18279457, 73.56337068, 126.0434512, 194.3001899, 276.7188396, 370.9872404, 474.525746, 584.7174788, 698.9984541, 814.8879307, 930.0170453, 1042.178469, 1149.392328, 1249.971553, 1342.570514, 1426.207848, 1500.26219, 1564.445105, 1618.758252, 1663.442134, 1698.922723, 1725.760589, 1744.605441, 1756.157569, 1761.136588, 1760.257185, 1754.21114, 1743.654715, 1729.20045, 1711.412455, 1690.804388, 1667.839433, 1642.931697, 1616.448585, 1588.713784, 1560.0106, 1530.585431, 1500.651249, 1470.390973, 1439.960685, 1409.492634, 1379.098007, 1348.869472, 1318.883473, 4124.783498, 6130.065636, 1230.943841, 1202.436086, 1174.375023, 1146.776223, 1119.649532, 1092.999939, 1066.828356, 2502.684011, 1015.906379, 1366.607548, 747.2849712, 683.2666223, 675.4686445, 667.3257692, 658.8505088, 650.0551081, 640.9515346, 631.5514713, 621.8663123, 611.907161, 601.6848301, 591.2098444, 580.4924441, 569.542591, 558.3699745, 546.9840201, 535.3938979, 523.6085318, 511.6366095, 499.4865931, 487.1667296, 474.6850614, 462.0494375, 449.2675242, 436.3468158, 423.2946452, 410.118194, 396.824503, 383.4204816, 369.9129176, 356.308486, 342.6137582, 328.8352099, 314.9792297, 301.0521261, 287.0601356, 273.0094287, 258.9061174, 244.7562604, 230.5658698, 216.340916, 202.0873332, 187.8110241, 173.5178642, 159.2137066, 144.9043855, 130.59572, 116.2935175, 102.0035769, 87.73169164, 73.48365225, 59.26524903, 45.08227425, 30.94052435, 16.84580188, 3.363752905, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
			//* PCXMC 1.5, Al 1.2mm, 109KVp
			//double [] spectrum = {2.955266803, 5.534583134, 35.53063201, 20.78504989, 50.10945855, 98.9266474, 169.1105885, 260.0284174, 369.2513097, 493.3143776, 628.2639988, 769.9886413, 914.4277354, 1057.738864, 1196.452481, 1327.605535, 1448.83427, 1558.413734, 1655.244323, 1738.795765, 1809.023517, 1866.272443, 1911.179806, 1944.585811, 1967.456388, 1980.820069, 1985.718866, 1983.171895, 1974.149924, 1959.558842, 1940.230127, 1916.916613, 1890.292118, 1860.95375, 1829.426, 1796.165903, 1761.568761, 1725.974056, 1689.671287, 1652.905576, 1615.882911, 1578.774999, 1541.723672, 1504.844866, 1468.232175, 1431.960007, 1396.086368, 1360.655307, 3472.502797, 4966.973123, 1257.291784, 1223.861716, 1190.950986, 1158.5562, 1126.670174, 1095.282699, 1064.381199, 2122.305934, 1003.977217, 1253.30148, 786.1018455, 731.0427146, 716.8576874, 702.2039203, 687.1004379, 671.5660379, 655.6192709, 639.2784244, 622.5615106, 605.4862593, 588.070113, 570.3302259, 552.283465, 533.9464137, 515.3353769, 496.4663881, 477.3552179, 458.0173827, 438.4681553, 418.7225753, 398.7954603, 378.7014171, 358.4548537, 338.0699904, 317.5608713, 296.9413762, 276.2252309, 255.4260187, 234.5571907, 213.6320761, 192.6638916, 171.6657515, 150.6506761, 129.6316007, 108.6213836, 87.63281374, 66.67861822, 45.77146919, 24.92399031, 4.977138706, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
			//* PCXMC 1.5, Al 1.2mm, 81KVp
			//double [] spectrum = {5.378719397, 11.49101655, 67.12856602, 42.13840172, 100.9364506, 197.8129205, 335.2339375, 510.0244283, 714.7811096, 939.6352271, 1173.886102, 1407.319085, 1631.128801, 1838.432695, 2024.422216, 2186.243436, 2322.712599, 2433.959903, 2521.069511, 2585.757072, 2630.104018, 2656.352883, 2666.75903, 2663.490021, 2648.562707, 2623.808682, 2590.860072, 2551.149209, 2505.917269, 2456.228218, 2402.9855, 2346.949702, 2288.756026, 2228.930879, 2167.907149, 2106.038006, 2043.609155, 1980.849596, 1917.940972, 1855.025634, 1792.213549, 1729.588181, 1667.211485, 1605.128117, 1543.36897, 1481.954142, 1420.895412, 1360.19829, 1843.377528, 2165.763967, 1180.271382, 1121.004084, 1062.08199, 1003.49979, 945.2529603, 887.338147, 829.7534654, 1040.71992, 715.5755988, 727.4210604, 573.9540227, 517.2982556, 468.0250995, 418.2334156, 368.0000507, 317.4008233, 266.5105365, 215.4029968, 164.1510415, 112.8265696, 61.50057796, 12.2895214, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
			//* PCXMC 1.5, Al 1.2mm, 70KVp
			//double [] spectrum = {7.481922341, 16.50277126, 92.86281442, 60.83853289, 144.9671387, 282.2596166, 474.4437357, 714.5596079, 989.5192823, 1283.358776, 1580.139999, 1865.998848, 2130.235548, 2365.596519, 2568.006225, 2736.006804, 2870.104526, 2972.149214, 3044.810436, 3091.171012, 3114.433025, 3117.719489, 3103.951195, 3075.779185, 3035.556262, 2985.334486, 2926.878875, 2861.690343, 2791.033077, 2715.963189, 2637.35668, 2555.935555, 2472.291509, 2386.906945, 2300.173352, 2212.407152, 2123.863256, 2034.746552, 1945.221583, 1855.420659, 1765.450604, 1675.398367, 1585.335643, 1495.322682, 1405.411399, 1315.647914, 1226.074601, 1136.731752, 1052.909937, 967.8248154, 870.4834816, 782.4645221, 694.8838466, 607.7889466, 521.2301659, 435.2608781, 349.9375973, 267.8868369, 181.4711198, 99.11088352, 19.59335481, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
			//* PCXMC 1.5, Al 0.0mm, 70KVp, Anode 6 deg
			//double [] spectrum = {7.481922341, 16.50277126, 92.86281442, 60.83853289, 144.9671387, 282.2596166, 474.4437357, 714.5596079, 989.5192823, 1283.358776, 1580.139999, 1865.998848, 2130.235548, 2365.596519, 2568.006225, 2736.006804, 2870.104526, 2972.149214, 3044.810436, 3091.171012, 3114.433025, 3117.719489, 3103.951195, 3075.779185, 3035.556262, 2985.334486, 2926.878875, 2861.690343, 2791.033077, 2715.963189, 2637.35668, 2555.935555, 2472.291509, 2386.906945, 2300.173352, 2212.407152, 2123.863256, 2034.746552, 1945.221583, 1855.420659, 1765.450604, 1675.398367, 1585.335643, 1495.322682, 1405.411399, 1315.647914, 1226.074601, 1136.731752, 1052.909937, 967.8248154, 870.4834816, 782.4645221, 694.8838466, 607.7889466, 521.2301659, 435.2608781, 349.9375973, 267.8868369, 181.4711198, 99.11088352, 19.59335481, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
			
			
			//System.out.println(spectrum.length);
			
			HalfValueLayerFunction hvlf = new HalfValueLayerFunction(energies, spectrum, 
					EnergyDependentCoefficients.getPhotonMassAttenuationLUT(EnergyDependentCoefficients.Material.Aluminum), 
					EnergyDependentCoefficients.getMassEnergyAbsorptionLUT(EnergyDependentCoefficients.Material.CsI),
					new LinearInterpolatingDoubleArray(EnergyDependentCoefficients.airEnergies, EnergyDependentCoefficients.airAbsoprtion));
			hvlf.runOptimization();
	
			System.out.println(peakVoltage + "kVp\t" + hvlf.getOptimalX() * 10 + "[mm]\t" + hvlf.evaluate(hvlf.optimalX)+ "\t" + CONRAD.DOUBLE_EPSILON);
			
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
}
