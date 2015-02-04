package edu.stanford.rsl.conrad.physics;

import edu.stanford.rsl.conrad.fitting.LinearFunction;

public class Constants {
	
	/** fine struct const (unitless)	 */
	public static double ALPHA   = 7.297351E-3;
	
	/** classical elec radius (cm)	 */
	public static double RE		= 2.8179E-13;
	
	/** electron rest mass (keV)	 */
	public static double MoC2	= 	511.003;
	
	/** charge on an electron (C)	 */
	public static double E_CHG	= 	1.6021892E-19;
	
	/** electrons per mAs		 */
	public static double EpMAS	=	6.24146E15;
	
	/** Avogadro's number (/mol)	 */
	public static double Av		= 6.02252E23;
	
	/** Coul/kg/R for air		 */
	public static double C_kg_R	=	2.580E-4;
	
	/** work function for air (eV)	 */
	public static double W_air   =	33.97;		

	/** atomic number, tungsten	 */
	public static double W_Z		= 74.0;
	
	/** atomic weight, tungsten (g/mol) */
	public static double W_A		= 183.85;
	
	/** tungsten mass density (g/cm3) */
	public static double W_mdensity	= 19.3;		

	/** tungsten K-edge (keV)	 */
	public static double W_K		= 69.5;
	
	/** 
	 * tungsten K-alpha 1 line	 
	 */
	public static double W_Ka1	= 	59.32;
	/** tungsten K-alpha 2 line	 */
	public static double W_Ka2	=	57.98;
	/** tungsten K-beta 1 line	 */
	public static double W_Kb1	=	67.2;
	/** 
	 * tungsten K-beta 2 line<br>
	 * incorrectly 67.1 in Tucker paper */
	public static double W_Kb2	=	69.1;		

	/** 
	 * tungsten K-alpha 1 line<br>
	 * fluorescent yield
	 */
	public static double W_Ka1_f	=	0.45;	
	
	/** 
	 * tungsten K-alpha 2 line<br>
	 * fluorescent yield
	 */
	public static double W_Ka2_f	=	0.2592;
	
	/** 
	 * tungsten K-beta 1 line<br>
	 * fluorescent yield
	 */
	public static double W_Kb1_f	=	0.1521;
	
	/** 
	 * tungsten K-beta 2 line<br>
	 * fluorescent yield
	 */
	public static double W_Kb2_f	=	0.0387;

	/** atomic number, Rhenium	 */
	public static double Re_Z	=	75.0;
	/** atomic weight, Rhenium (g/mol) */
	public static double Re_A	=	186.2;
	/** Rhenium mass density (g/cm3) */
	public static double Re_mdensity	=20.53;

	/** Rhenium K-edge (keV)	 */
	public static double Re_K    =	71.5;
	/** 
	 * Rhenium K-alpha 1 line	 
	 */
	public static double Re_Ka1	=	61.14;
	/** 
	 * Rhenium K-alpha 2 line	 
	 */
	public static double Re_Ka2	=	59.72;
	/** 
	 * Rhenium K-beta 1 line	 
	 */
	public static double Re_Kb1	=	69.2;
	/** 
	 * Rhenium K-beta 2 line	 
	 */
	public static double Re_Kb2	=	71.2;
	
	/** 
	 * Rhenium K-alpha 1 line<br>
	 * fluorescent yield
	 */
	public static double Re_Ka1_f=	0.04988;
	/** 
	 * Rhenium K-alpha 2 line<br>
	 * fluorescent yield
	 */
	public static double Re_Ka2_f=	0.02883;
	/** 
	 * Rhenium K-beta 1 line<br>
	 * fluorescent yield
	 */
	public static double Re_Kb1_f=	0.0173;
	/** 
	 * Rhenium K-beta 2 line<br>
	 * fluorescent yield
	 */
	public static double Re_Kb2_f=	0.00429;

	/** atomic number, molybdenum	 */
	public static double Mo_Z	=	42.0;
	/** atomic weight, molybdenum (g/mol) */
	public static double Mo_A	=	95.94;
	
	/** molybdenum mass density	 */
	public static double Mo_mdensity	= 10.22;		

	/** molybdenum K-edge (keV)	 */
	public static double Mo_K    =	19.965;
	
	/** molybdenum K-alpha 1 line	 */
	public static double Mo_Ka1	=	17.48;
	/** molybdenum K-alpha 2 line	 */
	public static double Mo_Ka2	=	17.37;
	/** molybdenum K-beta 1 line	 */
	public static double Mo_Kb1	=	19.61;
	/** molybdenum K-beta 2 line	 */
	public static double Mo_Kb2	=	19.96;

	/** 
	 * molybdenum K-alpha 1 line<br>
	 * fluorescent yield
	 */
	public static double Mo_Ka1_f=	0.5599;
	/** 
	 * molybdenum K-alpha 2 line<br>
	 * fluorescent yield
	 */
	public static double Mo_Ka2_f=	0.2833;
	/** 
	 * molybdenum K-beta 1 line<br>
	 * fluorescent yield
	 */
	public static double Mo_Kb1_f=	0.1344;
	/** 
	 * molybdenum K-beta 2 line<br>
	 * fluorescent yield
	 */
	public static double Mo_Kb2_f=	0.0224;

	/** aluminum mass density 2.70 (g/cm3) */
	public static double Al_mdensity	=2.70;
	/** soft bone mass density 1.20 (g/cm3) */
	public static double Bone_mdensity	= 1.20;
	/** Cortical bone (ICRU-44) mass density 1.920E+00 (g/cm3) */
	public static double CorticalBone_mdensity	= 1.920E+00;
	/** B-100 Bone-Equivalent Plastic mass density  1.450E+00 (g/cm3) */
	public static double SoftBone_mdensity	= 1.450E+00;
	/** soft tissue mass density 1.00 (g/cm3) */
	public static double SoftTissue_mdensity	= 1.00;
	/** water mass density 1.00 (g/cm3)	 */
	public static double Water_mdensity=1.00;
	/** air mass density (g/cm3)	 */
	public static double AIR_mdensity=0.001213;
	/** oil mass density 0.85 (g/cm3)	 */
	public static double OIL_mdensity=0.85;
	/** Titanium mass density 4.540E+00 (g/cm3)	 */
	public static double Ti_mdensity=4.540E+00;
	/** Gold mass density 1.932E+01 (g/cm3)	 */
	public static double Au_mdensity=1.932E+01;
	/** Lexan mass density 1.25 (g/cm3)	 */
	public static double LEXAN_mdensity=1.25;
	/** Lucite mass density 1.19 (g/cm3)	 */
	public static double Lucite_mdensity=1.19;
	/** Teflon mass density 2.250E+00 (g/cm3)	 */
	public static double TEFLON_mdensity=2.250E+00;
	/** Polystyrene mass density 1.06E+00 (g/cm3)	 */
	public static double Polystyrene_mdensity=1.06E+00;	
	/** Pyrex glass mass density 2.23 (g/cm3)	 */
	public static double PYREX_mdensity=2.23;	
	/** CsI mass density 4.51 (g/cm3)	 */
	public static double CsI_mdensity=4.51;	
	

	/** 
	 * CT Values according to CatPhan Manual
	 */
	public static double AirCTValue = -1000;
	public static double WaterCTValue = 0;
	public static double AcrylicCTValue = 120;
	public static double SoftTissueCTValue = 35;
	public static double SoftBoneCTValue = 300;
	public static double TeflonCTValue = 990;
	public static double PolystyreneCTValue = -35;
	public static double LuciteCTNumber = 110;
	public static double PDS2BaseMaterialCTNumber = 292.306;
	public static double PlasticineMaterialCTNumber = 252.3;	
	public static double QrmHA1100MaterialCTNumber = 1661.417;	
	public static double LegCylinder = 68.962;
	
	/** W target fraction Tungsten	 */
	public static double TF_W	=	0.9;		
	/** W target fraction Rhenium	 */
	public static double TF_Re	=	0.1;		


	/**
	 * Empirical coefficients that are used to ensure agreement between
	 * calculated and experimental exposure values for both the bremsstrahlung
	 * and characteristic components are defined here.  The experimental
	 * values are those published by Tucker et al.  Small changes in these
	 * coefficients were required to the Tucker values, which are included
	 * in the square brackets. 
	 */
	
	/**
	 * Bremsstrahlung<BR>
	 * photons/electron [3.685E-2] (3.90e-2 for R2)
	 */
	public static double A0_W	= 3.6E-2;
	/**
	 * Bremsstrahlung<BR>
	 *  photons/(electron keV) [2.900E-5] (2.30e-5 for R2)
	 */
	public static double A1_W	= 2.9E-5; 
	/**
	 * Bremsstrahlung<BR>
	 * photons/electron [3.033E-2] (3.0e-2)
	 */
	public static double A0_Mo	= 2.3E-2;  
	/**
	 * Bremsstrahlung<BR>
	 * photons/(electron keV) [-7.494E-5] (-6.2e-5)
	 */
	public static double A1_Mo	=-0.E-5;  

	/**
	 * Characteristic<br>
	 * photons/electron [1.349E-3]
	 */
	public static double Ak_W	= 1.30E-3;  
	/**
	 * Characteristic<br>
	 * dimensionless exponent [1.648]
	 */
	public static double nk_W	= 1.65;     
	/**
	 * Characteristic<br>
	 * photons/electron [7.773E-4] (7.90e-4)
	 */
	public static double Ak_Mo	= 6.9E-4;
	/**
	 * Characteristic<br>
	 * dimensionless exponent [1.613] (1.59)
	 */
	public static double nk_Mo	= 1.613;
	
	/**
	 * Mass attenuation coefficients for Aluminium
	 */
	public static double [] aAl = {  .172651e0,  .223596e0,  -.703254e0,   .842642e0,  -.564110e0,  .237632e0, -.431417e-1,   .461017e-2, -.234452e-3};
	public static double [] aWl = {  .000000e0, -.170740e2,   .672096e2,  -.109599e3,   .969173e2, -.489856e2,   .147314e2,  -.239704e1,   .162248e0};
	public static double [] aWh = {  .645476e0, -.670778e1,   .391826e2,  -.121589e3,   .221639e3, -.234715e3,   .148981e3,  -.485430e2,   .553248e1};
	public static double [] aMol= {  .000000e0, -.952203e2,   .192429e3,  -.163398e3,   .755826e2, -.204435e2,   .328347e1,  -.288511e0,   .106862e-1};
	public static double [] aMoh= {  .213590e0, -.495334e0,   .290085e1,  -.863362e1,   .145314e2, -.134362e2,   .841471e1,  -.272723e1,   .347591e0};
	public static double [] aRel= {  .000000e0,  .120729e2,  -.473461e2,   .801273e2,  -.740837e2,  .417432e2,  -.136463e2,   .245235e1,  -.186391e0};
	public static double [] aReh= { -.824620e-1, .428168e1,  -.309579e2,   .127384e3,  -.341862e3,  .481770e3,  -.428135e3,   .208457e3,  -.431927e2};
	public static double [] aOIL= {  .217381e0,  .170909e0,  -.571149e0,   .616428e0,  -.374410e0,  .139485e0,  -.307085e-1,  .384552e-2, -.205047e-3};
	public static double [] aLEX= {  .199252e0,  .169128e0,  -.557894e0,   .611422e0,  -.377486e0,  .143073e0,  -.316256e-1,  .400862e-2, -.216943e-3};
	public static double [] aPYR= {  .181903e0,  .203691e0,  -.651756e0,   .764128e0,  -.503556e0,  .206070e0,  -.397921e-1,  .451715e-2, -.237949e-3};
	public static double [] aAIR= {  .186903e0,  .172942e0,  -.562708e0,   .627935e0,  -.395378e0,  .153723e0,  -.337658e-1,  .432223e-2, -.238894e-3};

	/**
	 * converts from CT values to mass density
	 * @param ctValue
	 * @return the density estimate
	 */
	public static double computeMassDensity(double ctValue){
		double [] ctValues = {AirCTValue, SoftBoneCTValue, WaterCTValue, TeflonCTValue, PolystyreneCTValue};
		double [] massDensities = {AIR_mdensity, SoftBone_mdensity, Water_mdensity, TEFLON_mdensity, Polystyrene_mdensity};
		LinearFunction func = new LinearFunction();
		func.fitToPoints(ctValues, massDensities);
		return func.evaluate(ctValue);
	}
	
	/**
	 * converts from mass density to CT value
	 * @param massDensity
	 * @return the CT estimate
	 */
	public static double computeCTValue(double massDensity){
		double [] ctValues = {AirCTValue, SoftBoneCTValue, WaterCTValue, TeflonCTValue, PolystyreneCTValue};
		double [] massDensities = {AIR_mdensity, SoftBone_mdensity, Water_mdensity, TEFLON_mdensity, Polystyrene_mdensity};
		LinearFunction func = new LinearFunction();
		func.fitToPoints(massDensities, ctValues);
		return func.evaluate(massDensity);
	}
	
}
/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/