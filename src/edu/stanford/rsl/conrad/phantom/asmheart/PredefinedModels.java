/*
 * Copyright (C) 2010-2014 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.phantom.asmheart;

public abstract class PredefinedModels {

	/**
	 * This enum lists all currently available parameter sets.
	 * @author Mathias Unberath
	 *
	 */
	public enum parameterSets{
		CONSENSUS("Mean heart shape", new double[16]),
		Male_1("Male, 56 y/o, 56% EF", new double[]{0.022544633743535714, 0.21366267246301, -0.14564951608394075, 0.1293173176263205, -0.22068740751069113, 0.256929936068349, -0.22637531194093038, -0.01965228752145983, -0.2264441400440098, 0.12417225104418468, -0.19140784032643554, -0.16755756864029042, 0.29057124902309167, -0.07157275039564767, -0.5622090604138195, -0.37573167835005084}),
		Male_2("Male, 62 y/o, 48% EF, dilated", new double[]{-0.22905062285225697, 0.44787201767258317, -0.32473883001297565, 0.040222293243028506, 0.2771626048379564, 0.08591340860963428, 0.006035160322433137, 0.16599960168071543, -0.0955207944639088, 0.28004890135756993, -0.3791343184887183, 0.154368412566871, 0.05230659592094386, 0.11312568012199696, 0.30716980275084127, 0.11835221683109462}),
		Female_1("Female, 81 y/o, 62% EF", new double[]{-0.22195131279368494, -0.11480065636800553, -0.008168355827022366, 0.02758316396625994, -0.10611587562866047, 0.19610023461516465, -0.1355384550269164, 0.08325846982522841, 0.2411560215105199, 0.23006625828565547, 0.4368111449102145, -0.25630160920367967, 0.4027227632763022, 0.014125686175094259, 0.34476152092497736, -0.034563036539214664}),
		Female_2("Female, 52 y/o, 54% EF", new double[]{-0.35306459910952503, -0.24568489166183832, 0.07464225739612732, -0.06643350963979718, -0.15061969169359385, -0.2492101844706318, 0.13378026421076245, -0.3008813713022111, -0.09852013993329066, -0.02441803132861153, -0.049587244082861394, -0.15655754622756152, 0.360894142076674, 0.21832622485011446, 0.03854109632232743, 0.055448172929560446});
		
		private double[] parameters;
		private String name;
		
		private parameterSets(String name, double[] param){
			this.name = name;
			this.parameters = param;
		}
		public String getName(){
			return this.name;
		}
		
		public double[] getValue(){
			return this.parameters;
		}
	}
	
	/**
	 * Returns the parameter set for a heart type described by the key.
	 * @param key The key-word for the heart model.
	 * @return The paramters of the corresponding example heart.
	 */
	public static double[] getValue(String key){
		double[] val = new double[parameterSets.CONSENSUS.getValue().length];
		for(parameterSets ps : parameterSets.values()){
			if(key.equals(ps.getName())){
				val = ps.getValue();
			}
		}
		return val;
	}
	
	/**
	 * Returns all currently available heart shapes as string array.
	 * @return String array containing available heart shapes.
	 */
	public static String[] getList(){
		int nTypes = parameterSets.values().length;
		String[] list = new String[nTypes];
		
		int i = 0;
		for(parameterSets ps : parameterSets.values()){
			list[i] = ps.getName();
			i++;
		}
		
		return list;
	}

}
