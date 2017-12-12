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
		// KL_6064
		Male_1("Male, 56 y/o, 56% EF", new double[]{0.09826978020720925, 0.9313339972730911, -0.6348715217854997, 0.5636811191828822, -0.9619541074510817, 1.1199316268922994, -0.9867471080630119, -0.08566233531544881, -0.9870471228088108, 0.5412542938935639, -0.8343274329843017, -0.7303665089284058, 1.2665707103900483, -0.31197838608588896, -2.4506124794867925, -1.637776415814801}),
		// CB_3938
		Male_2("Male, 62 y/o, 48% EF, dilated", new double[]{-0.9984085179680363, 1.9522288646744523, -1.415503743070194, 0.17532491152382013, 1.2081237854171494, 0.37448786602451306, 0.026306653953552393, 0.723575488394243, -0.4163654900748979, 1.220704860267238, -1.6526081803204873, 0.6728763104537849, 0.2279991657000111, 0.4931034075710927, 1.3389221286982391, 0.5158853529107551}),
		// MC_8275
		Female_1("Female, 81 y/o, 62% EF", new double[]{-0.9674633428538589, -0.5004044597602751, -0.035605037584872094, 0.12023222427203963, -0.46254837817066163, 0.8547811054921194, -0.590798428425961, 0.36291525616200115, 1.051174727390677, 1.0028355701856995, 1.904015638075926, -1.117192813585694, 1.7554278273848545, 0.061572438545405464, 1.5027806293333597, -0.15065678345634048}),
		// CL_2740
		Female_2("Female, 52 y/o, 54% EF", new double[]{-1.5389729080601202, -1.0709156147086918, 0.3253580569074704, -0.28957695498461106, -0.6565360149996282, -1.0862820098086134, 0.5831346523348847, -1.311511491500277, -0.4294393338727, -0.10643573096162787, -0.216145785845878, -0.6824185228546381, 1.573101094628032, 0.9516619508463875, 0.1679967440422924, 0.2416929824039216});
		
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
