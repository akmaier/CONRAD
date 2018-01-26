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
		Male_1("Male, 56 y/o, 56% EF", new double[]{0.09826978021880595, 0.9313339972705663, -0.6348715217874883, 0.5636811191825208, -0.9619541074508376, 1.11993162689239, -0.9867471080634033, -0.08566233531504264, -0.9870471228087027, 0.5412542938934521, -0.8343274329844081, -0.730366508928559, 1.2665707103899584, -0.31197838608591555, -2.4506124794867956, -1.6377764158147925}),		
		// CB_3938
		Male_2("Male, 62 y/o, 48% EF, dilated", new double[]{-0.9984085179435974, 1.9522288646852437, -1.4155037430729585, 0.17532491152206714, 1.2081237854166302, 0.3744878660248143, 0.02630665395417368, 0.723575488394582, -0.41636549007451484, 1.2207048602669808, -1.652608180320767, 0.672876310453733, 0.22799916570010761, 0.4931034075711377, 1.3389221286982278, 0.5158853529107578}),
		// MC_8275
		Female_1("Female, 81 y/o, 62% EF", new double[]{-0.9674633428605156, -0.5004044597482089, -0.03560503758380885, 0.12023222427270018, -0.46254837817041755, 0.854781105491683, -0.590798428425981, 0.362915256161431, 1.0511747273909122, 1.0028355701860756, 1.9040156380757012, -1.1171928135859635, 1.7554278273846922, 0.0615724385454853, 1.502780629333355, -0.15065678345633934}),
		// CL_2740
		Female_2("Female, 52 y/o, 54% EF", new double[]{-1.5389729080735708, -1.0709156146884617, 0.32535805691024744, -0.28957695498267694, -0.6565360149991084, -1.0862820098093873, 0.5831346523343194, -1.311511491500362, -0.4294393338735068, -0.10643573096178495, -0.21614578584581726, -0.6824185228548257, 1.573101094627925, 0.951661950846417, 0.16799674404226703, 0.24169298240392126});
		
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
