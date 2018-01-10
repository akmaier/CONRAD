/*
 * Copyright (C) 2017 Tobias Geimer
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.apps.activeshapemodel;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.StringTokenizer;

import edu.stanford.rsl.conrad.geometry.shapes.activeshapemodels.PCA;
import edu.stanford.rsl.conrad.io.PcaIO;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;

/**
 * This class provides methods for updating the saved eigenvalues produced by {@link PCA} and {@link ActiveShapeModel} in the
 * conrad.geometry.shapes.activeshapemodel package.
 * Older versions of {@link PCA} wrongly used eigenvalues as the singular values of the data matrix instead of their squares divided by numSamples-1.
 * Additionally, {@link ActiveShapeModel} now scales with the square root of the new eigenvalues (standard variation) instead of the eigenvalues
 * themselves (would be variance). As a result, potential pc scores that have been produced with older versions prior to 2017-12-21 need to be updated in order to be
 * compatible with current implementation.
 * This way we ensure backwards compatibility with older {@link CONRADCardiacModel} model and score files, as well as other {@link PcaIO}-written objects.
 * 
 * Adjust the path in main to the files you want to update or use the static update-call.
 * Do not run this update twice on the same file!
 * No check is performed whether or not a file has already been updated.
 * 
 * @author Tobias Geimer
 *
 */
public class PcaHotfixScript {
	// ========================================================================================================================
	// STATIC CALL
	/**
	 * Updates the static heart phases of a CONRADCardiacModel files for model (.ccm) and scores (.ccs) located in directory.
	 * Run this only for {@link PcaIO}-written files, that have been created with the old version of {@link PCA}
	 * and {@link ActiveShapeModel} from the conrad.geometry.shapes.activeshapemodel package before the Update on
	 * December 21th, 2017.
	 * 
	 * @param directory the directory containing the files
	 * @param ccmFileName the model file
	 * @param ccsFileName the score file
	 */
	public static void updatePhase(String directory, String ccmFileName, String ccsFileName)	{
		PcaHotfixScript updater = new PcaHotfixScript(directory, ccmFileName, ccsFileName);
		updater.runForPhase();
	}
	
	/**
	 * Recalculates the dynamic files for model (.ccm) and scores (.ccs)  of the CONRADCardiacModel located in directory.
	 * It is assume that the phase specific score files (.ccs) are provided in phaseScoreFiles (path starting from the given directory).
	 * This step replicates chapter 2.3 in Unberath et al., "Open-source 4D statistical shape model of the heart for x-ray projection imaging", Proc. ISBI 2015.
	 * 
	 * Run this only for {@link ConradCardiacModel} files of the second PCA step, that have been created with the old version of {@link PCA}
	 * and {@link ActiveShapeModel} from the conrad.geometry.shapes.activeshapemodel package before the Update on December 21th, 2017.
	 * Run this after updating the corresponding phase files first.
	 * 
	 * @param directory the directory containing the files
	 * @param ccmFileName the model file
	 * @param ccsFileName the score file
	 */
	public static void updateDynamicModel(String directory, String ccmFileName, String ccsFileName, ArrayList<String> phaseScoreFiles)	{
		PcaHotfixScript updater = new PcaHotfixScript(directory, ccmFileName, ccsFileName);
		updater.runForDynamic(phaseScoreFiles);
	}
	
	// ========================================================================================================================
	
	
	// --------------------
	// MEMBERS
	// --------------------
	// Directory properties
	String directory;
	
	// Filename, w/ or w/o file-ending (.ccm / .ccs)
	String ccm;
	String ccs;
	
	// PCA properties
	double[] olEV;
	double[] nuEV;
	
	SimpleMatrix olScores;
	SimpleMatrix nuScores;
	ArrayList<String> names = new ArrayList<>();
	int numSamples;
	
	// --------------------
	// CONSTRUCTOR
	// --------------------
	public PcaHotfixScript(String dir, String ccm, String ccs) {
		this.directory = dir;
		this.ccm = ccm;
		this.ccs = ccs;
	}
	
	
	// --------------------
	// METHODS
	// --------------------
	/**
	 *  Updates the phase specific *.ccm & *.ccs files of the first step PCA.
	 */
	public void runForPhase() {
		// Compose file paths.
		String ccmPath = directory + "\\" + ccm;
		if( ccmPath.substring(ccmPath.length()-4).compareTo(".ccm") != 0 )
			ccmPath += ".ccm";
		
		String ccsPath = directory + "\\" + ccs;
		if( ccsPath.substring(ccsPath.length()-4).compareTo(".ccs") != 0 )
			ccsPath += ".ccs";
		
		// Load old files.
		PcaIO ccmFile = this.loadCCM(ccmPath);
		this.olEV = ccmFile.getEigenValues();
		
		this.olScores = this.parseScores(ccsPath);
		
		// Update the Eigenvalues.
		this.nuEV = this.updateEigenvalues(this.olEV, this.numSamples);
		
		// Update the Scores.
		this.nuScores = this.updateScores(this.olScores, this.olEV, this.nuEV);
		
		// Write the new files.
		this.writeScores(ccsPath, this.names, this.nuScores);
		
		PcaIO ccmFileNew = new PcaIO(ccmFile.getPointDimension(), this.nuEV, ccmFile.getEigenVectors(), ccmFile.getConsensus(),ccmFile.getConnectivity());
		ccmFileNew.setFilename(ccmPath);
		ccmFileNew.writeFile();
	}
	
	/**
	 * Recalculates the *.ccm and *.ccs	files of the second step PCA by concatenating the phase scores of each patient
	 * and recalculating the PCA (including new eigenvectors.
	 * This is necessary because the second step PCA's eigenvectors depend on the updated scores of step 1 pca.
	 * For more detail see Unberath et al., "Open-source 4D statistical shape model of the heart for x-ray projection imaging", Proc. ISBI 2015
	 * 
	 * @param phaseScoreFiles The *.ccs files containing the phase specific scores of step 1 PCA (i.e. phase_0.ccs - phase_9.ccs of ConradCardiacModel).
	 */
	public void runForDynamic(ArrayList<String> phaseScoreFiles) {
		// Compose file paths
		String ccmPath = directory + "\\" + ccm;
		if( ccmPath.substring(ccmPath.length()-4).compareTo(".ccm") != 0 )
			ccmPath += ".ccm";

		String ccsPath = directory + "\\" + ccs;
		if( ccsPath.substring(ccsPath.length()-4).compareTo(".ccs") != 0 )
			ccsPath += ".ccs";

		final int numPhases = phaseScoreFiles.size();
		int[] numPCs = new int[numPhases];
		int totalPC = 0;
		
		// First find out how many total principal components there are
		// (for the ConradCardiacModel phantom these should be 16 each)
		for(int i = 0; i < numPhases; i++){
			String scoreFile = phaseScoreFiles.get(i);
			System.out.println(scoreFile);
			if( scoreFile.substring(scoreFile.length()-4).compareTo(".ccs") != 0 )
				scoreFile += ".ccs";
			SimpleMatrix scores = this.parseScores(this.directory + "\\" + scoreFile);
			numPCs[i] = scores.getRows();
			totalPC += numPCs[i];
		}
		
		// Perform PCA on the concatenated scores of the phases for each patient
		SimpleMatrix scores = new SimpleMatrix(totalPC, this.numSamples);
		int offs = 0;
		for(int i = 0; i < numPhases; i++){
			offs += (i == 0) ? 0 : numPCs[i-1];
			String scoreFile = phaseScoreFiles.get(i);
			if( scoreFile.substring(scoreFile.length()-4).compareTo(".ccs") != 0 )
				scoreFile += ".ccs";
			scores.setSubMatrixValue(offs, 0, parseScores(this.directory + "\\" + scoreFile));			
		}
		
		PCA scorePCA = new PCA(scores, 1);
		scorePCA.variationThreshold = 0.92;
		scorePCA.run();
		System.out.print(scorePCA.numComponents);

		ArrayList< double[] > proj = new ArrayList< double[] >();
		for(int i = 0; i < this.numSamples; i++){
			proj.add(scorePCA.projectTrainingShape(i));
		}
		System.out.println(this.names.size());
		this.writeScores(ccsPath, this.names, proj);

		PcaIO modelWriter = new PcaIO(ccmPath, scorePCA);
		modelWriter.writeFile();
	}	

	// ---------------
	// Eigenvalues
	private double[] updateEigenvalues(double[] old, int numSamples) {
		double[] ev = new double[old.length];
		
		for( int i = 0; i < ev.length; i++ ) {
			ev[i] = Math.pow(old[i], 2) / (numSamples-1.0);
		}
		return ev;
	}
	
	// ---------------
	// Scores
	private SimpleMatrix updateScores(SimpleMatrix old, double[] oldEigenvalues, double[] newEigenvalues) {
		SimpleMatrix scoreMatrix = new SimpleMatrix(old.getRows(),old.getCols());
		
		for( int colIdx = 0; colIdx < old.getCols(); colIdx++ ) {
			for( int rowIdx = 0; rowIdx < old.getRows(); rowIdx++ ) {
				// Old scores are with respect to the old eigenvalue (supposed to be variance, but remember, they were not squared, which why we do all this)
				double value = old.getElement(rowIdx, colIdx) * oldEigenvalues[rowIdx];
				
				// New scores are with respect to the square root of the new eigenvalues (standard deviation).
				value /= Math.sqrt(newEigenvalues[rowIdx]);
				
				scoreMatrix.setElementValue(rowIdx, colIdx, value);	
			}
		}
		
		return scoreMatrix;
	}
		
	
	
	// ---------------
	// IO
	private PcaIO loadCCM(String filepath) {
		PcaIO pio = new PcaIO(filepath);
		try {
			pio.readFile();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return pio;
	}
	
	/**
	 * Writes the scores to file.
	 * @param filepath to the file to be written
	 * @param names of the sample
	 * @param scores to be written
	 */
	private void writeScores(String filepath, ArrayList<String> names, ArrayList<double[]> scores){
		try {
			PrintWriter writer = new PrintWriter(filepath,"UTF-8");
			writer.println("NUM_SAMPLES: " + names.size());
			writer.println("NUM_PRINCIPAL_COMPONENTS: " + scores.get(0).length);
			
			for(int i = 0; i < names.size(); i++){
				String line = names.get(i);
				for( int j = 0; j < scores.get(0).length; j++){
					line += " " + Double.valueOf(scores.get(i)[j]);
				}
				writer.println(line);
			}
			writer.close();			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		}
	}
	
	// ---------------------------------------------
	// Scores handling from BuildCONRADCardiacModel
	// ---------------------------------------------
	/**
	 * Reads the scores for each data set from a .ccs file.
	 * @param filename
	 * @return
	 */
	private SimpleMatrix parseScores(String filename){
		try {
			FileReader fr = new FileReader(filename);
			BufferedReader br = new BufferedReader(fr);
			
			String line = br.readLine();
			StringTokenizer tok = new StringTokenizer(line);
			tok.nextToken(); // skip "NUM_SAMPLES:"
			this.numSamples = Integer.parseInt(tok.nextToken());
			line = br.readLine();
			tok = new StringTokenizer(line);
			tok.nextToken(); // skip "NUM_PRINCIPAL_COMPONENTS:"
			int numPC = Integer.parseInt(tok.nextToken());
			SimpleMatrix m = new SimpleMatrix(numPC, this.numSamples);
			this.names.clear();
			for(int i = 0; i < this.numSamples; i++){
				line = br.readLine();
				tok = new StringTokenizer(line);
				// First token is the folder name, save it.
				this.names.add(tok.nextToken()); 
				for(int j = 0; j < numPC; j++){
					m.setElementValue(j, i, Double.parseDouble(tok.nextToken()));
				}
			}
			
			
			br.close();
			fr.close();
			return m;
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return new SimpleMatrix();
	}
	
	
	/**
	 * Writes the scores to file.
	 * @param filename
	 * @param names
	 * @param scores
	 */
	private void writeScores(String filename, ArrayList<String> names, SimpleMatrix scores){
		try {
			PrintWriter writer = new PrintWriter(filename,"UTF-8");
			writer.println("NUM_SAMPLES: " + names.size());
			writer.println("NUM_PRINCIPAL_COMPONENTS: " + scores.getCol(0).getLen());
			
			for(int i = 0; i < names.size(); i++){
				String line = names.get(i);
				for( int j = 0; j < scores.getCol(0).getLen(); j++){
					line += " " + Double.valueOf(scores.getCol(i).getElement(j));
				}
				writer.println(line);
			}
			writer.close();			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		}
	}
	
	
	// --------------------
	// MAIN
	// --------------------	
	
	public static void main( String[] args ) {
		// =================================
		// ONLY RUN THIS ONCE FOR EACH FILE!
		// =================================
		/*
		// Uncomment and adjust paths.
		String dir = "C:\\Reconstruction\\CONRAD\\data\\CardiacModel\\CardiacModel";
		String[] nums = new String[] {"0","1","2","3","4","5","6","7","8","9"};
		ArrayList<String> phaseScoreFiles = new ArrayList<>(); 
		for( String num : nums ) {
			String ccm = "phase_"+num+".ccm";
			String ccs = "phase_"+num+".ccs";
			PcaHotfixScript.updatePhase(dir,ccm,ccs);
			phaseScoreFiles.add("CardiacModel\\" + ccs);
		}

		String dir2 = "C:\\Reconstruction\\CONRAD\\data\\CardiacModel";
		String ccm2 = "CCmModel.ccm";
		String ccs2 = "CCmExampleScores.ccs";
		PcaHotfixScript.updateDynamicModel(dir2, ccm2, ccs2, phaseScoreFiles);
		 */
	}	
	
}

/*
 * Copyright (C) 2010-2017 Tobias Geimer
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
