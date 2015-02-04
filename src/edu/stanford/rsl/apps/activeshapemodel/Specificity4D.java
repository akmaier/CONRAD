/*
 * Copyright (C) 2010-2014 Mathias Unberath
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
import java.util.Random;
import java.util.StringTokenizer;

import edu.stanford.rsl.conrad.geometry.shapes.activeshapemodels.ActiveShapeModel;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.phantom.asmheart.CONRADCardiacModelConfig;

public class Specificity4D {



	/**
	 * Path to the folder, where the heart model PCA files are stored.
	 */
	public static final String HEART_MODEL_BASE = System.getProperty("user.dir") + "\\data\\CardiacModel\\";;
	/**
	 * Number of phases obtained in the dynamic CT scan.
	 */
	public static final int numPhases = 10;
	/**
	 * Number of components in the whole heart model.
	 */
	public static final int numModelComponents = 6;
	/**
	 * The vertex dimension of the model's vertices.
	 */
	public static final int vertexDimension = 3;

	static final int nTest = 100;



	/**
	 * 
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception{

		int nC = 20;
		double[] m = new double[nC];
		double[] s = new double[nC];

		for(int i = 0; i < nC; i++){
			double[] r = specificityTest(i);
			m[i] = r[0];
			s[i] = r[1];
		}

		write(m,s,nTest);
	}

	/**
	 * 
	 * @param phase
	 */
	private static double[] specificityTest(int numComponents){

		ArrayList<double[]> specificityShapes = new ArrayList<double[]>();

		CONRADCardiacModelConfig info = new CONRADCardiacModelConfig(HEART_MODEL_BASE);
		info.read();		

		ActiveShapeModel pASM = new ActiveShapeModel(HEART_MODEL_BASE + "\\CCmScores.ccm");
		for(int testCase = 0; testCase < nTest; testCase++){
			double[] scores = new double[pASM.numComponents];
			Random rand = new Random();
			for(int i = 0; i < numComponents; i++){
				scores[i] = (rand.nextDouble() - 0.5);
			}
			specificityShapes.add(scores);
		}

		//int numPoints = specificityShapes.get(0).getRows()/3;
		ArrayList<double[]> trainingS = getScores(HEART_MODEL_BASE+"CCmExamples.ccm");

		double[] err = new double[specificityShapes.size()];

		int numPoints = 0;

		ArrayList<ActiveShapeModel> asmList = new ArrayList<ActiveShapeModel>();
		for(int i = 0; i < numPhases; i++){
			String pcaFile = HEART_MODEL_BASE + "\\CardiacModel\\phase_" + i + ".ccm";
			ActiveShapeModel asm = new ActiveShapeModel(pcaFile);
			asmList.add(asm);			
		}

		for(int test = 0; test < specificityShapes.size(); test++){

			System.out.println("Testing instance: " + " at instance "+ Integer.valueOf(test+1) +" of " + specificityShapes.size());
			double max = Double.MAX_VALUE;
			for(int j = 0; j < trainingS.size(); j++){
				System.out.println("\t Comparing to training set " + Integer.valueOf(j));
				double error = 0;
				for(int phase = 0; phase < numPhases; phase++){
					int start = 0;
					for(int i = 0; i < phase; i++){
						start += (i==0) ? 0:info.principalComp[i-1];
					}
					// gets the mode parameters only for the phase specified

					double[] param = pASM.getModel(trainingS.get(j)).getPoints().getCol(0).getSubVec(start, info.principalComp[phase]).copyAsDoubleArray();
					SimpleMatrix current = asmList.get(phase).getModel(param).getPoints();

					double[] paramT = pASM.getModel(specificityShapes.get(test)).getPoints().getCol(0).getSubVec(start, info.principalComp[phase]).copyAsDoubleArray();
					SimpleMatrix testInst = asmList.get(phase).getModel(paramT).getPoints();

					if(test==0 || j == 0 || phase == 0){
						numPoints = current.getRows();
					}

					SimpleMatrix shape = SimpleOperators.subtract(testInst, current);
					double val = 0;
					for(int i = 0; i < numPoints; i++){
						val += shape.getRow(i).normL2();
					}				
					error += val/numPoints;
				}
				max = (error < max) ? error:max;				
			}
			err[test] = max / numPhases;
		}
		double[] ret = new double[]{0,0};
		double mean = 0;
		for(int i = 0; i < nTest; i++){
			mean += err[i];
		}
		mean /= nTest;
		ret[0] = mean;
		double std = 0;
		for(int i = 0; i < nTest; i++){
			std += Math.pow(err[i]-mean,2);
		}
		std = Math.sqrt(std/(nTest-1));
		ret[1] = std;
		return ret;

	}

	private static void write(double[] err,double[] std, int n){
		String outFile = HEART_MODEL_BASE + "\\specificityResults.txt";
		try {
			PrintWriter writer = new PrintWriter(outFile,"UTF-8");
			writer.println("NumShapes: "+ n);
			String line = "";
			for(int i = 0; i < err.length; i++){
				line += " " + Double.valueOf(err[i]);
			}
			String line1 = "";
			for(int i = 0; i < err.length; i++){
				line1 += " " + Double.valueOf(std[i]);
			}
			writer.println(line);
			writer.println(line1);

			writer.close();			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		}
	}

	private static ArrayList<double[]> getScores(String filename){
		ArrayList<double[]> s = new ArrayList<double[]>();
		try {
			FileReader fr = new FileReader(filename);
			BufferedReader br = new BufferedReader(fr);

			String line = br.readLine();
			StringTokenizer tok = new StringTokenizer(line);
			tok.nextToken(); // skip "NUM_SAMPLES:"
			int numSamples = Integer.parseInt(tok.nextToken());
			line = br.readLine();
			tok = new StringTokenizer(line);
			tok.nextToken(); // skip "NUM_PRINCIPAL_COMPONENTS:"
			int numPC = Integer.parseInt(tok.nextToken());
			for(int i = 0; i < numSamples; i++){
				double[] m = new double[numPC];	
				line = br.readLine();
				tok = new StringTokenizer(line);
				tok.nextToken(); // skip "<STUDY_NAME>:"
				for(int j = 0; j < numPC; j++){
					m[j] = Double.parseDouble(tok.nextToken());
				}
				s.add(m);
			}


			br.close();
			fr.close();
			return s;

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return s;
	}

}	






