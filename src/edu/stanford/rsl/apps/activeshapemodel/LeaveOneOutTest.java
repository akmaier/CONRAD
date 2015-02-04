/*
 * Copyright (C) 2014 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.apps.activeshapemodel;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;

import edu.stanford.rsl.apps.activeshapemodel.BuildCONRADCardiacModel.heartComponents;
import edu.stanford.rsl.conrad.geometry.shapes.activeshapemodels.ActiveShapeModel;
import edu.stanford.rsl.conrad.geometry.shapes.activeshapemodels.GPA;
import edu.stanford.rsl.conrad.geometry.shapes.activeshapemodels.PCA;
import edu.stanford.rsl.conrad.geometry.shapes.mesh.DataMatrix;
import edu.stanford.rsl.conrad.geometry.shapes.mesh.Mesh;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.utils.UserUtil;

public class LeaveOneOutTest {

	/**
	 * Path to model data, i.e. the meshes stored in folders corresponding to their heart phase.
	 * The naming convention needs to follow:
	 * .../study_id/.../phase_#/meshname.vtk
	 */
	public static final File DATA_PATH = new File("E:\\_uni_\\Masterthesis\\Data\\");
	/**
	 * Path to the folder, where the heart model PCA files are stored.
	 */
	public static final String HEART_MODEL_BASE = "C:\\research\\data\\Test\\HeartBase\\";
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
	/**
	 * Variation threshold.
	 */
	static double variationTh = 0.9;
	/**
	 * Keyword indicating folders containing phases.
	 */
	private static final String PHASE_KEY = "phase_";
	/**
	 * Keyword indicating folders containing phase-folders that have already been segmented and contain meshes.
	 */
	private static final String ANALYSIS_KEY = "analysis";
	
	private static int[] vertexOffs = new int[numModelComponents];
	private static int totalVertices;
	private static int[] triangleOffs = new int[numModelComponents];
	private static int totalTriangles;
	private static SimpleMatrix errors;
	//==========================================================================================
	// METHODS
	//==========================================================================================

	public static void main(String[] args) throws Exception{
		variationTh = UserUtil.queryDouble("Select variance threshold for principal component dimensionality reduction:", 0.9);
		
		// initialize offset array for easier use later on
		int c = 0;
		totalVertices = 0;
		totalTriangles = 0;
		for(heartComponents hc : heartComponents.values()){
			vertexOffs[c] = totalVertices;
			triangleOffs[c] = totalTriangles;			
			totalVertices += hc.getNumVertices();
			totalTriangles += hc.getNumTriangles();
			c++;
		}
		
		// get all valid folders
		ArrayList<String> folders = getValidFolders();
		// run GPA and PCA
		performLeaveOneOut(folders);
	}
	
	
	/**
	 * This method performs leave-one-out tests on all heart model components. It uses data from all folders passed to it via the folders list.
	 * @param folders The folders being used for input.
	 */
	private static void performLeaveOneOut(ArrayList<String> folders){
		
		errors = new SimpleMatrix(folders.size(), numPhases);
		
		for(int test = 0; test < folders.size(); test++){
			// create ONE mesh object at each phase for each training set and perform GPA and PCA on ALL heart components at the same time
			for(int i = 0; i < numPhases; i++){
				GPA populationGPA = new GPA(folders.size()-1);
				SimpleMatrix leftOut = new SimpleMatrix(totalVertices,3);
				for(int j = 0; j < folders.size(); j++){
					int componentCount = 0;
					int gpaCount = 0;
					if(j != test){
						SimpleMatrix currentVert = new SimpleMatrix(totalVertices,3);
						if(j == 0){
							SimpleMatrix currentTriangles = new SimpleMatrix(totalTriangles,3);
							for(heartComponents hc : heartComponents.values()){
								String currentComponent = "wrp_" + hc.getName() + ".vtk";
								String currentFile = folders.get(j) + "\\" + PHASE_KEY + i + "\\" + currentComponent;
								Mesh currentMesh = new Mesh(currentFile);
								currentVert.setSubMatrixValue(vertexOffs[componentCount], 0, currentMesh.getPoints());
								currentTriangles.setSubMatrixValue(triangleOffs[componentCount], 0, currentMesh.getConnectivity());
								componentCount++;
							}
							populationGPA.setConnectivity(currentTriangles);
						}else{
							for(heartComponents hc : heartComponents.values()){
								String currentComponent = "wrp_" + hc.getName() + ".vtk";
								String currentFile = folders.get(j) + "\\" + PHASE_KEY + i + "\\" + currentComponent;
								Mesh currentMesh = new Mesh(currentFile);
								currentVert.setSubMatrixValue(vertexOffs[componentCount], 0, currentMesh.getPoints());
								componentCount++;
							}							
						}
						populationGPA.addElement(gpaCount, currentVert);
						gpaCount++;
					}else{
						for(heartComponents hc : heartComponents.values()){
							String currentComponent = "wrp_" + hc.getName() + ".vtk";
							String currentFile = folders.get(j) + "\\" + PHASE_KEY + i + "\\" + currentComponent;
							Mesh currentMesh = new Mesh(currentFile);
							leftOut.setSubMatrixValue(vertexOffs[componentCount], 0, currentMesh.getPoints());
							componentCount++;
						}
					}
				}
				populationGPA.runGPA();
				
				// create PCA file for each phase
				PCA populationPCA = new PCA(new DataMatrix(populationGPA));
				populationPCA.variationThreshold = variationTh;
				populationPCA.run();
				
				ActiveShapeModel asm = new ActiveShapeModel(populationPCA);
				asm.projectShape(leftOut);
				errors.setElementValue(test, i, asm.getFittingError());
				
				System.out.println("Error of set "+test+" at phase: " + i+" is: " + errors.getElement(test, i));
				System.out.println("______________________________________");
				
			}
		}
		
		double avg = 0;
		double std = 0;
		for(int i = 0; i < folders.size(); i++){
			for(int j = 0; j < numPhases; j++){
				avg += errors.getElement(i, j);
			}
		}
		avg /= (folders.size() * numPhases);
		
		for(int i = 0; i < folders.size(); i++){
			for(int j = 0; j < numPhases; j++){
				std += Math.pow(errors.getElement(i, j)-avg,2);
			}
		}
		std = Math.sqrt(std/(folders.size() * numPhases -1));
		
		writeResultsToFile(folders, errors, avg, std);
		System.out.println("Done.");
	}	
	
	private static void writeResultsToFile(ArrayList<String> folders, SimpleMatrix errors, double avg, double std){
		String outFile = HEART_MODEL_BASE + "\\leaveOneOutResults.txt";
		try {
			PrintWriter writer = new PrintWriter(outFile,"UTF-8");
			writer.println("Average Error: "+ avg);
			writer.println("Standard deviation: "+ std);
			for(int i = 0; i < folders.size(); i++){
				String line = folders.get(i);
				for( int j = 0; j < errors.getCols(); j++){
					line += " " + Double.valueOf(errors.getElement(i, j));
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
	
	/**
	 * Searches all sub-directories for the keyword ANALYSIS_KEY. This keyword indicates, that registration has been performed and meshes exist and 
	 * are assumed to be stored in this folder.
	 * @return An ArrayList containing the folders.
	 */
	private static ArrayList<String> getValidFolders(){
		String[] fl = DATA_PATH.list();
		ArrayList<String> validFolders = new ArrayList<String>();
		
		for(int i = 0; i < fl.length; i++){
			File fi = new File(DATA_PATH + "\\" + fl[i]);
			if(fi.isDirectory()){
				String[] list = fi.list();
				for(int j = 0; j < list.length; j++){
					if(list[j].contains(ANALYSIS_KEY)){
						validFolders.add(DATA_PATH + "\\" + fl[i] + "\\" + list[j]);
					}
				}
			}
		}
		return checkIfAllFoldersValid(validFolders);
	}
	
	/**
	 * This method checks if all directories in the ArrayList contain the necessary amount of phases.
	 * @param f The ArrayList of directories to be checked.
	 * @return An ArrayList containing only the valid directories.
	 */
	private static ArrayList<String> checkIfAllFoldersValid(ArrayList<String> f){
		ArrayList<String> fl = new ArrayList<String>();
		for(int i = 0; i < f.size(); i++){
			File file = new File(f.get(i));
			String[] list = file.list();
			int[] count = new int[numPhases];
			int cnt = 0;
			for(int j = 0; j < list.length; j++){
				File check = new File(f.get(i) + "\\" + list[j]);
				if(check.isDirectory() && check.getName().contains(PHASE_KEY)){
					int strPos = check.getName().indexOf(PHASE_KEY) + PHASE_KEY.length();
					int idx = Integer.valueOf(check.getName().substring(strPos));
					count[idx] = 1;
				}
			}
			for(int k = 0; k < numPhases; k++){
				cnt += count[k];
			}
			if(cnt == numPhases){
				fl.add(f.get(i));
			}else{
				System.out.println("Missing files in dataset: " + f.get(i));
			}
		}
		return fl;
	}
}
/*
 * Copyright (C) 2010-2014 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/

