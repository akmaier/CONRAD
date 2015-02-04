/*
 * Copyright (C) 2010-2014 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.phantom.asmheart;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.StringTokenizer;

import edu.stanford.rsl.apps.activeshapemodel.BuildCONRADCardiacModel.heartComponents;

public class CONRADCardiacModelConfig {

	/**
	 * Directory-name for input or output operation.
	 */
	private String dir;
	/**
	 * Default configuration file name.
	 */
	private static final String CONFIG_FILE_NAME = "cnfg.ccc";
	
	public int numPhases;
	public int[] principalComp;
	public int numAnatComp;
	public int totalVertex;
	public int vertexDim;
	public int[] vertexOffs;
	public int totalTriangles;
	public int[] triangleOffs;
	
	//==========================================================================================
	// METHODS
	//==========================================================================================

	/**
	 * Constructs the object and sets the directory for I/O operations.
	 * @param directory The directory.
	 */
	public CONRADCardiacModelConfig(String directory){
		this.dir = directory;
	}
	
	/**
	 * Writes the configuration to the config file specified in CONFIG_FILE_NAME
	 * @param numPhases
	 * @param principalComp
	 * @param numComponents
	 * @param numVertices
	 * @param vertexDim
	 * @param vertices
	 * @param numTriangles
	 * @param triangles
	 */
	public void write(int numPhases, int[] principalComp, int numComponents, int numVertices,int vertexDim, int[] vertices, int numTriangles, int[] triangles){
		try {
			PrintWriter writer = new PrintWriter(dir + "\\" + CONFIG_FILE_NAME,"UTF-8");
			writer.println("ANATOMICAL_COMPONENTS: " + numComponents);
			String componentS = "COMPONENT_ORDER:";
			for(heartComponents hc : heartComponents.values()){
				componentS += " " + hc.getName();
			}
			writer.println(componentS);
			writer.println("CARDIAC_PHASES: " + numPhases);
			String phasepc = "NUMBER_PRINCIPAL_COMPONENTS_EACH_PHASE:";
			for(int i = 0; i < numPhases; i++){
				phasepc += " " + Integer.valueOf(principalComp[i]);
			}
			writer.println(phasepc);
			writer.println("TOTAL_VERTEX_COUNT: " + numVertices);
			writer.println("VERTEX_DIMENSION: " + vertexDim);
			String vert = "VERTEX_COMPONENT_OFFSETS:";
			for(int i = 0; i < numComponents; i++){
				vert += " " + Integer.valueOf(vertices[i]);
			}
			writer.println(vert);
			writer.println("TOTAL_TRIANGLE_COUNT: " + numTriangles);
			String tri = "TRIANGLE_COMPONENT_OFFSETS:";
			for(int i = 0; i < numComponents; i++){
				tri += " " + Integer.valueOf(triangles[i]);
			}
			writer.println(tri);
			writer.close();			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		}
	}
	
	public void read(){
		try {
			FileReader fr = new FileReader(dir + "\\" + CONFIG_FILE_NAME);
			BufferedReader br = new BufferedReader(fr);
			
			String line = br.readLine();
			StringTokenizer tok = new StringTokenizer(line);
			tok.nextToken(); // skip "ANATOMICAL_COMPONENTS:"
			numAnatComp = Integer.parseInt(tok.nextToken());
			line = br.readLine(); // skip "COMPONENT_ORDER"
			line = br.readLine();
			tok = new StringTokenizer(line);
			tok.nextToken(); // skip "PHASES:"
			numPhases = Integer.parseInt(tok.nextToken());
			line = br.readLine();
			tok = new StringTokenizer(line);
			tok.nextToken(); // skip "NUMBER_PRINCIPAL_COMPONENTS_EACH_PHASE:"
			this.principalComp = new int[numPhases];
			for(int i = 0; i < numPhases; i++){
				principalComp[i] = Integer.parseInt(tok.nextToken());
			}
			line = br.readLine();
			tok = new StringTokenizer(line);
			tok.nextToken(); // skip "TOTAL_VERTEX_COUNT:"
			this.totalVertex = Integer.parseInt(tok.nextToken());
			line = br.readLine();
			tok = new StringTokenizer(line);
			tok.nextToken(); // skip "VERTEX_DIM:"
			this.vertexDim = Integer.parseInt(tok.nextToken());
			line = br.readLine();
			tok = new StringTokenizer(line);
			tok.nextToken(); // skip "VERTEX_COMPONENT_OFFS:"
			this.vertexOffs = new int[numAnatComp];
			for(int k = 0; k < numAnatComp; k++){
				vertexOffs[k] = Integer.parseInt(tok.nextToken());
			}
			line = br.readLine();
			tok = new StringTokenizer(line);
			tok.nextToken(); // skip "TOTAL_TRIANGLE_COUNT:"
			this.totalTriangles = Integer.parseInt(tok.nextToken());
			line = br.readLine();
			tok = new StringTokenizer(line);
			tok.nextToken(); // skip "TRIANGLE_COMPONENT_OFFS:"
			this.triangleOffs = new int[numAnatComp];
			for(int k = 0; k < numAnatComp; k++){
				triangleOffs[k] = Integer.parseInt(tok.nextToken());
			}		
			
			br.close();
			fr.close();
			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	
	
	
	
	
	
	
	
	
	
	
	
	
}
