/*
 * Copyright (C) 2010-2014 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.phantom.asmheart;

import java.io.File;
import java.util.ArrayList;

import edu.stanford.rsl.apps.activeshapemodel.BuildCONRADCardiacModel.heartComponents;
import edu.stanford.rsl.conrad.geometry.General;
import edu.stanford.rsl.conrad.geometry.shapes.activeshapemodels.ActiveShapeModel;
import edu.stanford.rsl.conrad.geometry.shapes.compound.CompoundShape;
import edu.stanford.rsl.conrad.geometry.shapes.mesh.Mesh;
import edu.stanford.rsl.conrad.geometry.shapes.mesh.Mesh4D;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Triangle;
import edu.stanford.rsl.conrad.io.RotTransIO;
import edu.stanford.rsl.conrad.io.VTKMeshIO;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.phantom.AnalyticPhantom4D;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;
import edu.stanford.rsl.conrad.rendering.PrioritizableScene;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.UserUtil;

public class CONRADCardiacModel4D extends AnalyticPhantom4D{

	private static final boolean DEBUG = false;
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 3491173229903879891L;
	/**
	 * File containing the PcaIO file containing the necessary information for model generation.
	 */
	private String heartBase;
	/**
	 * The number of phases in the model.
	 */
	private static final int numPhases = 10;
	/**
	 * Number of heart-beats in the scene.
	 */
	private int heartBeats;
	/**
	 * The splines describing the mesh's variation.
	 */
	private ArrayList<Mesh4D> splines;
	
	private SimpleMatrix rot;
	private SimpleVector trans;
	
	//==========================================================================================
	// METHODS
	//==========================================================================================
	
	public CONRADCardiacModel4D(){
	}
	
	/**
	 * Configures the heart simulation. Currently uses learning shapes as heart description. This will be changed later using the HeartParameterLUT class. 
	 */
	public void configure() throws Exception{
		//heartBase = UserUtil.queryString("Specify path to model data file.", "C:\\Stanford\\CONRAD\\data\\CardiacModel\\");
		heartBase = System.getProperty("user.dir") + "\\data\\CardiacModel\\";
		
		// perform sanity check
		File scF = new File(heartBase);
		File[] listOfFiles = scF.listFiles();
		if(listOfFiles.length < 7){
			throw new Exception("CONRADCardiacModel files are not found at "+heartBase+".\n Please download them from: https://www5.cs.fau.de/conrad/data/heart-model/");
		}
		
		heartBeats = UserUtil.queryInt("Number of heart beats in this scene:", 1);
		
		boolean rotTrans = UserUtil.queryBoolean("Apply rotatation and translation?");
		if(!rotTrans){
			new SimpleMatrix();
			this.rot = SimpleMatrix.I_3;
			this.trans = new SimpleVector(3);
		}else{
			String rtfile = UserUtil.queryString("Specify file containing rotation and translation:", heartBase + "rotTrans.txt");
			RotTransIO rtinput = new RotTransIO(rtfile);
			this.rot = rtinput.getRotation();
			this.trans = rtinput.getTranslation();
		}
		
		this.max = new PointND(0, 0, 0);
		this.min = new PointND(0, 0, 0);
		
		// read config file for offsets etc
		CONRADCardiacModelConfig info = new CONRADCardiacModelConfig(heartBase);
		info.read();
		// read the PCs of all phases
		ActiveShapeModel parameters = new ActiveShapeModel(heartBase + "\\CCmScores.ccm");
		double[] scores;
		boolean predefined = UserUtil.queryBoolean("Use predefined model?");
		if(predefined){
			Object tobj = UserUtil.chooseObject("Choose heart to be simulated:", "Predefined models:", PredefinedModels.getList(), PredefinedModels.getList()[0]);
			scores = PredefinedModels.getValue(tobj.toString());
		}else{
			scores = UserUtil.queryArray("Specify model parameters: ", new double[parameters.numComponents]);
			// the scores are defined with respect to variance but we want to have them with respect to standard deviation therefore divide by 
			// sqrt of variance
			for(int i = 0; i < parameters.numComponents; i++){
				scores[i] /= Math.sqrt(parameters.getEigenvalues()[i]);
			}
		}
		
		SimpleVector paramVec = parameters.getModel(scores).getPoints().getCol(0);
		
		// for all components
		// loop through all phases and create splines describing the motion for each vertex
		System.out.println("Starting model generation.\n");
		System.out.println("__________________________________");
		System.out.println("Calculating phantom at each phase.\n");
		this.splines = new ArrayList<Mesh4D>();
		
		for(int i = 0; i < numPhases; i++){
			int start = 0;
			for(int j = 0; j < i; j++){
				start += (j==0) ? 0:info.principalComp[j-1];
			}
			double[] param = paramVec.getSubVec(start, info.principalComp[i]).copyAsDoubleArray();
			createPhantom(i, param, info, splines);
		}
		// calculate splines
		System.out.println("Fitting temporal splines.\n");
		for(int i = 0; i < info.numAnatComp; i++){
			splines.get(i).calculateSplines();
		}
		System.out.println("__________________________________");
		setConfigured(true);
		System.out.println("Configuration done.\n");
	}
		
	/**
	 * Creates the meshes of one single phase and adds it to the ArrayList of 4D meshes.
	 * @param phase
	 * @param parameters
	 * @param info
	 * @param splines
	 */
	private void createPhantom(int phase, double[] parameters, CONRADCardiacModelConfig info, ArrayList<Mesh4D> splines){
		String pcaFile = heartBase + "\\CardiacModel\\phase_" + phase + ".ccm";
		ActiveShapeModel asm = new ActiveShapeModel(pcaFile);
		Mesh allComp = asm.getModel(parameters);
		
		if(phase == 0){
			for(int i = 0; i < info.numAnatComp; i++){
				splines.add(new Mesh4D());
			}
		}
		
		int count = 0;
		for(heartComponents hc : heartComponents.values()){
			Mesh comp = new Mesh();
			SimpleMatrix pts = allComp.getPoints().getSubMatrix(info.vertexOffs[count], 0, hc.getNumVertices(), info.vertexDim);
			// rotate and translate points
			for(int i = 0; i < pts.getRows(); i++){
				SimpleVector row = SimpleOperators.multiply(rot, pts.getRow(i));
				row.add(trans);
				pts.setRowValue(i, row);
			}
			comp.setPoints(pts);
			comp.setConnectivity(allComp.getConnectivity().getSubMatrix(info.triangleOffs[count], 0, hc.getNumTriangles(), 3));
			splines.get(count).addMesh(comp);
			count++;
			
		}
	}
		
	/**
	 * Creates a physical object from a mesh for rendering. 
	 * @param m The mesh to be converted.
	 * @param material The material of the physical object.
	 * @return The mesh as physical object with specified material.
	 */
	private PhysicalObject createPhysicalObject(String nameStr, Mesh m, String material){
		PhysicalObject po = new PhysicalObject();
		po.setMaterial(MaterialsDB.getMaterialWithName(material));
		po.setShape(createCompoundShape(m));
		po.setNameString(nameStr);		
		return po;
	}
	
	/**
	 * Creates a CompoundShape consisting of all faces of the input mesh. The mesh has to be a triangular mesh.
	 * @param m The triangular mesh.
	 * @return The compound shape consisting of all faces of the mesh represented as triangles.
	 */
	private CompoundShape createCompoundShape(Mesh m){
		CompoundShape cs = new CompoundShape();
		SimpleMatrix vertices = m.getPoints();
		SimpleMatrix faces = m.getConnectivity();
		
		for(int i = 0; i < m.numConnections; i++){
			addTriangleAtIndex(cs, vertices, faces, i);
		}		
		return cs;
	}
	
	/**
	 * Constructs the triangle corresponding to the i-th face in a mesh given the connectivity information fcs and the vertices vtc and adds it to 
	 * the CompoundShape.
	 * @param vtc The vertices of the mesh.
	 * @param fcs The faces of the mesh, i.e connectivity information.
	 * @param i The index of the face to be constructed.
	 */
	private void addTriangleAtIndex(CompoundShape cs, SimpleMatrix vtc, SimpleMatrix fcs, int i){
		SimpleVector face = fcs.getRow(i);
		
		SimpleVector dirU = vtc.getRow((int)face.getElement(1));
		dirU.subtract(vtc.getRow((int)face.getElement(0)));
		double l2 = dirU.normL2();
		SimpleVector dirV = vtc.getRow((int)face.getElement(2));
		dirV.subtract(vtc.getRow((int)face.getElement(0)));
		if(dirV.normL2() < l2){
			l2 = dirV.normL2();
		}
		double nN = General.crossProduct(dirU.normalizedL2(), dirV.normalizedL2()).normL2();
		
		if(l2 < Math.sqrt(CONRAD.DOUBLE_EPSILON) || nN < Math.sqrt(CONRAD.DOUBLE_EPSILON)){
		}else{
			Triangle t =  new Triangle(	new PointND(vtc.getRow((int)face.getElement(0))),
										new PointND(vtc.getRow((int)face.getElement(1))),
										new PointND(vtc.getRow((int)face.getElement(2))));
		cs.add(t);
		}
	}

	/**
	 * Calculate the current state of the heart at time t using spline evaluation of all vertices in all heart components.
	 */
	@Override
	public PrioritizableScene getScene(double time) {


		
		double t = (time*heartBeats)%1;
		
		System.out.println("Evaluating time step: " + time);
		
		PrioritizableScene scene = new PrioritizableScene();
		
		int componentCount = 0;
		for(heartComponents hc : heartComponents.values()){
			String material;
			if(hc.getName().contains("myocardium")){
				material = "Heart";
			}else if(hc.getName().contains("aorta")){
				material = "Aorta";
			}else if(hc.getName().contains("leftVentricle")){
				material = "CoronaryArtery";
			}else{
				material = "Blood";
			}
			
			Mesh m = splines.get(componentCount).evaluateLinearInterpolation(t);
			//Mesh m = splines.get(componentCount).evaluateSplines(t);
			
			if(DEBUG){
				String inspectionFolder = heartBase + "meshReference/" + t + "/";
				File inspect = new File(inspectionFolder);
				if(!inspect.exists()) inspect.mkdirs();
				String inspectionFile = inspectionFolder + hc.getName() + ".vtk";
				VTKMeshIO wr = new VTKMeshIO(inspectionFile);
				wr.setMesh(m);
				wr.write();
			}
			
			PhysicalObject po = createPhysicalObject(hc.toString(), m, material);
			scene.add(po);
			
			componentCount++;
		}
		
		
		scene.setMax(max);
		scene.setMin(min);
		
		return scene;
	}
	
	
	
	@Override
	public PointND getPosition(PointND initialPosition, double initialTime,
			double time) {
		return null;
	}

	@Override
	public ArrayList<PointND> getPositions(PointND initialPosition,
			double initialTime, double... times) {
		return null;
	}

	@Override
	public String getBibtexCitation() {
		return CONRAD.CONRADBibtex;
	}

	@Override
	public String getMedlineCitation() {
		return CONRAD.CONRADMedline;
	}

	@Override
	public String getName() {
		return "ConradCardiacModel4D";
	}


}

