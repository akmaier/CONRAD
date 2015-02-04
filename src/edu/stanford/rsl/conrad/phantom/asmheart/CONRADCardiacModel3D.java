/*
 * Copyright (C) 2010-2014 Mathias Unberath
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.phantom.asmheart;

import java.io.File;

import edu.stanford.rsl.apps.activeshapemodel.BuildCONRADCardiacModel.heartComponents;
import edu.stanford.rsl.conrad.geometry.General;
import edu.stanford.rsl.conrad.geometry.shapes.activeshapemodels.ActiveShapeModel;
import edu.stanford.rsl.conrad.geometry.shapes.compound.CompoundShape;
import edu.stanford.rsl.conrad.geometry.shapes.mesh.Mesh;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Triangle;
import edu.stanford.rsl.conrad.io.RotTransIO;
import edu.stanford.rsl.conrad.io.VTKMeshIO;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.phantom.AnalyticPhantom;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.UserUtil;

public class CONRADCardiacModel3D extends AnalyticPhantom{

	/**
	 * 
	 */
	private static final long serialVersionUID = 3491173229903879891L;
	boolean DEBUG = false;
	/**
	 * File containing the PcaIO file containing the necessary information for model generation.
	 */
	private String heartBase;
	
	private SimpleMatrix rot;
	private SimpleVector trans;
	
	//==========================================================================================
	// METHODS
	//==========================================================================================
	/**
	 * Configures the heart simulation. 
	 */
	public void configure() throws Exception{
		//heartBase = UserUtil.queryString("Specify path to model directory.", "C:\\Stanford\\CONRAD\\data\\CardiacModel\\");
		heartBase = System.getProperty("user.dir") + "\\data\\CardiacModel\\";
		int phase = UserUtil.queryInt("Specify phase to be modeled [0,9].", 0);
		
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
		int start = 0;
		for(int i = 0; i < phase; i++){
			start += (i==0) ? 0:info.principalComp[i-1];
		}
		// gets the mode parameters only for the phase specified
		double[] param = parameters.getModel(scores).getPoints().getCol(0).getSubVec(start, info.principalComp[phase]).copyAsDoubleArray();
		
		String pcaFile = heartBase + "\\CardiacModel\\phase_" + phase + ".ccm";
		createPhantom(pcaFile, param, info);
	}
	
	/**
	 * Create the phantom using the PCA file and parameters specified via user util. The material assignment could be reworked.
	 * @param pcaFile
	 * @param parameters
	 * @param info
	 */
	private void createPhantom(String pcaFile, double[] parameters, CONRADCardiacModelConfig info){
		ActiveShapeModel asm = new ActiveShapeModel(pcaFile);
		Mesh allComp = asm.getModel(parameters);
		
		int count = 0;
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
			PhysicalObject modelPO = createPhysicalObject(hc.getName(), comp, material);
			add(modelPO);
			count++;
			
			if(DEBUG){
				String inspectionFolder = heartBase + "meshReference/";
				File inspect = new File(inspectionFolder);
				if(!inspect.exists()) inspect.mkdirs();
				String inspectionFile = inspectionFolder + hc.getName() + ".vtk";
				VTKMeshIO wr = new VTKMeshIO(inspectionFile);
				wr.setMesh(comp);
				wr.write();
			}
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
	
	
	@Override
	public String getBibtexCitation() {
		// TODO Auto-generated method stub
		return CONRAD.CONRADBibtex;
	}

	@Override
	public String getMedlineCitation() {
		// TODO Auto-generated method stub
		return CONRAD.CONRADMedline;
	}

	@Override
	public String getName() {
		// TODO Auto-generated method stub
		return "ConradCardiacModel";
	}

}




