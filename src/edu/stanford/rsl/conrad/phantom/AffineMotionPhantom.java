package edu.stanford.rsl.conrad.phantom;

import java.util.ArrayList;
import java.util.Iterator;

import edu.stanford.rsl.conrad.geometry.AbstractShape;
import edu.stanford.rsl.conrad.geometry.motion.AbstractAffineMotionField;
import edu.stanford.rsl.conrad.geometry.motion.AffineMotionField;
import edu.stanford.rsl.conrad.geometry.motion.MotionField;
import edu.stanford.rsl.conrad.geometry.motion.timewarp.CombinedTimeWarper;
import edu.stanford.rsl.conrad.geometry.motion.timewarp.ConstantTimeWarper;
import edu.stanford.rsl.conrad.geometry.motion.timewarp.HarmonicTimeWarper;
import edu.stanford.rsl.conrad.geometry.motion.timewarp.PeriodicTimeWarper;
import edu.stanford.rsl.conrad.geometry.motion.timewarp.SigmoidTimeWarper;
import edu.stanford.rsl.conrad.geometry.motion.timewarp.TimeWarper;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.transforms.AffineTransform;
import edu.stanford.rsl.conrad.geometry.transforms.ComboTransform;
import edu.stanford.rsl.conrad.geometry.transforms.Transform;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.rendering.PrioritizableScene;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.UserUtil;

public class AffineMotionPhantom extends AnalyticPhantom4D {

	private static final long serialVersionUID = -255412548002793134L;

	private AnalyticPhantom staticPhantom = null;
	private AbstractAffineMotionField mf = null;


	@Override
	public void configure() throws Exception {
		staticPhantom = UserUtil.queryPhantom("Select Phantom", "Select Phantom");
		staticPhantom.setBackground(this.getBackgroundMaterial());
		staticPhantom.configure();
		staticPhantom.setConfigured(true);

		SimpleVector rotAxis = new SimpleVector(UserUtil.queryArray("Please provide rotation axis...", new double[]{1,0,0}));
		double rotAngle = UserUtil.queryDouble("Please provide rotation angle (degree)", 0) * Math.PI / 180.0;
		SimpleVector translation = new SimpleVector(UserUtil.queryArray("Please provide translation vector...", new double[]{1,1,1}));	
		mf = new AffineMotionField(new PointND(0,0,0), rotAxis, rotAngle, translation);

		double rep = UserUtil.queryDouble("Please provide No. of harmonic repetitions", 1);
		warper = new CombinedTimeWarper(new HarmonicTimeWarper(rep),  new PeriodicTimeWarper(), new SigmoidTimeWarper());

		max = staticPhantom.getMax();
		min = staticPhantom.getMin();
		this.setConfigured(true);
	}

	public void setStaticAnalyticPhantom(AnalyticPhantom phantom){
		staticPhantom = phantom;
		if (isConfigured() && !staticPhantom.isConfigured()) {
			try {
				staticPhantom.configure();
				staticPhantom.setConfigured(true);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}

	public AnalyticPhantom getStaticAnalyticPhantom(){
		return staticPhantom;
	}



	@Override
	public PointND getPosition(PointND initialPosition, double initialTime,
			double time) {
		double t = warper.warpTime(time);
		return mf.getPosition(initialPosition, initialTime, t);
	}

	@Override
	public ArrayList<PointND> getPositions(PointND initialPosition,
			double initialTime, double... times) {
		ArrayList<PointND> list = new ArrayList<PointND>();
		for (int i=0; i< times.length; i++){
			list.add(getPosition(initialPosition, initialTime, times[i]));
		}
		return list;
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
	public PrioritizableScene getScene(double time) {

		PrioritizableScene scene = new PrioritizableScene();

		double t = warper.warpTime(time);
		//System.out.print(t);
		//System.out.print(", ");

		// apply motion to all physical object in static phantom and add to current scene
		if (staticPhantom instanceof AffineMotionPhantom){
			Iterator<PhysicalObject> objIt = ((AffineMotionPhantom) staticPhantom).getScene(t).iterator();
			Transform tform = mf.getTransform(0, t);
			tform = new AffineTransform(tform.getRotation(3), tform.getTranslation(3));
			while (objIt.hasNext()) {
				PhysicalObject obj = new PhysicalObject((PhysicalObject) objIt.next());
				AbstractShape shape = obj.getShape().clone();
				obj.setShape(shape);
				obj.applyTransform(tform);
				scene.add(obj);
			}
		}
		else{	
			Iterator<PhysicalObject> objIt = staticPhantom.iterator();
			Transform tform = mf.getTransform(0, t);
			tform = new AffineTransform(tform.getRotation(3), tform.getTranslation(3));
			while (objIt.hasNext()) {
				PhysicalObject obj = new PhysicalObject((PhysicalObject) objIt.next());
				AbstractShape shape = obj.getShape().clone();
				obj.setShape(shape);
				obj.applyTransform(tform);
				scene.add(obj);
			}
		}

		scene.setBackground(staticPhantom.getBackgroundMaterial());

		// set scene limits
		scene.setMin(min);
		scene.setMax(max);

		return scene;
	}

	@Override
	public String getName() {
		return "Affine Motion Phantom";
	}

	public void setMotionField(AbstractAffineMotionField mf) {
		this.mf = mf;
	}

	@Override
	public MotionField getMotionField() {
		return mf;
	}

}
/*
 * Copyright (C) 2010-2014 Martin Berger
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
 */
