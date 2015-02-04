/*
 * Copyright (C) 2010-2014 Chris Schwemmer
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.phantom;

import java.util.ArrayList;

import edu.stanford.rsl.conrad.geometry.motion.RotationMotionField;
import edu.stanford.rsl.conrad.geometry.motion.timewarp.HarmonicTimeWarper;
import edu.stanford.rsl.conrad.geometry.motion.timewarp.IdentityTimeWarper;
import edu.stanford.rsl.conrad.geometry.motion.timewarp.PeriodicTimeWarper;
import edu.stanford.rsl.conrad.geometry.motion.timewarp.RestPhaseTimeWarper;
import edu.stanford.rsl.conrad.geometry.motion.timewarp.SigmoidTimeWarper;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Sphere;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;
import edu.stanford.rsl.conrad.rendering.PrioritizableScene;
import edu.stanford.rsl.conrad.utils.CONRAD;
import edu.stanford.rsl.conrad.utils.Configuration;
import edu.stanford.rsl.conrad.utils.DoubleArrayUtil;
import edu.stanford.rsl.conrad.utils.RegKeys;
import edu.stanford.rsl.jpop.utils.UserUtil;

/**
 * A simple 4D phantom similar to a swinging pendulum - spherical version.
 * 
 * @author Chris Schwemmer
 *
 */
public class MovingBallPhantom extends AnalyticPhantom4D {

	private static final long serialVersionUID = 8308889276453752657L;
	
	private RotationMotionField mf;
	private boolean dynamic;
	private PointND refPosition;
	
	private PeriodicTimeWarper periWarper;
	private HarmonicTimeWarper harmoWarper;
	private SigmoidTimeWarper sigWarper;
	private RestPhaseTimeWarper restWarper;
	
	private double radius, angle;
	private PointND initialCentre;

	public MovingBallPhantom(){
	}
	
	public void configure() throws Exception {
		radius = UserUtil.queryDouble("Radius of sphere", 10);
		angle = UserUtil.queryDouble("Angle of deflection (degrees)", 20) * Math.PI / 180.0;
		double accFac = UserUtil.queryDouble("Acceleration factor", 4.0);
		double rp = UserUtil.queryDouble("Rest phase (%)", 20);
		double heartCycles = UserUtil.queryDouble("Number of \"heart beats\" in Scene (acquisition time * heart rate):", 4.0);
		
		Configuration config = Configuration.getGlobalConfiguration();
		int dimz = config.getGeometry().getProjectionStackSize() * config.getNumSweeps();
		double [] heartStates = new double [dimz];
		for (int i = 0; i < dimz; i++){
			heartStates[i]= (i*heartCycles) / (dimz);
		}
		config.getRegistry().put(RegKeys.HEART_PHASES, DoubleArrayUtil.toString(heartStates));
		
		dynamic = UserUtil.queryBoolean("Create dynamic scene?");
		
		mf = new RotationMotionField(new PointND(0, 0, 0), new SimpleVector(0, 1, 0), 2.0 * angle);
		
		harmoWarper = new HarmonicTimeWarper(heartCycles);
		sigWarper = new SigmoidTimeWarper();
		periWarper = new PeriodicTimeWarper();
		restWarper = new RestPhaseTimeWarper();
		this.warper = new IdentityTimeWarper();
		
		sigWarper.setAcc(accFac);
		restWarper.setRestPhase(rp / 100);
		
		PointND range = mf.getPosition(new PointND(0, 0, 5.0 * radius), 0, 0.5);
		
		initialCentre = new PointND(-range.get(0), range.get(1), range.get(2));
		PointND endPos = new PointND(range.get(0), range.get(1), range.get(2));
		
		double maxExt = Math.max(endPos.get(0), endPos.get(2));
		
		this.max = new PointND(1.7 * maxExt, 1.7 * maxExt, 1.7 * maxExt);
		this.min = new PointND(-1.7 * maxExt, -1.7 * maxExt, -1.7 * maxExt);
		
		if (!dynamic) {
			double refHeartPhase = UserUtil.queryDouble("Reference heart phase", 0.75);
			refPosition = mf.getPosition(initialCentre, 0, sigWarper.warpTime(periWarper.warpTime(restWarper.warpTime(refHeartPhase))));
		}
	}
	
	@Override
	public PointND getPosition(PointND initialPosition, double initialTime,
			double time) {
		double nT = sigWarper.warpTime(periWarper.warpTime(restWarper.warpTime(harmoWarper.warpTime(time))));
		System.out.print(nT);
		System.out.print(", ");
		return mf.getPosition(initialPosition, initialTime, nT);
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
		PhysicalObject obj = new PhysicalObject();
		obj.setMaterial(MaterialsDB.getMaterial("I"));
		obj.setNameString("Contrast Bolus");
		
		System.out.print(time);
		System.out.print(", ");
		PointND centre;
		if (dynamic)
			centre = getPosition(initialCentre, 0, time);
		else
			centre = refPosition.clone();
		System.out.println(centre.toString());
		
		scene.setMin(min);
		scene.setMax(max);
		
		Sphere sp = new Sphere(radius, centre);
		sp.setName("Moving Ball");
		obj.setShape(sp);
		scene.add(obj);
		return scene;
	}

	@Override
	public String getName() {
		return "Moving Ball Phantom";
	}

}
