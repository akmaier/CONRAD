/*
 * Copyright (C) 2010-2014 Andreas Maier
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.phantom;

import edu.stanford.rsl.conrad.geometry.General;
import edu.stanford.rsl.conrad.geometry.Rotations;
import edu.stanford.rsl.conrad.geometry.shapes.simple.Ellipsoid;
import edu.stanford.rsl.conrad.geometry.transforms.ComboTransform;
import edu.stanford.rsl.conrad.geometry.transforms.ScaleRotate;
import edu.stanford.rsl.conrad.geometry.transforms.Translation;
import edu.stanford.rsl.conrad.numerics.SimpleMatrix;
import edu.stanford.rsl.conrad.numerics.SimpleVector;
import edu.stanford.rsl.conrad.physics.PhysicalObject;
import edu.stanford.rsl.conrad.physics.materials.Material;
import edu.stanford.rsl.conrad.physics.materials.database.MaterialsDB;

/**
 * 3-D definition of the Shepp-Logan Phantom. The center slice is identical to the original publication
 * of Shepp and Logan. The third dimension was created by the taking the minimal extent of the x- and y-directions.
 * <br><br><b>Be careful when using this phantom and have a look at <a href=http://lists.fau.de/pipermail/project-conrad/2014-October/000054.html>this comment</a> on the CONRAD mailing list</b>
 * @author Happy Coding Seminar
 *
 */
public class SheppLoganPhantom extends AnalyticPhantom {

	/**
	 * 
	 */
	private static final long serialVersionUID = 5781085850910993618L;

	@Override
	public String getBibtexCitation() {
		String bibtex = "@Article{Shepp74-TFR,\n" +
		"  author = {{Shepp}, L. A. and {Logan}, B. F.},\n" +
		"  title = {{The Fourier reconstruction of a head section}},\n" +
		"  journal = {IEEE Transactions on Nuclear Science},\n" +
		"  volume = {21},\n" +
		"  pages = {21-43},\n" +
		"  year = {1974}\n" +
		"}";
		return bibtex;
	}

	@Override
	public String getMedlineCitation() {
		return "Shepp LA, Logan BF. The Fourier reconstruction of a head section. IEEE Transactions on nuclear science 21:21-43. 1974";
	}


	@Override
	public String getName() {
		return "Shepp Logan Phantom (Adapted)";
	}

	public SheppLoganPhantom() {
		float scalingFactor = 100;
		ScaleRotate scaling = new ScaleRotate(new SimpleMatrix("[[1 0 0];[0 1 0];[0 0 1]]").multipliedBy(scalingFactor));
		// Body of the phantom.
		Ellipsoid a = new Ellipsoid(.69, .92, .69);
		Ellipsoid b = new Ellipsoid(.6624, .874, .6624);
		Ellipsoid c = new Ellipsoid(.11, .31, .11);
		Ellipsoid d = new Ellipsoid(.16, .41, .16);
		Ellipsoid e = new Ellipsoid(.21, .25, .21);
		Ellipsoid f = new Ellipsoid(.046, .046, .046);
		Ellipsoid g = new Ellipsoid(.046, .046, .046);
		Ellipsoid h = new Ellipsoid(.046, .023, .023);
		Ellipsoid i = new Ellipsoid(.023, .023, .023);
		Ellipsoid j = new Ellipsoid(.023, .046, .023);

		a.applyTransform(scaling);
		b.applyTransform(new ComboTransform(new Translation(new SimpleVector(0, -.0184, 0)), scaling));
		c.applyTransform(new ComboTransform(new ScaleRotate(Rotations.createBasicZRotationMatrix(General.toRadians(-18))),
											new Translation(new SimpleVector(.22, 0, 0)),
											scaling));
		d.applyTransform(new ComboTransform(new ScaleRotate(Rotations.createBasicZRotationMatrix(General.toRadians(18))),
											new Translation(new SimpleVector(-.22, 0, 0)),
											scaling));
		e.applyTransform(new ComboTransform(new Translation(new SimpleVector(0, .35, 0)), scaling));
		f.applyTransform(new ComboTransform(new Translation(new SimpleVector(0, .1, 0)), scaling));
		g.applyTransform(new ComboTransform(new Translation(new SimpleVector(0, -.1, 0)), scaling));
		h.applyTransform(new ComboTransform(new Translation(new SimpleVector(-.08, -.605, 0)), scaling));
		i.applyTransform(new ComboTransform(new Translation(new SimpleVector(0, -.605, 0)), scaling));
		j.applyTransform(new ComboTransform(new Translation(new SimpleVector(.06, -.605, 0)), scaling));

		PhysicalObject po;
		Material water = MaterialsDB.getMaterial("water");
		po = new PhysicalObject();
		Material mat = (Material) water.clone();
		mat.setDensity(2.0);
		po.setMaterial(mat);
		po.setShape(a);
		add(po);
		
		po = new PhysicalObject();
		mat = (Material) water.clone();
		mat.setDensity(2.0-.98);
		po.setMaterial(mat); // D = 1.95
		po.setShape(b);
		add(po);
		
		po = new PhysicalObject();
		mat = (Material) water.clone();
		mat.setDensity(1.0-.02);
		po.setMaterial(mat); // D = 1.95
		po.setShape(c);
		add(po);
		
		po = new PhysicalObject();
		mat = (Material) water.clone();
		mat.setDensity(1.0-.02);
		po.setMaterial(mat); // D = 1.95
		po.setShape(d);
		add(po);
		
		po = new PhysicalObject();
		mat = (Material) water.clone();
		mat.setDensity(1.01);
		po.setMaterial(mat); // D = 1.95
		po.setShape(e);
		add(po);

		po = new PhysicalObject();
		mat = (Material) water.clone();
		mat.setDensity(1.01);
		po.setMaterial(mat); // D = 1.95
		po.setShape(f);
		add(po);

		po = new PhysicalObject();
		mat = (Material) water.clone();
		mat.setDensity(1.01);
		po.setMaterial(mat); // D = 1.95
		po.setShape(g);
		add(po);

		po = new PhysicalObject();
		mat = (Material) water.clone();
		mat.setDensity(1.01);
		po.setMaterial(mat); // D = 1.95
		po.setShape(h);
		add(po);

		po = new PhysicalObject();
		mat = (Material) water.clone();
		mat.setDensity(1.01);
		po.setMaterial(mat); // D = 1.95
		po.setShape(i);
		add(po);

		po = new PhysicalObject();
		mat = (Material) water.clone();
		mat.setDensity(1.01);
		po.setMaterial(mat); // D = 1.95
		po.setShape(j);
		add(po);

	}
}
