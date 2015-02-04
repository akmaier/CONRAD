package edu.stanford.rsl.conrad.geometry.shapes.compound;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;

import edu.stanford.rsl.conrad.geometry.AbstractCurve;
import edu.stanford.rsl.conrad.geometry.AbstractShape;
import edu.stanford.rsl.conrad.geometry.shapes.simple.PointND;
import edu.stanford.rsl.conrad.geometry.transforms.Transform;
import edu.stanford.rsl.conrad.numerics.SimpleOperators;

public class LinearOctree extends CompoundShape {

	/**
	 * 
	 */
	private static final long serialVersionUID = 8039686125389239903L;
	protected CompoundShape octant1 = new CompoundShape();
	protected CompoundShape octant2 = new CompoundShape();
	protected CompoundShape octant3 = new CompoundShape();
	protected CompoundShape octant4 = new CompoundShape();
	protected CompoundShape octant5 = new CompoundShape();
	protected CompoundShape octant6 = new CompoundShape();
	protected CompoundShape octant7 = new CompoundShape();
	protected CompoundShape octant8 = new CompoundShape();
	private PointND center;
	private int dimension = -1;
	
	public LinearOctree(PointND center){
		this.center = center;
	}

	public LinearOctree (PointND min, PointND max){
		this(new PointND(SimpleOperators.add(min.getAbstractVector(), max.getAbstractVector()).dividedBy(2)));
		this.min = min;
		this.max = max;

	}
	
	public LinearOctree (PointND min, PointND max, double random){
		this(new PointND(SimpleOperators.add(min.getAbstractVector(), SimpleOperators.subtract(min.getAbstractVector(), max.getAbstractVector()).multipliedBy(random))));
		this.min = min;
		this.max = max;

	}

	public LinearOctree (PointND min, PointND max, PointND center){
		this(center);
		this.min = min;
		this.max = max;
	}

	public LinearOctree(LinearOctree lo){
		super(lo);
		dimension = lo.dimension;
		center = (lo.center!=null) ? lo.center.clone() : null;
		octant1 = (lo.octant1!=null) ? (CompoundShape)lo.octant1.clone() : null;
		octant2 = (lo.octant2!=null) ? (CompoundShape)lo.octant2.clone() : null;
		octant3 = (lo.octant3!=null) ? (CompoundShape)lo.octant3.clone() : null;
		octant4 = (lo.octant4!=null) ? (CompoundShape)lo.octant4.clone() : null;
		octant5 = (lo.octant5!=null) ? (CompoundShape)lo.octant5.clone() : null;
		octant6 = (lo.octant6!=null) ? (CompoundShape)lo.octant6.clone() : null;
		octant7 = (lo.octant7!=null) ? (CompoundShape)lo.octant7.clone() : null;
		octant8 = (lo.octant8!=null) ? (CompoundShape)lo.octant8.clone() : null;
	}
	
	@Override
	protected synchronized void init(){
		if (dirty){
			min = new PointND(Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE);
			max = new PointND(-Double.MAX_VALUE, -Double.MAX_VALUE, -Double.MAX_VALUE);
			for (AbstractShape s: createToDoList()){
				min.updateIfLower(s.getMin());
				max.updateIfHigher(s.getMax());
			}
			super.generateBoundingPlanes();
			dirty = false;
		}
	}

	@Override
	public boolean add(AbstractShape shape){
		boolean revan = false;
		if (shape.isBounded()){
			if (dimension == -1) {
				dimension = shape.getDimension();
			}
			if (shape.getDimension() != dimension){
				throw new RuntimeException("Dimensions do not match. My dimension is " + dimension + ". You added a shape of dimension " + shape.getDimension() + "!");
			}
			int dim [] = new int[3];
			for (int i = 0; i < 3; i++){
				if (shape.getMin().get(i) > center.get(i)){
					// completely right from the center
					dim[i] = 2;
				} else {
					if ((shape.getMax().get(i) >  center.get(i))){
						// overlaps the center
						dim[i] = 1;
					} else {
						// completely left from the center;
						dim[i] = 0;
					}
				}
			}
			//System.out.println(center + " " + shape.getMin() + " " + shape.getMax() + " " + dim[0] + dim[1] + dim[2]);
			if (testOctant(1, dim)) revan = octant1.add(shape);
			else if (testOctant(2, dim)) revan = octant2.add(shape);
			else if (testOctant(3, dim)) revan = octant3.add(shape);
			else if (testOctant(4, dim)) revan = octant4.add(shape);
			else if (testOctant(5, dim)) revan = octant5.add(shape);
			else if (testOctant(6, dim)) revan = octant6.add(shape);
			else if (testOctant(7, dim)) revan = octant7.add(shape);
			else revan = octant8.add(shape);
			dirty = true;
		} else {
			throw new RuntimeException("Cannot add an unbounded shape");
		}
		return revan;
	}

	@Override
	public PointND getMin(){
		if (dirty){
			init();
		}
		return super.getMin();
	}

	@Override
	public PointND getMax(){
		if (dirty){
			init();
		}
		return super.getMax();
	}


	/**
	 * Tests whether the dimension code is in the respective octant.<br><BR>
	 * The dimension code is a three dimensional int [] with the following coding for each dimension:<BR>
	 * <li>
	 * 0: object is completely left of the center axis, i.e. numerically smaller
	 * </li>
	 * <li> 1: object overlaps the center axis
	 * </li>
	 * <li> 2: object is completely right of the center axis, i.e. numerically greater
	 * </li><br><BR>
	 * This figure shows the eight octants in 3D:<BR><BR>
	 * <img src="http://upload.wikimedia.org/wikipedia/commons/thumb/6/60/Octant_numbers.svg/565px-Octant_numbers.svg.png" width =480 height=480>
	 * 
	 *  
	 * 
	 * @param i
	 * @param dim
	 * @return
	 */
	private boolean testOctant(int i, int ... dim){
		boolean revan = false;
		switch (i){
		case 1:
			revan = (dim[0]>0)&&(dim[1]>0)&&(dim[2]>0);
			break;
		case 2:
			revan = (dim[0]>0)&&(dim[1]<2)&&(dim[2]>0);
			break;
		case 3:
			revan = (dim[0]<2)&&(dim[1]<2)&&(dim[2]>0);
			break;
		case 4:
			revan = (dim[0]<2)&&(dim[1]>0)&&(dim[2]>0);
			break;
		case 5:
			revan = (dim[0]>0)&&(dim[1]>0)&&(dim[2]<2);
			break;		
		case 6:
			revan = (dim[0]>0)&&(dim[1]<2)&&(dim[2]<2);
			break;
		case 7:
			revan = (dim[0]<2)&&(dim[1]<2)&&(dim[2]<2);
			break;
		case 8:
			revan = (dim[0]<2)&&(dim[1]>0)&&(dim[2]<2);
			break;
		default:
			revan = false;
		}
		return revan;
	}

	@Override
	public void applyTransform(Transform t) {
		min = t.transform(min);
		max = t.transform(max);
		center = t.transform(center);
		ArrayList<AbstractShape> list = createToDoList();
		for (AbstractShape s : list) {
			s.applyTransform(t);
		}
	}

	@Override
	public PointND evaluate(PointND u) {
		return null;
	}

	@Override
	public int getDimension() {
		return dimension;
	}

	@Override
	public int getInternalDimension() {
		return octant1.size() + 
		octant2.size() +
		octant3.size() +
		octant4.size() +
		octant5.size() +
		octant6.size() +
		octant7.size() +
		octant8.size()
		;
	}

	@Override
	public ArrayList<PointND> intersect(AbstractCurve other) {
		if (dirty){
			init();
		}
		ArrayList<PointND> hits = new ArrayList<PointND>();
		if (octant1.size() > 0) {
			if (octant1.getHitsOnBoundingBox(other).size() > 0){
				for (AbstractShape s: octant1){
					hits.addAll(s.intersect(other));
				}
			}
		}
		if (octant2.size() > 0) {
			if (octant2.getHitsOnBoundingBox(other).size() > 0){
				for (AbstractShape s: octant2){
					hits.addAll(s.intersect(other));
				}
			}
		}
		if (octant3.size() > 0) {
			if (octant3.getHitsOnBoundingBox(other).size() > 0){
				for (AbstractShape s: octant3){
					hits.addAll(s.intersect(other));
				}
			}
		}
		if (octant4.size() > 0) {
			if (octant4.getHitsOnBoundingBox(other).size() > 0){
				for (AbstractShape s: octant4){
					hits.addAll(s.intersect(other));
				}
			}
		}
		if (octant5.size() > 0) {
			if (octant5.getHitsOnBoundingBox(other).size() > 0){
				for (AbstractShape s: octant5){
					hits.addAll(s.intersect(other));
				}
			}
		}
		if (octant6.size() > 0) {
			if (octant6.getHitsOnBoundingBox(other).size() > 0){
				for (AbstractShape s: octant6){
					hits.addAll(s.intersect(other));
				}
			}
		}
		if (octant7.size() > 0) {
			if (octant7.getHitsOnBoundingBox(other).size() > 0){
				for (AbstractShape s: octant7){
					hits.addAll(s.intersect(other));
				}
			}
		}
		if (octant8.size() > 0) {
			if (octant8.getHitsOnBoundingBox(other).size() > 0){
				for (AbstractShape s: octant8){
					hits.addAll(s.intersect(other));
				}
			}
		}
		return hits;
	}

	@Override
	public boolean isBounded() {
		return true;
	}

	/**
	 * Creates a list with each shape in this LineaerOctree appearing exactly once.
	 * @return
	 */
	private ArrayList<AbstractShape> createToDoList(){
		ArrayList<AbstractShape> toDo = new ArrayList<AbstractShape>();
		for (AbstractShape s: octant1) {
			toDo.add(s);
		}
		for (AbstractShape s: octant2) {
			toDo.add(s);
		}
		for (AbstractShape s: octant3) {
			toDo.add(s);
		}
		for (AbstractShape s: octant4) {
			toDo.add(s);
		}
		for (AbstractShape s: octant5) {
			toDo.add(s);
		}
		for (AbstractShape s: octant6) {
			toDo.add(s);
		}
		for (AbstractShape s: octant7) {
			toDo.add(s);
		}
		for (AbstractShape s: octant8) {
			toDo.add(s);
		}
		return toDo;
	}

	@Override
	public PointND[] getRasterPoints(int number){
		ArrayList<PointND[]> lists = new ArrayList<PointND[]>();
		ArrayList<AbstractShape> toDo = createToDoList();
		int sum = 0;
		for (AbstractShape s: toDo) {
			PointND [] pts = s.getRasterPoints(number/ toDo.size());
			sum += pts.length;
			lists.add(pts);
		}
		PointND [] points = new PointND[sum];
		int increment = 0;
		for (PointND [] pts : lists){
			System.arraycopy(pts, 0, points, increment, pts.length);
			increment += pts.length;
		}
		return points;
	}

	public String toString(){
		return "Octree at " + center + "\n" + 
		"Octant 1: " + octant1.size() + " " + octant1.getMin() + " " + octant1.getMax()+ "\n" +
		"Octant 2: " + octant2.size() + " " + octant2.getMin() + " " + octant2.getMax() + "\n" +
		"Octant 3: " + octant3.size() + " " + octant3.getMin() + " " + octant3.getMax()+ "\n" +
		"Octant 4: " + octant4.size() + " " + octant4.getMin() + " " + octant4.getMax()+ "\n" +
		"Octant 5: " + octant5.size() + " " + octant5.getMin() + " " + octant5.getMax()+ "\n" +
		"Octant 6: " + octant6.size() + " " + octant6.getMin() + " " + octant6.getMax()+ "\n" +
		"Octant 7: " + octant7.size() + " " + octant7.getMin() + " " + octant7.getMax()+ "\n" +
		"Octant 8: " + octant8.size() + " " + octant8.getMin() + " " + octant8.getMax()+ "\n" + 
		"Octant 1: " + octant1.toString() + "\n" +
		"Octant 2: " + octant2.toString() + "\n" +
		"Octant 3: " + octant3.toString() + "\n" +
		"Octant 4: " + octant4.toString() + "\n" +
		"Octant 5: " + octant5.toString() + "\n" +
		"Octant 6: " + octant6.toString() + "\n" +
		"Octant 7: " + octant7.toString() + "\n" +
		"Octant 8: " + octant8.toString() + "\n";
	}

	public int size() {
		return createToDoList().size();
	}

	@Override
	public boolean addAll(Collection<? extends AbstractShape> arg0) {
		boolean revan = false;
		for (AbstractShape s: arg0){
			if (add(s)){
				revan = true;
			}
		}
		dirty = true;
		return revan;
	}

	@Override
	public void clear() {
		octant1.clear();
		octant2.clear();
		octant3.clear();
		octant4.clear();
		octant5.clear();
		octant6.clear();
		octant7.clear();
		octant8.clear();
		dirty = true;
	}

	@Override
	public boolean contains(Object arg0) {
		return createToDoList().contains(arg0);
	}

	@Override
	public boolean containsAll(Collection<?> arg0) {
		return createToDoList().containsAll(arg0);
	}

	@Override
	public boolean isEmpty() {
		return createToDoList().isEmpty();
	}

	@Override
	public Iterator<AbstractShape> iterator() {
		return createToDoList().iterator();
	}

	@Override
	public boolean remove(Object arg0) {
		boolean revan = false;
		revan = revan || octant1.remove(arg0);
		revan = revan || octant2.remove(arg0);
		revan = revan || octant3.remove(arg0);
		revan = revan || octant4.remove(arg0);
		revan = revan || octant5.remove(arg0);
		revan = revan || octant6.remove(arg0);
		revan = revan || octant7.remove(arg0);
		revan = revan || octant8.remove(arg0);
		if (revan) dirty = true;
		return revan;
	}

	@Override
	public boolean removeAll(Collection<?> arg0) {
		boolean revan = false;
		for (Object o : arg0){
			revan = revan || remove(o);
		}
		if (revan) dirty = true;
		return revan;
	}

	@Override
	public boolean retainAll(Collection<?> arg0) {
		boolean revan = false;
		for (AbstractShape s: this){
			if (!arg0.contains(s)){
				revan = revan || remove(s);
			}
		}
		if (revan) dirty = true;
		return revan;
	}

	@Override
	public Object[] toArray() {
		return createToDoList().toArray();
	}

	@Override
	public <T> T[] toArray(T[] arg0) {
		return createToDoList().toArray(arg0);
	}

	@Override
	public AbstractShape clone() {
		return new LinearOctree(this);
	}
	
}
/*
 * Copyright (C) 2010-2014 Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/