/*
 * Copyright (C) 2010-2014 - Andreas Maier 
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/
package edu.stanford.rsl.conrad.data.numeric.iterators;

import edu.stanford.rsl.conrad.data.numeric.Grid2D;
import edu.stanford.rsl.conrad.data.numeric.NumericGrid;

public class NumericFloatIterator{

	protected NumericGrid m_grid;

	/**
	 * The condition for out of bound values
	 * 0 = clamp to zero
	 * 1 = clamp to constant
	 * 2 = clamp to edge value
	 * 3 = symmetrically extend value
	 * 4 = extend periodically
	 */
	protected int boundaryCondition;

	/**
	 * The boundary size for each dimension. 
	 * Half of the boundary size is extended in each direction. 
	 * If N is the given size and N is odd, the boundary on the right side is extended by N/2 + 1, 
	 * whereas the boundary on the left side is extended by N/2.  
	 */
	protected int[] boundarySize;

	/**
	 * The value to set the boundary if using a clamp to constant boundary condition
	 */
	protected float clampValue;

	/**
	 * The iterators direction
	 */
	protected int[] direction;

	/**
	 * The current iterator position
	 */
	protected int[] position;

	protected int dim;


	public NumericFloatIterator(NumericGrid g){
		this(g, 0, new int[g.getSize().length], 0.f, new int[g.getSize().length]);
	}

	public NumericFloatIterator(NumericGrid g, int boundCond, int[] boundSize, float clampValue, int[] pos, int []dir) {
		// set the grid
		this.m_grid = g;
		// initialize the boundaries
		boundaryCondition = boundCond;
		boundarySize = boundSize;
		this.clampValue = clampValue;
		// set initial direction
		direction = dir;
		// set initial position
		position = pos;
		// the dimension
		dim = g.getSize().length;
	}

	public NumericFloatIterator(NumericGrid g, int boundCond, int[] boundSize, float clampValue, int[] pos) {
		// set the grid
		this.m_grid = g;
		// initialize the boundaries
		boundaryCondition = boundCond;
		boundarySize = boundSize;
		this.clampValue = clampValue;
		// set initial direction
		direction = new int[g.getSpacing().length];
		direction[0]=1;
		// set initial position
		position = pos;
		// the dimension
		dim = g.getSize().length;
	}


	public boolean hasNext(){
		boolean val = true;
		for (int i=0; i < dim; ++i){
			int tmp = position[i];
			if (tmp < 0 - boundarySize[i]/2 || tmp >= m_grid.getSize()[i] + boundarySize[i]/2 + boundarySize[i]%2)
				val = false;
		}
		return val;

	}

	private void move(){
		for (int i=0; i < dim; ++i)
			position[i]+=direction[i];
	}

	public float getNext(){
		int[] currentPosition = null;
		Float val = null;
		for (int i=0; i < dim; ++i){
			boolean bleft=position[i] < 0;
			boolean bright=position[i] >= m_grid.getSize()[i];
			// check if we have a boundary case
			if (bleft || bright) {
				switch (boundaryCondition) {
				case 0: 
					val = new Float(0.f);
					break;
				case 1: 
					val = new Float(clampValue);
					break;
				case 2:
					currentPosition = (currentPosition==null) ? position.clone() : currentPosition;
					currentPosition[i] = (bleft) ? 0 : m_grid.getSize()[i]-1;
					break;
				case 3:
					currentPosition = (currentPosition==null) ? position.clone() : currentPosition;
					// TODO not save if grid dimension is smaller than the boundary
					currentPosition[i] = (bleft) ? Math.abs(position[i]) -1  : m_grid.getSize()[i] - (position[i] % m_grid.getSize()[i]) -1;
					break;
				case 4:
					currentPosition = (currentPosition==null) ? position.clone() : currentPosition;
					currentPosition[i] = (bleft) ? m_grid.getSize()[i] + ((position[i]+1) % m_grid.getSize()[i]) -1 : (position[i] % m_grid.getSize()[i]);
					break;
				default:
					break;
				}
			}
		}
		float out = (val != null) ? val.floatValue() : ((currentPosition!=null) ? m_grid.getValue(currentPosition) : m_grid.getValue(position));
		this.move();
		return out;
	}
	
	public int[] getPosition(){
		return position;
	}
	
	public void setPosition(int... pos){
		position = pos;
	}
	
	public void setDirection(int... dir){
		direction = dir;
	}
	
	public int[] getDirection(){
		return direction.clone();
	}
 
	
	public static void main(String[] args) {
		NumericGrid g = new Grid2D(8,8);
		for (int i=0; i < 8; i++){
			for (int j=0; j<8; j++)
			((Grid2D)g).setAtIndex(i, j, i);
		}
			
		NumericFloatIterator it = new NumericFloatIterator(g,4,new int[]{16,16},10.f,new int[]{-8,-8},new int[]{1,0});
		while (it.hasNext())
		{
			System.out.print(it.getPosition()[0]+ "/" + it.getPosition()[1] + ": " + it.getNext() + "\n");
			
		}
	}

}


