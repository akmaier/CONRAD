package edu.stanford.rsl.tutorial.iterative;

import edu.stanford.rsl.conrad.data.numeric.Grid3D;

/**
 * @author Mario Amrehn
 *
 */
public final class GridOp {

	private GridOp() {
	}

	// a = a+b
	public static Grid3D addInPlace(Grid3D sumA, Grid3D sumB) throws Exception {
		sumA.getGridOperator().addBy(sumA, sumB);
		return sumA;
	}

	// a = a + scalar
	public static Grid3D addInPlace(Grid3D sumA, float sumB) {
		sumA.getGridOperator().addBy(sumA, sumB);
		return sumA;
	}

	// c = a+b
	public static Grid3D add(Grid3D sumA, Grid3D sumB) throws Exception {
		return add(sumA, sumB, 0, 0, 0);
	}

	// c = a+b
	public static Grid3D add(Grid3D sumA, Grid3D sumB, int aXOff, int aYOff, int aZOff) throws Exception {
		int[] sA = sumA.getSize();
		int[] sB = sumB.getSize();
		if (sA[0] != sB[0] || sA[1] != sB[1] || sA[2] != sB[2])
			throw new Exception("GridOperation: Sizes of grids don't match");

		Grid3D res = new Grid3D(sumA);
		for (int x = aXOff; x < sA[0]+aXOff; ++x)
			for (int y = aYOff; y < sA[1]+aYOff; ++y)
				for (int z = aZOff; z < sA[2]+aZOff; ++z){
					int xIdx = (x >= sA[0] || x < 0) ? Math.min(Math.max(0, x), sA[0]-1) : x;
					int yIdx = (y >= sA[1] || y < 0) ? Math.min(Math.max(0, y), sA[1]-1) : y;
					int zIdx = (z >= sA[2] || z < 0) ? Math.min(Math.max(0, z), sA[2]-1) : z;
					res.setAtIndex(x, y, z,  sumA.getAtIndex(xIdx,yIdx,zIdx) + sumB.getAtIndex(x,y,z));
				}
		res.setSpacing(sumA.getSpacing());
		res.setOrigin(sumA.getOrigin());
		return res;
	}

	// c = a + scalar
	public static Grid3D add(Grid3D sumA, float sumB) throws Exception {
		Grid3D res = new Grid3D(sumA);
		res.getGridOperator().addBy(res, sumB);
		res.setSpacing(sumA.getSpacing());
		res.setOrigin(sumA.getOrigin());
		return res;
	}

	// c = a-b
	public static Grid3D sub(Grid3D min, Grid3D sub) throws Exception {
		return sub(min, sub, 0, 0, 0, true);
	}

	// c = a'-b
	public static Grid3D sub(Grid3D min, Grid3D sub, int aXOff, int aYOff, int aZOff, final boolean offsetLeft) throws Exception {
		int[] sA = min.getSize();
		int[] sB = sub.getSize();
		if (sA[0] != sB[0] || sA[1] != sB[1] || sA[2] != sB[2])
			throw new Exception("GridOperation: Sizes of grids don't match");

		Grid3D res = new Grid3D(min);
		for (int x = aXOff; x < sA[0]+aXOff; ++x)
			for (int y = aYOff; y < sA[1]+aYOff; ++y)
				for (int z = aZOff; z < sA[2]+aZOff; ++z){
					int xIdx = (x >= sA[0] || x < 0) ? Math.min(Math.max(0, x), sA[0]-1) : x;
					int yIdx = (y >= sA[1] || y < 0) ? Math.min(Math.max(0, y), sA[1]-1) : y;
					int zIdx = (z >= sA[2] || z < 0) ? Math.min(Math.max(0, z), sA[2]-1) : z;
					if(offsetLeft)
						res.setAtIndex(x-aXOff,y-aYOff,z-aZOff,  min.getAtIndex(xIdx,yIdx,zIdx) - sub.getAtIndex(x-aXOff,y-aYOff,z-aZOff));
					else
						res.setAtIndex(x-aXOff,y-aYOff,z-aZOff,  min.getAtIndex(x-aXOff,y-aYOff,z-aZOff) - sub.getAtIndex(xIdx,yIdx,zIdx));
				}
		res.setSpacing(min.getSpacing());
		res.setOrigin(min.getOrigin());
		return res;
	}

	// a = a-b
	public static Grid3D subInPlace(Grid3D min, Grid3D sub) throws Exception {
		min.getGridOperator().subtractBy(min, sub);
		return min;
	}

	//	// a = a-b'
	//	public static void subInPlaceForDivergence(Grid3D min, Grid3D sub, int aXOff, int aYOff, int aZOff) throws Exception {
	//		int[] sA = min.getSize();
	//		int[] sB = sub.getSize();
	//		if (sA[0] != sB[0] || sA[1] != sB[1] || sA[2] != sB[2])
	//			throw new Exception("GridOperation: Sizes of grids don't match");
	//
	//		float[][][] a = min.getBuffer();
	//		float[][][] b = sub.getBuffer();
	//		for (int x = aXOff; x < sA[0]+aXOff; ++x)
	//			for (int y = aYOff; y < sA[1]+aYOff; ++y)
	//				for (int z = aZOff; z < sA[2]+aZOff; ++z){
	//					int xIdx = (x >= sA[0] || x < 0) ? Math.min(Math.max(0, x), sA[0]-1) : x;
	//					int yIdx = (y >= sA[1] || y < 0) ? Math.min(Math.max(0, y), sA[1]-1) : y;
	//					int zIdx = (z >= sA[2] || z < 0) ? Math.min(Math.max(0, z), sA[2]-1) : z;
	//					if(0 != aXOff){
	//						if(0 == xIdx) continue;
	//						if(sA[0]-1 == xIdx){
	//							a[x-aXOff][y-aYOff][z-aZOff]
	//						}
	//					}
	//					a[x-aXOff][y-aYOff][z-aZOff] -= b[xIdx][yIdx][zIdx];
	//				}
	//	}

	// c = a/b (per element)
	public static Grid3D div(Grid3D divid, Grid3D divis) throws Exception {
		Grid3D res = new Grid3D(divid);
		res.getGridOperator().divideBy(res, divis);
		res.setSpacing(divid.getSpacing());
		res.setOrigin(divid.getOrigin());
		return res;
	}

	public static Grid3D divInPlace(Grid3D divid, Grid3D divis) throws Exception {
		divid.getGridOperator().divideBy(divid, divis);
		return divid;
	}

	// c = a * scalar
	public static Grid3D mul(Grid3D multA, float multB) {
		Grid3D res = new Grid3D(multA);
		res.getGridOperator().multiplyBySave(res, multB);
		res.setSpacing(multA.getSpacing());
		res.setOrigin(multA.getOrigin());
		return res;
	}

	// c = a*b (per element)
	public static Grid3D mul(Grid3D multA, Grid3D multB) throws Exception {
		Grid3D res = new Grid3D(multA);
		res.getGridOperator().multiplyBySave(res, multB);
		res.setSpacing(multA.getSpacing());
		res.setOrigin(multA.getOrigin());
		return res;
	}

	// a = a*b
	public static Grid3D mulInPlace(Grid3D multA, Grid3D multB) throws Exception {
		multA.getGridOperator().multiplyBySave(multA, multB);
		return multA;
	}

	// a = a * scalar
	public static Grid3D mulInPlace(Grid3D multA, float multB) throws Exception {
		multA.getGridOperator().multiplyBySave(multA, multB);
		return multA;
	}

	public static double rmse(Grid3D a, Grid3D b) throws Exception {
		return a.getGridOperator().rmse(a, b);
	}

	public static double l1Norm(Grid3D c) {
		return c.getGridOperator().normL1(c);
	}

	

	/**
	 * @param a
	 * @return minimum value of grid a
	 */
	public static double min(Grid3D a) {
		return a.getGridOperator().min(a);
	}

	/**
	 * @param a
	 * @return maximum value of grid a
	 */
	public static double max(Grid3D a) {
		return a.getGridOperator().max(a);
	}

	/**
	 * @param a
	 * @return element wise square root values of grid a
	 */
	public static Grid3D sqrtInPlace(Grid3D a) {
		a.getGridOperator().pow(a, 2);
		return a;
	}

	public static int numNeg(Grid3D a) {
		return a.getGridOperator().countNegativeElements(a);
	}


	public static double sum(Grid3D c) {
		return c.getGridOperator().sumSave(c);
	}

	///////////////////
	// Readability improvements
	///////////////////

	// a = a^2
	public static Grid3D square(Grid3D a) throws Exception {
		return mul(a, a);
	}

	// d = a+b+c
	public static Grid3D add(Grid3D a, Grid3D b, Grid3D c) throws Exception {
		Grid3D res = add(a, b);
		addInPlace(res, c);
		return res;
	}

	// d = a+b+c + s
	public static Grid3D add(Grid3D a, Grid3D b, Grid3D c, float s)
			throws Exception {
		Grid3D res = add(a, b);
		addInPlace(res, c);
		addInPlace(res, s);
		return res;
	}

}
