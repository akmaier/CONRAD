package edu.stanford.rsl.conrad.numerics;


import edu.stanford.rsl.conrad.numerics.SimpleOperators;
import edu.stanford.rsl.conrad.utils.CONRAD;

import java.io.Serializable;
import java.util.Scanner;
import java.util.Vector;


/**
 * 
 * @author Andreas Keil
 */
public class SimpleMatrix implements Serializable {

	private static final long serialVersionUID = -8285872089886352233L;

	
	/////////////////////////////////////////////////
	// pre-defined constants                       //
	/////////////////////////////////////////////////

	public static final SimpleMatrix I_2 = new SimpleMatrix(2, 2);
	public static final SimpleMatrix I_3 = new SimpleMatrix(3, 3);
	public static final SimpleMatrix I_4 = new SimpleMatrix(4, 4);
	static {
		I_2.identity();
		I_3.identity();
		I_4.identity();
	}

	
	/////////////////////////////////////////////////
	// Properties                                  //
	/////////////////////////////////////////////////
	
	protected int rows;
	protected int cols;
	protected double[][] buf;

	
	/////////////////////////////////////////////////
	// Constructors                                //
	/////////////////////////////////////////////////
	
	/**
	 * Create an empty matrix. 
	 */
	public SimpleMatrix() {
		this.init(0, 0);
	}
	
	/**
	 * Create a matrix with row rows and column cols
	 * @param rows is number of rows in matrix
	 * @param cols is number of columns in matrix
	 */
	public SimpleMatrix(final int rows, final int cols) {
		this.init(rows, cols);
	}
	
	/**
	 * Creates a new matrix from another
	 * @param otherMat is matrix to be copied
	 */
	public SimpleMatrix(final SimpleMatrix otherMat) {
		this.init(otherMat);
	}
	
	/**
	 * Creates a new matrix from 2x2 double array
	 * @param otherBuffer
	 */
	public SimpleMatrix(final double[][] otherBuffer) {
		this.init(otherBuffer);
	}
	
	/**
	 * Creates a new matrix from string data
	 * @param str is dat string
	 */
	public SimpleMatrix(final String str) {
		this.init(str);
	}
	
	/**
	 * Creates a new matrix from a jama matrix
	 * @param other
	 */
	public SimpleMatrix(final Jama.Matrix other) {
		this(other.getArray());
	}

	
	/////////////////////////////////////////////////
	// Methods                                     //
	/////////////////////////////////////////////////
	/**
	 * Initialize zero matrix
	 * @param rows number of rows
	 * @param cols number of columns
	 */
	public void init(final int rows, final int cols) {
		assert (rows >= 0) : new IllegalArgumentException("Number of rows has to be greater than or equal to zero!");
		assert (cols >= 0) : new IllegalArgumentException("Number of columns has to be greater than or equal to zero!");
		if (this.rows != rows || this.cols != cols) {
			this.rows = rows;
			this.cols = cols;
			this.buf = new double[this.rows][this.cols];
		}
	}
	
	/**
	 * Initialize matrix with data from supplied matrix
	 * @param otherMat is source matrix
	 */
	public void init(final SimpleMatrix otherMat) {
		if (this.rows != otherMat.rows || this.cols != otherMat.cols) {
			this.rows = otherMat.rows;
			this.cols = otherMat.cols;
			this.buf = new double[this.rows][this.cols];
		}
		for (int r = 0; r < this.rows; ++r)
			System.arraycopy(otherMat.buf[r], 0, this.buf[r], 0, this.cols);
	}
	
	/**
	 * Initialize matrix with data from 2x2 double array
	 * @param otherBuffer is source double array
	 */
	public void init(final double[][] otherBuffer) {
		final int rows = otherBuffer.length;
		assert rows >= 0 : new IllegalArgumentException("Number of rows has to be greater than or equal to zero!");
		final int cols = (rows == 0) ? 0 : otherBuffer[0].length;
		assert cols >= 0 : new IllegalArgumentException("Number of columns has to be greater than or equal to zero!");
		if (this.rows != rows || this.cols != cols) {
			this.rows = rows;
			this.cols = cols;
			this.buf = new double[this.rows][this.cols];
		}
		for (int r = 0; r < this.rows; ++r) {
			assert (otherBuffer[r].length == this.cols) : new IllegalArgumentException("Given Matrix is not rectangular!");
			System.arraycopy(otherBuffer[r], 0, this.buf[r], 0, cols);
		}
	}
	
	/**
	 * Initialize matrix with data string of the form []
	 * @param str is data string
	 */
	public void init(final String str) {
		String strTrim = str.trim();
		if ((strTrim.charAt(0) != '[') || (strTrim.charAt(strTrim.length() - 1) != ']')) throw new RuntimeException("Error parsing matrix string!");
		Scanner scanner = new Scanner(strTrim.substring(1, strTrim.length()-1).trim());
		scanner.useDelimiter("\\s*;\\s*");
		Vector<SimpleVector> matrixDbl = new Vector<SimpleVector>();
		while (scanner.hasNext())
			matrixDbl.add(new SimpleVector(scanner.next()));
		int rows = matrixDbl.size();
		int cols = matrixDbl.elementAt(0).getLen();
		this.init(rows, cols);
		for (int r = 0; r < rows; ++r) {
			SimpleVector rowDbl = matrixDbl.elementAt(r);
			if (rowDbl.getLen() != cols) throw new RuntimeException("Error parsing matrix string!");
			this.setRowValue(r, rowDbl);
		}
	}

	@Override
	public SimpleMatrix clone() {
		return new SimpleMatrix(this);
	}
	
	/**
	 * @return 2x2 double array containing matrix entries
	 */
	public double[][] copyAsDoubleArray(){
		double[][] array = new double[this.rows][this.cols];
		this.copyTo(array);
		return array; 
	}
	
	/**
	 * Copy matrix entries to supplied 2x2 double array
	 * @param otherBuffer is 2x2 double array to be populated with matrix entries
	 */
	public void copyTo(final double[][] otherBuffer) {
		assert (this.rows == otherBuffer.length) : new IllegalArgumentException("Copying is only possible to an array of the same size!");
		for (int r = 0; r < this.rows; ++r) {
			assert (otherBuffer[r].length == this.cols) : new RuntimeException("Given array is not rectangular!");
			System.arraycopy(this.buf[r], 0, otherBuffer[r], 0, this.cols);
		}
	}
	
	/**
	 * @return number of rows in matrix
	 */
	public int getRows() {
		return this.rows;
	}
	
	/**
	 * @return number of columns in matrix
	 */
	public int getCols() {
		return this.cols;
	}
	
	/** Sets all matrix entries to the given value. */
	public void fill(final double value) {
		for (int r = 0; r < this.rows; ++r)
			java.util.Arrays.fill(this.buf[r], value);
	}
	
	/** Sets all matrix entries to 0.0. */
	public void zeros() {
		this.fill(0.0);
	}
	
	/** Sets all matrix entries to 1.0. */
	public void ones() {
		this.fill(1.0);
	}
	
	/**
	 * Assigns random values to the entries of the matrix.
	 * Values are uniformly distributed in the given interval [min, max).
	 * @param min  The lower bound of the interval the values are drawn from.
	 * @param max  The upper bound of the interval the values are drawn from. Note that value max
	 *             itself is excluded from the interval and therefore never assigned.
	 */
	public void randomize(final double min, final double max) {
		for (int r = 0; r < this.rows; ++r)
			for (int c = 0; c < this.cols; ++c)
				this.buf[r][c] = (max-min)*Math.random() + min;
	}
	
	/** Sets the matrix to the identity matrix, i.e. diagonal entries are set to 1.0, others to 0.0. */
	public void identity() {
		assert (this.rows == this.cols) : new RuntimeException("Matrix is not square and therefore cannot be set to an identity matrix!");
		for (int r = 0; r < this.rows; ++r)
			for (int c = 0; c < this.cols; ++c)
				this.buf[r][c] = (r == c) ? 1.0 : 0.0;
	}
	
	/**
	 * Retrieve matrix entry in the specified row and column
	 * @param row is row containing entry
	 * @param col is column containing entry
	 * @return entry in the specified row and column
	 */
	public double getElement(final int row, final int col) {
		return this.buf[row][col];
	}
	
	/**
	 * Replaces matrix entry in the specified row and column with given value
	 * @param row is row containing entry to be replaced
	 * @param col is column containing entry to be replaced
	 * @param val is value to replace matrix entry in the specified row and column
	 */
	public void setElementValue(final int row, final int col, final double val) {
		this.buf[row][col] = val;
	}
	
	/**
	 * Creates a new sub matrix of this matrix
	 * @param firstRow is the first row of entries to be copied to sub matrix
	 * @param firstCol is first column of entries to be copied to sub matrix
	 * @param sizeRows is number of rows to be copied starting from first row
	 * @param sizeCols is number of columns to be copied starting from first column
	 * @return a sub matrix of this matrix 
	 */
	public SimpleMatrix getSubMatrix(final int firstRow, final int firstCol, final int sizeRows, final int sizeCols) {
		final SimpleMatrix subMatrix = new SimpleMatrix(sizeRows, sizeCols);
		for (int r = 0; r < sizeRows; ++r)
			System.arraycopy(this.buf[r+firstRow], firstCol, subMatrix.buf[r], 0, sizeCols); 
		return subMatrix;
	}
	
	/**
	 * Creates a new sub matrix with entries from ordered rows and ordered columns provided
	 * @param selectRows is ordered array containing rows to be copied
	 * @param selectCols is ordered array containing columns to be copied
	 * @return a sub matrix of this matrix 
	 */
	public SimpleMatrix getSubMatrix(final int[] selectRows, final int[] selectCols) {
		final SimpleMatrix subMatrix = new SimpleMatrix(selectRows.length, selectCols.length);
		int subRow = 0;
		for (int r : selectRows) {
			int subCol = 0;
			for (int c : selectCols) {
				subMatrix.buf[subRow][subCol] = this.buf[r][c];
				++subCol;
			}
			++subRow;
		}
		return subMatrix;
	}
	

	public SimpleMatrix getSubMatrix(final int deleteRow, final int deleteCol) {
		final SimpleMatrix subMatrix = new SimpleMatrix(this.rows-1, this.cols-1);
		for (int r = 0; r < deleteRow; ++r) {
			System.arraycopy(this.buf[r], 0, subMatrix.buf[r], 0, deleteCol);
			System.arraycopy(this.buf[r], deleteCol+1, subMatrix.buf[r], deleteCol, this.cols-deleteCol-1);
		}
		for (int r = deleteRow+1; r < this.rows; ++r) {
			System.arraycopy(this.buf[r], 0, subMatrix.buf[r-1], 0, deleteCol);
			System.arraycopy(this.buf[r], deleteCol+1, subMatrix.buf[r-1], deleteCol, this.cols-deleteCol-1);
		}
		return subMatrix;
	}
	
	/**
	 * Replaces matrix entries starting at firsRow and firstCol with entries from subMatrix
	 * @param firstRow is row of first element to be replaced
	 * @param firstCol is column of first element to be replaced
	 * @param subMatrix is sub matrix containing entries to replace matrix entries
	 */
	public void setSubMatrixValue(final int firstRow, final int firstCol, final SimpleMatrix subMatrix) {
		for (int r = 0; r < subMatrix.rows; ++r)
			for (int c = 0; c < subMatrix.cols; ++c)
				this.buf[r+firstRow][c+firstCol] = subMatrix.buf[r][c];
	}
	
	/**
	 * Returns a vector containing a sub row in current matrix.
	 * @param row is row containing desired sub row
	 * @param firstCol is the starting column of desired sub row
	 * @param sizeCols is number of columns in sub row
	 * @return vector containing sub row of row, from firstCol to firstCol + sizeCols
	 */
	public SimpleVector getSubRow(final int row, final int firstCol, final int sizeCols) {
		final SimpleVector subVector = new SimpleVector(sizeCols);
		for (int c = 0; c < sizeCols; ++c)
			subVector.buf[c] = this.buf[row][firstCol+c];
		return subVector;
	}
	
	/**
	 * Replace the entries of sub row starting at [row,firstCol] with subRow
	 * @param row is row containing desired sub row
	 * @param firstCol is the starting column of desired sub row
	 * @param subRow is vector containing new entries
	 */
	public void setSubRowValue(final int row, final int firstCol, final SimpleVector subRow) {
		for (int c = 0; c < subRow.getLen(); ++c)
			this.buf[row][firstCol+c] = subRow.buf[c];
	}
	
	/**
	 * Retrieve row from index row of matrix
	 * @param row is index of row to retrieved
	 * @return row vector
	 */
	public SimpleVector getRow(final int row) {
		return this.getSubRow(row, 0, this.cols);
	}
	
	/**
	 * Replace  row with newRow
	 * @param row is index of row to be replaced
	 * @param newRow is vector containing new entries
	 */
	public void setRowValue(final int row, final SimpleVector newRow) {
		assert (this.cols == newRow.getLen()) : new IllegalArgumentException("Dimension mismatch!");
		this.setSubRowValue(row, 0, newRow);
	}
	
	/**
	 * Returns a vector containing a sub column in current matrix.
	 * @param col is index of column containing desired sub column
	 * @param firstRow is the starting column of desired sub column
	 * @param sizeRows is number of columns in sub column
	 * @return vector containing sub column of col, from firstCol to firstRow + sizeRows
	 */
	public SimpleVector getSubCol(final int firstRow, final int col, final int sizeRows) {
		final SimpleVector subVector = new SimpleVector(sizeRows);
		for (int r = 0; r < sizeRows; ++r)
			subVector.buf[r] = this.buf[firstRow+r][col];
		return subVector;
	}
	
	/**
	 * Replace the entries of sub column starting at [col,firstRow] with subCol
	 * @param col is index of column containing desired sub column
	 * @param firstRow is the starting row of desired sub column
	 * @param subCol is vector containing new entries
	 */
	public void setSubColValue(final int firstRow, final int col, final SimpleVector subCol) {
		for (int r = 0; r < subCol.getLen(); ++r)
			this.buf[firstRow+r][col] = subCol.buf[r];
	}
	
	/**
	 * Retrieve column from index col of matrix
	 * @param col is index of column to retrieved
	 * @return column vector
	 */
	public SimpleVector getCol(final int col) {
		return this.getSubCol(0, col, this.rows);
	}
	
	/**
	 * Replace col with newCol
	 * @param col is index of column to be replaced
	 * @param newCol is vector containing new entries
	 */
	public void setColValue(final int col, final SimpleVector newCol) {
		assert (this.rows == newCol.getLen()) : new IllegalArgumentException("Dimension mismatch!");
		this.setSubColValue(0, col, newCol);
	}
	
	
	/**
	 * @return vector containing diagonal entries of matrix
	 */
	public SimpleVector getDiag() {
		final int min_rc = Math.min(this.rows, this.cols);
		final SimpleVector diag = new SimpleVector(min_rc);
		for (int i = 0; i < min_rc; ++i)
			diag.buf[i] = this.buf[i][i];
		return diag;
	}
	
	/**
	 * Replace diagonal entries of matrix with diag
	 * @param diag is vector containing new entries
	 */
	public void setDiagValue(final SimpleVector diag) {
		for (int i = 0; i < diag.getLen(); ++i)
			this.buf[i][i] = diag.buf[i];
	}
	
	/**
	 * Add addend to entry at [row,col]  in place
	 * @param row of entry to be updated
	 * @param col of entry to be updated
	 * @param addend is value to be added to entry at [row,col]
	 */
	public void addToElement(final int row, final int col, final double addend) {
		this.buf[row][col] += addend;
	}
	
	/**
	 * Subtract subtrahend from entry at [row,col] in place
	 * @param row of entry to be updated
	 * @param col of entry to be updated
	 * @param subtrahend is value to be subtracted from entry at [row,col]
	 */
	public void subtractFromElement(final int row, final int col, final double subtrahend) {
		this.buf[row][col] -= subtrahend;
	}
	
	/**
	 * Multiply factor to entry at [row,col] in place
	 * @param row of entry to be updated
	 * @param col of entry to be updated
	 * @param factor is value to be multiplied to entry at [row,col]
	 */
	public void multiplyElementBy(final int row, final int col, final double factor) {
		this.buf[row][col] *= factor;
	}
	
	/**
	 * Divide divisor from entry at [row,col] in place
	 * @param row of entry to be updated
	 * @param col of entry to be updated
	 * @param divisor is value to be divided from entry at [row,col]
	 */
	public void divideElementBy(final int row, final int col, final double divisor) {
		this.buf[row][col] /= divisor;
	}
	
	/**
	 * Add addend to all entries in matrix in place
	 * @param addend is value to be added to all entries in matrix
	 */
	public void add(final double addend) {
		for (int r = 0; r < this.rows; ++r)
			for (int c = 0; c < this.cols; ++c)
				this.buf[r][c] += addend;
	}
	
	/**
	 * Subtract subtrahend from all entries in matrix in place
	 * @param subtrahend is value to be subtracted from all entries in matrix
	 */
	public void subtract(final double subtrahend) {
		for (int r = 0; r < this.rows; ++r)
			for (int c = 0; c < this.cols; ++c)
				this.buf[r][c] -= subtrahend;
	}
	
	/**
	 * Multiply factor to all entries in matrix in place
	 * @param factor is value to be multiplied to all entries in matrix
	 */
	public void multiplyBy(final double factor) {
		for (int r = 0; r < this.rows; ++r)
			for (int c = 0; c < this.cols; ++c)
				this.buf[r][c] *= factor;
	}
	
	
	/**
	 * Multiply factor to all entries in matrix [current matrix is not updated]
	 * @param factor is value to be multiplied to all entries in matrix
	 * @return new matrix with updated entries
	 */
	public SimpleMatrix multipliedBy(final double factor) {
		final SimpleMatrix result = new SimpleMatrix(this.rows, this.cols);
		for (int r = 0; r < this.rows; ++r)
			for (int c = 0; c < this.cols; ++c)
				result.buf[r][c] = this.buf[r][c] * factor;
		return result;
	}
	
	/**
	 * Divide all entries in matrix by divisor in place
	 * @param divisor is value to be divided from all entries in matrix
	 */
	public void divideBy(final double divisor) {
		this.multiplyBy(1.0 / divisor);
	}
	
	/**
	 * Divide all entries in matrix by divisor [current matrix is not updated]
	 * @param divisor is value to be divided from all entries in matrix
	 * @return new matrix with updated entries
	 */
	public SimpleMatrix dividedBy(final double divisor) {
		return this.multipliedBy(1.0 / divisor);
	}
	
	/**
	 * Method to add a set of matrices to this matrix in place.
	 * @param addends are  set of matrices to be added to this matrix.
	 */
	public void add(final SimpleMatrix... addends) {
		assert addends.length >= 1 : new IllegalArgumentException("At least one other matrix has to be given!");
		for (SimpleMatrix addend : addends) {
			assert addend.rows == this.rows;
			assert addend.cols == this.cols;
			for (int r = 0; r < this.rows; ++r)
				for (int c = 0; c < this.cols; ++c)
					this.buf[r][c] += addend.buf[r][c];
		}
	}
	
	/**
	 * Method to subtract a set of matrices to this matrix in place.
	 * @param subtrahends are  set of matrices to be from this matrix.
	 */
	public void subtract(final SimpleMatrix... subtrahends) {
		assert subtrahends.length >= 1 : new IllegalArgumentException("At least one other matrix has to be given!");
		for (SimpleMatrix subtrahend : subtrahends) {
			assert subtrahend.rows == this.rows;
			assert subtrahend.cols == this.cols;
			for (int r = 0; r < this.rows; ++r)
				for (int c = 0; c < this.cols; ++c)
					this.buf[r][c] -= subtrahend.buf[r][c];
		}
	}
	
	/**
	 * ordered multiplication of matrix entries in place
	 * @param other
	 */
	public void multiplyElementWiseBy(final SimpleMatrix other) {
		assert (other.rows == this.rows) && (other.cols == this.cols);
		for (int r = 0; r < this.rows; ++r)
			for (int c = 0; c < this.cols; ++c)
				this.buf[r][c] *= other.buf[r][c];
	}
	
	/**
	 * ordered division of matrix entries in place
	 * @param other
	 */
	public void divideElementWiseBy(final SimpleMatrix other) {
		assert (other.rows == this.rows) && (other.cols == this.cols);
		for (int r = 0; r < this.rows; ++r)
			for (int c = 0; c < this.cols; ++c)
				this.buf[r][c] /= other.buf[r][c];
	}
	
	/**
	 * multiply all the entries in a matrix by -1 in place.
	 */
	public void negate() {
		for (int r = 0; r < this.rows; ++r)
			for (int c = 0; c < this.cols; ++c)
				this.buf[r][c] = -this.buf[r][c];
	}
	
	/**
	 * @return a copy of this matrix with all entries multiplied by -1
	 */
	public SimpleMatrix negated() {
		final SimpleMatrix result = new SimpleMatrix(this.rows, this.cols);
		for (int r = 0; r < this.rows; ++r)
			for (int c = 0; c < this.cols; ++c)
				result.buf[r][c] = -this.buf[r][c];
		return result;
	}
	
	/**
	 * Performs a matrix transpose in place.
	 */	
	public void transpose() {
		assert (this.rows == this.cols) : new RuntimeException("In-place transposition not possible for non-square matrices!");
		double tmp;
		for (int r = 0; r < this.rows; ++r) {
			for (int c = r+1; c < this.cols; ++c) {
				tmp = this.buf[r][c];
				this.buf[r][c] = this.buf[c][r];
				this.buf[c][r] = tmp;
			}
		}
	}
	
	/**
	 * @return a transposed copy of this matrix.
	 */
	public SimpleMatrix transposed() {
		final SimpleMatrix result = new SimpleMatrix(this.cols, this.rows);
		for (int r = 0; r < result.getRows(); ++r) {
			for (int c = 0; c < result.getCols(); ++c) {
				result.buf[r][c] = this.buf[c][r];
			}
		}
		return result;
	}

	public static enum MatrixNormType {
		/** The L_1-induced norm is equivalent to the maximum absolute column sum of the matrix. */
		MAT_NORM_L1,
		/** The L_2-induced norm is the largest singular value of the matrix M or the largest eigenvalue of A^* * A. */
		MAT_NORM_L2,
		/** The L_infinity-induced norm is equivalent to the maximum absolute row sum of the matrix. */
		MAT_NORM_LINF,
		/** The Frobenius norm is the entry-wise 2-norm (the sum of squares of all entries). */
		MAT_NORM_FROBENIUS
	}
	
	public double norm(final MatrixNormType normType) {
		switch (normType) {
		case MAT_NORM_L1:
			SimpleVector columnSumOfAbs = new SimpleVector(this.cols);
			for (int c = 0; c < this.cols; ++c) columnSumOfAbs.buf[c] = this.getCol(c).norm(SimpleVector.VectorNormType.VEC_NORM_L1);
			return columnSumOfAbs.norm(SimpleVector.VectorNormType.VEC_NORM_LINF);
		case MAT_NORM_L2:
			// TODO: Implement matrix 2-norm once SVD is available.
			// Decomposition::SVD<typename Derived::Scalar> svd;
			// svd(M);
			// return svd.getS()(0,0);
			throw new RuntimeException("Not implemented yet! Do the SVD decomposition first.");
		case MAT_NORM_LINF:
			SimpleVector rowSumOfAbs = new SimpleVector(this.rows);
			for (int r = 0; r < this.rows; ++r) rowSumOfAbs.buf[r] = this.getRow(r).norm(SimpleVector.VectorNormType.VEC_NORM_L1);
			return rowSumOfAbs.norm(SimpleVector.VectorNormType.VEC_NORM_LINF);
		case MAT_NORM_FROBENIUS:
			double result = 0.0;
			for (int r = 0; r < this.rows; ++r)
				for (int c = 0; c < this.cols; ++c)
					result +=  this.buf[r][c]*this.buf[r][c];
			return Math.sqrt(result);
		default:
			throw new RuntimeException("Matrix norm type not implemented yet!");
		}
	}
	
	/**
	 * @return the determinant of this matrix
	 */
	public double determinant() {
		assert (this.isSquare()) : new RuntimeException("Matrix is not square and therefore determinatn cannot be computed!");;
		assert (this.rows >= 1) : new IllegalArgumentException("The matrix must be at least 1x1!");
		switch (this.rows) {
		case 1:
			return this.buf[0][0];
		case 2:
			return this.buf[0][0]*this.buf[1][1] - this.buf[0][1]*this.buf[1][0];
		case 3:
			return this.buf[0][0]*this.buf[1][1]*this.buf[2][2] + this.buf[0][1]*this.buf[1][2]*this.buf[2][0] + this.buf[0][2]*this.buf[1][0]*this.buf[2][1]
			- this.buf[2][0]*this.buf[1][1]*this.buf[0][2] - this.buf[2][1]*this.buf[1][2]*this.buf[0][0] - this.buf[2][2]*this.buf[1][0]*this.buf[0][1];
		default:
			// TODO: use an LU decomposition for big matrices (of a size greater then 5-10)
			double det = 0.0;
			double factor = 1.0;
			for (int i = 0; i < this.cols; ++i) {
				det += factor * this.buf[0][i] * this.getSubMatrix(0, i).determinant();
				factor *= -1.0;
			}
			return det;
		}
	}
	
	/**
	 * @return the condition number of this matrix
	 */
	public double conditionNumber(final MatrixNormType normType) {
		assert (this.isSquare()) : new RuntimeException("Matrix is not square and therefore determinant cannot be computed!");;
		assert (this.rows >= 1) : new IllegalArgumentException("The matrix must be at least 1x1!");
		switch (normType) {
		case MAT_NORM_L1:
			return this.norm(MatrixNormType.MAT_NORM_L1)/this.inverse(InversionType.INVERT_QR).norm(MatrixNormType.MAT_NORM_L1);
		case MAT_NORM_L2:
			throw new RuntimeException("Not implemented yet! Do the SVD or Eigenvalue decomposition first.");
		case MAT_NORM_LINF:
			SimpleVector absDiag = this.getDiag().absoluted();
			return absDiag.max()/absDiag.min();
		case MAT_NORM_FROBENIUS:
			throw new RuntimeException("Not implemented yet!");
		default:
			throw new RuntimeException("Matrix norm type not implemented yet!");
		}
	}
	
	/**
	 * @return true if matrix is a square matrix i.e numRows = numCols
	 */
	public boolean isSquare() {
		return (this.rows == this.cols);
	}
	
	/**
	 * Determines if matrix is an identity matrix 
	 * @param delta is error tolerance
	 * @return true is matrix 
	 */
	public boolean isIdentity(final double delta) {
		if (!this.isSquare()) throw new IllegalArgumentException("Only square matrices can be tested for identity!");
		for (int r = 0; r < this.rows; ++r) {
			for (int c = 0; c < this.cols; ++c) {
				double el = this.buf[r][c];
				if (r == c) {
					if (Math.abs(el - 1.0) > delta) return false;
				} else {
					if (Math.abs(el) > delta) return false;
				}
			}
		}
		return true;
	}

	public boolean isSingular(final double delta) {
		if (!this.isSquare()) throw new IllegalArgumentException("Only square matrices can be tested for singularity!");
		// TODO: consider using a LU or QR decomposition here (or inside determinant)
		return (Math.abs(this.determinant()) < delta);
	}

	/**
	 * Test for upper triangularity of a matrix.
	 * 
	 * This function does not test for squareness. Matrices with zeros below the main
	 * diagonal are also considered upper triangular as long as all of the non-square extension is
	 * filled with zeros.
	 * 
	 * @return  true if the matrix is upper triangular, false otherwise
	 */
	public boolean isUpperTriangular() {
		for (int r = 1; r < this.rows; ++r) {
			final int cMax = Math.min(this.cols, r);
			for (int c = 0; c < cMax; ++c) {
				if (this.buf[r][c] != 0.0) return false;
			}
		}
		return true;
	}
	
	/**
	 * Determines if matrix is orthogonal
	 * @param maxErr is tolerance
	 * @return true if matrix is orthogonal
	 */
	public boolean isOrthogonal(final double maxErr) {
		if (!this.isSquare()) throw new IllegalArgumentException("Only square matrices can be tested for orthogonality!");
		final int n = this.rows;
		final SimpleMatrix MtM = SimpleOperators.multiplyMatrixProd(this.transposed(), this);
		for (int r = 0; r < n; ++r)
			for (int c = 0; c < n; ++c)
				if (Math.abs(MtM.buf[r][c] - ((r == c) ? 1.0 : 0.0)) > maxErr) return false;
		return true;
	}

	
	public boolean isSpecialOrthogonal(final double maxErr) {
		if (!this.isOrthogonal(maxErr)) return false;
		return (Math.abs(this.determinant() - 1.0) <= maxErr);
	}
	
	/**
	 * Determines if matrix is a rotation matrix
	 * @param maxErr
	 * @return true if matrix is 2x2 and a rotation matrix, otherwise an illegalArgumentException is thrown
	 */
	public boolean isRotation2D(final double maxErr) {
		if (this.rows != 2) throw new IllegalArgumentException("Only 2x2 matrices can be tested whether they are 2D rotation matrices!");
		return this.isSpecialOrthogonal(maxErr);
	}
	
	/**
	 * Determines if matrix is a rotation matrix
	 * @param maxErr
	 * @return true if matrix is 3x3 and a rotation matrix, otherwise an illegalArgumentException is thrown
	 */
	public boolean isRotation3D(final double maxErr) {
		if (this.rows != 3) throw new IllegalArgumentException("Only 3x3 matrices can be tested whether they are 3D rotation matrices!");
		return this.isSpecialOrthogonal(maxErr);
	}
	
	/**
	 * Determines if matrix is a rigid motion matrix
	 * @param maxErr
	 * @return true if matrix is 2x2 and a rotation matrix, otherwise an illegalArgumentException is thrown
	 */
	public boolean isRigidMotion2D(final double maxErr) {
		if (this.rows != 3) throw new IllegalArgumentException("Only 3x3 matrices can be tested whether they are 2D rigid motion matrices!");
		if (!this.isSquare()) return false;
		final double scale = this.buf[2][2];
		if (Math.abs(this.buf[2][0]) > maxErr || Math.abs(this.buf[2][1]) > maxErr || Math.abs(scale) < CONRAD.DOUBLE_EPSILON) return false;
		return this.getSubMatrix(0, 0, 2, 2).dividedBy(scale).isRotation2D(maxErr);
	}
	
	/**
	 * Determines if matrix is a rigid motion matrix
	 * @param maxErr
	 * @return true if matrix is 2x2 and a rotation matrix, otherwise an illegalArgumentException is thrown
	 */
	public boolean isRigidMotion3D(final double maxErr) {
		if (this.rows != 4) throw new IllegalArgumentException("Only 4x4 matrices can be tested whether they are 3D rigid motion matrices!");
		if (!this.isSquare()) return false;
		final double scale = this.buf[3][3];
		if (Math.abs(this.buf[3][0]) > maxErr || Math.abs(this.buf[3][1]) > maxErr || Math.abs(this.buf[3][2]) > maxErr || Math.abs(scale) < CONRAD.DOUBLE_EPSILON) return false;
		return this.getSubMatrix(0, 0, 3, 3).dividedBy(scale).isRotation3D(maxErr);
	}

	/**
	 * Set the algorithm to be used during inversion
	 */
	public static enum InversionType {
		/** Inverts any matrices using the LU decomposition */
		INVERT_LU,
		/** Inverts any matrices using the SVD decomposition */
		INVERT_SVD,
		/** Inverts any matrices using the LU decomposition */
		INVERT_QR,
		/** Inverts upper-triangular matrices using back substitution */
		INVERT_UPPER_TRIANGULAR,
		/** Inverts homogeneous rigid motion matrices directly */
		INVERT_RT
	}
	/**
	 * Inverts the given matrix using the specified inversion method.
	 * The type of inversion has to be specified by the user and should
	 * be chosen depending on the matrix' properties.
	 * <em>Warning:</em> Better use the Solvers class for solving linear algebra problems.
	 * Explicitly computing the inverse of a matrix is usually not needed and often yields numerically worse results.
	 * @param inversionType  The type of inversion to be used.
	 * @return  The inverse of the matrix.
	 */
	public SimpleMatrix inverse(final InversionType inversionType) {
		switch(inversionType) {
		case INVERT_LU:
			assert(this.isSquare()) : new IllegalArgumentException("The matrix must be square!");
			throw new RuntimeException("Inversion type not yet implemented!");
		case INVERT_SVD:
			DecompositionSVD decompositionSVD = new DecompositionSVD(this);
			return decompositionSVD.inverse(true);
		case INVERT_QR:
			assert(this.isSquare()) : new IllegalArgumentException("The matrix must be square!");
			DecompositionQR qr = new DecompositionQR(this);
			SimpleMatrix Rinv = qr.getR().inverse(InversionType.INVERT_UPPER_TRIANGULAR);
			SimpleMatrix Qinv = qr.getQ().transposed();
			return SimpleOperators.multiplyMatrixProd(Rinv, Qinv);
		case INVERT_UPPER_TRIANGULAR:
			assert(this.isUpperTriangular()) : new IllegalArgumentException("The matrix must be square!");
			SimpleMatrix Minv_triang = new SimpleMatrix(this.rows, this.cols);
			for (int col = 0; col < this.cols; ++col) {
				// set diagonal element
				Minv_triang.buf[col][col] = 1.0/this.buf[col][col];
				for (int row = col-1; row >= 0; --row) {
					// compute unknown in position (row, col) from inner product of the respective row and column in the matrix product U * Uinv = I
					double sum = 0.0;
					for (int l = row+1; l <= col ; ++l)
						sum += this.buf[row][l] * Minv_triang.buf[l][col];
					Minv_triang.buf[row][col] = -sum*Minv_triang.buf[row][row];
				}
			}
			return Minv_triang;
		case INVERT_RT:
			if (this.rows == 4) {
				assert this.isRigidMotion3D(Math.sqrt(CONRAD.DOUBLE_EPSILON)) : new IllegalArgumentException("The matrix must be square!");
				final double scale = this.buf[3][3]; // Attention: Don't just normalize the input matrix and invert it, because M*Minv would not be equal identity!
				if (scale == 0) throw new IllegalArgumentException("The matrix must be a rigid motion matrix with [0 0 0 a] (with a != 0) in the last row!");
				if (this.buf[3][0] != 0 || this.buf[3][1] != 0 || this.buf[3][2] != 0) throw new IllegalArgumentException("The matrix must be a rigid motion matrix with [0 0 0 a] in the last row!");
				///TODO: check that R really is a matrix from SO(3)
				SimpleMatrix Rinverse = this.getSubMatrix(0, 0, 3, 3).transposed().dividedBy(scale);
				SimpleVector t = this.getSubCol(0, 3, 3).dividedBy(scale); // last column
				SimpleMatrix Minv_rt = new SimpleMatrix(4, 4);
				Minv_rt.setSubMatrixValue(0, 0, Rinverse.dividedBy(scale));
				Minv_rt.setSubColValue(0, 3, SimpleOperators.multiply(Rinverse.negated(), t).dividedBy(scale));
				Minv_rt.buf[3][0] = 0;
				Minv_rt.buf[3][1] = 0;
				Minv_rt.buf[3][2] = 0;
				Minv_rt.buf[3][3] = 1/scale;				
				return Minv_rt;
			} else if (this.rows == 3) {
				assert this.isRigidMotion2D(Math.sqrt(CONRAD.DOUBLE_EPSILON)) : new IllegalArgumentException("The matrix must be square!");
				final double scale = this.buf[2][2]; // Attention: Don't just normalize the input matrix and invert it, because M*Minv would not be equal identity!
				if (scale == 0) throw new IllegalArgumentException("The matrix must be a rigid motion matrix with [0 0 a] (with a != 0) in the last row!");
				if (this.buf[2][0] != 0 || this.buf[2][1] != 0) throw new IllegalArgumentException("The matrix must be a rigid motion matrix with [0 0 a] in the last row!");
				///TODO: check that R really is a matrix from SO(3)
				SimpleMatrix Rinverse = this.getSubMatrix(0, 0, 2, 2).transposed().dividedBy(scale);
				SimpleVector t = this.getSubCol(0, 2, 2).dividedBy(scale); // last column
				SimpleMatrix Minv_rt = new SimpleMatrix(3, 3);
				Minv_rt.setSubMatrixValue(0, 0, Rinverse.dividedBy(scale));
				Minv_rt.setSubColValue(0, 2, SimpleOperators.multiply(Rinverse.negated(), t).dividedBy(scale));
				Minv_rt.buf[2][0] = 0;
				Minv_rt.buf[2][1] = 0;
				Minv_rt.buf[2][2] = 1/scale;
				return Minv_rt;
			} else
				throw new IllegalArgumentException("The matrix must have dimensions 3x3 or 4x4!");
		default:
			throw new RuntimeException("Unknown matrix inversion type!");
		}
	}

	
	/////////////////////////////////////////////////
	// Serialization and Persistence               //
	/////////////////////////////////////////////////
	
	/**
	 * return a serialized equivalent of this matix
	 */
	public String getMatrixSerialization() {
		return this.toString();
	}
	
	/**
	 * Initialize matrix using a serialized equivalent
	 * @param str
	 */
	public void setMatrixSerialization(final String str) {
		this.init(str);
	}
	
	@Override
	public String toString() {
		String result = new String();
		result += "[";
		for (int r = 0; r < this.rows; ++r) {
			if (r != 0) result += " ";
			result += "[";
			for (int c = 0; c < this.cols; ++c) {
				if (c != 0) result += " ";
				result += new Double(this.buf[r][c]);
			}
			result += "]\n";
		}
		result += "]";
		return result;
	}

}
/*
 * Copyright (C) 2010-2014 Andreas Keil
 * CONRAD is developed as an Open Source project under the GNU General Public License (GPL).
*/