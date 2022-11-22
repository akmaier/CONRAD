package edu.stanford.rsl.conrad.angio.util.data.organization;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;



/**
 * A k-d tree (short for k-dimensional tree) is a space-partitioning data
 * structure for organizing points in a k-dimensional space. k-d trees are a
 * useful data structure for several applications, such as searches involving a
 * multidimensional search key (e.g. range searches and nearest neighbor
 * searches). k-d trees are a special case of binary space partitioning trees.
 * 
 * 
 *
 * @author Justin Wetherell <phishman3579@gmail.com>
 */
public class KDTree2D<T extends ComparablePoint2D> {

	private int k = 2;
	private KdNode root = null;


	protected static final int X_AXIS = 0;
	protected static final int Y_AXIS = 1;
	
	/**
	 * Default constructor.
	 */
	public KDTree2D() {
	}

	/**
	 * Constructor for creating a more balanced tree. It uses the
	 * "median of points" algorithm.
	 * 
	 * @param list
	 *            of PointNDs.
	 */
	public KDTree2D(ArrayList<ComparablePoint2D> list) {
		root = createNode(list, k, 0);
	}



	/**
	 * Create node from list of PointNDs.
	 * 
	 * @param list
	 *            of PointNDs.
	 * @param k
	 *            of the tree.
	 * @param depth
	 *            depth of the node.
	 * @return node created.
	 */
	private static KdNode createNode(List<ComparablePoint2D> list, int k, int depth) {
		if (list == null || list.size() == 0)
			return null;

		int axis = depth % k;
		if (axis == X_AXIS)
			Collections.sort(list, ComparablePoint2D.X_COMPARATOR);//sorts the values in the list according to x value.
		else
			Collections.sort(list, ComparablePoint2D.Y_COMPARATOR);
		
		KdNode node = null;
		if (list.size() > 0) {
			int medianIndex = list.size() / 2;
			node = new KdNode(k, depth, list.get(medianIndex));
			List<ComparablePoint2D> less = new ArrayList<ComparablePoint2D>(list.size() - 1);
			List<ComparablePoint2D> more = new ArrayList<ComparablePoint2D>(list.size() - 1);
			// Process list to see where each non-median point lies
			for (int i = 0; i < list.size(); i++) {
				if (i == medianIndex)
					continue;
				ComparablePoint2D p = list.get(i);
				if (KdNode.compareTo(depth, k, p, node.id) <= 0) {
					less.add(p);
				} else {
					more.add(p);
				}
			}
			if ((medianIndex - 1) >= 0) {
				// Cannot assume points before the median are less since they
				// could be equal
				// List<ComparablePoint> less = list.subList(0, mediaIndex);
				if (less.size() > 0) {
					node.lesser = createNode(less, k, depth + 1);
					node.lesser.parent = node;
				}
			}
			if ((medianIndex + 1) <= (list.size() - 1)) {
				// Cannot assume points after the median are less since they
				// could be equal
				// List<ComparablePoint> more = list.subList(mediaIndex + 1,
				// list.size());
				if (more.size() > 0) {
					node.greater = createNode(more, k, depth + 1);
					node.greater.parent = node;
				}
			}
		}

		return node;
	}

	/**
	 * Add value to the tree. Tree can contain multiple equal values.
	 * 
	 * @param value
	 *            T to add to the tree.
	 * @return True if successfully added to tree.
	 */
	public boolean add(T value) {
		if (value == null)
			return false;

		if (root == null) {
			root = new KdNode(value);
			return true;
		}

		KdNode node = root;
		while (true) {
			if (KdNode.compareTo(node.depth, node.k, value, node.id) <= 0) {
				// Lesser
				if (node.lesser == null) {
					KdNode newNode = new KdNode(k, node.depth + 1, value);
					newNode.parent = node;
					node.lesser = newNode;
					break;
				}
				node = node.lesser;
			} else {
				// Greater
				if (node.greater == null) {
					KdNode newNode = new KdNode(k, node.depth + 1, value);
					newNode.parent = node;
					node.greater = newNode;
					break;
				}
				node = node.greater;
			}
		}

		return true;
	}

	/**
	 * Does the tree contain the value.
	 * 
	 * @param value
	 *            T to locate in the tree.
	 * @return True if tree contains value.
	 */
	public boolean contains(T value) {
		if (value == null)
			return false;

		KdNode node = getNode(this, value);
		return (node != null);
	}

	/**
	 * Locate T in the tree.
	 * 
	 * @param tree
	 *            to search.
	 * @param value
	 *            to search for.
	 * @return KdNode or NULL if not found
	 */
	private static final <T extends ComparablePoint2D> KdNode getNode(KDTree2D<T> tree, T value) {
		if (tree == null || tree.root == null || value == null)
			return null;

		KdNode node = tree.root;
		while (true) {
			if (node.id.equals(value)) {
				return node;
			} else if (KdNode.compareTo(node.depth, node.k, value, node.id) <= 0) {
				// Lesser
				if (node.lesser == null) {
					return null;
				}
				node = node.lesser;
			} else {
				// Greater
				if (node.greater == null) {
					return null;
				}
				node = node.greater;
			}
		}
	}

	/**
	 * Remove first occurrence of value in the tree.
	 * 
	 * @param value
	 *            T to remove from the tree.
	 * @return True if value was removed from the tree.
	 */
	public boolean remove(T value) {
		if (value == null)
			return false;

		KdNode node = getNode(this, value);
		if (node == null)
			return false;

		KdNode parent = node.parent;
		if (parent != null) {
			if (parent.lesser != null && node.equals(parent.lesser)) {
				List<ComparablePoint2D> nodes = getTree(node);
				if (nodes.size() > 0) {
					parent.lesser = createNode(nodes, node.k, node.depth);
					if (parent.lesser != null) {
						parent.lesser.parent = parent;
					}
				} else {
					parent.lesser = null;
				}
			} else {
				List<ComparablePoint2D> nodes = getTree(node);
				if (nodes.size() > 0) {
					parent.greater = createNode(nodes, node.k, node.depth);
					if (parent.greater != null) {
						parent.greater.parent = parent;
					}
				} else {
					parent.greater = null;
				}
			}
		} else {
			// root
			List<ComparablePoint2D> nodes = getTree(node);
			if (nodes.size() > 0)
				root = createNode(nodes, node.k, node.depth);
			else
				root = null;
		}

		return true;
	}

	/**
	 * Get the (sub) tree rooted at root.
	 * 
	 * @param root
	 *            of tree to get nodes for.
	 * @return points in (sub) tree, not including root.
	 */
	private static final List<ComparablePoint2D> getTree(KdNode root) {
		List<ComparablePoint2D> list = new ArrayList<ComparablePoint2D>();
		if (root == null)
			return list;

		if (root.lesser != null) {
			list.add(root.lesser.id);
			list.addAll(getTree(root.lesser));
		}
		if (root.greater != null) {
			list.add(root.greater.id);
			list.addAll(getTree(root.greater));
		}

		return list;
	}

	/**
	 * K Nearest Neighbor search
	 * 
	 * @param K
	 *            Number of neighbors to retrieve. Can return more than K, if
	 *            last nodes are equal distances.
	 * @param value
	 *            to find neighbors of.
	 * @return collection of T neighbors.
	 */
	@SuppressWarnings("unchecked")
	public Collection<T> nearestNeighbourSearch(int K, T value) {
		if (value == null)
			return null;

		// Map used for results
		TreeSet<KdNode> results = new TreeSet<KdNode>(new EuclideanComparator(value));

		// Find the closest leaf node
		KdNode prev = null;
		KdNode node = root;
		while (node != null) {
			if (KdNode.compareTo(node.depth, node.k, value, node.id) <= 0) {
				// Lesser
				prev = node;
				node = node.lesser;
			} else {
				// Greater
				prev = node;
				node = node.greater;
			}
		}
		KdNode leaf = prev;

		if (leaf != null) {
			// Used to not re-examine nodes
			Set<KdNode> examined = new HashSet<KdNode>();

			// Go up the tree, looking for better solutions
			node = leaf;
			while (node != null) {
				// Search node
				searchNode(value, node, K, results, examined);
				node = node.parent;
			}
		}

		// Load up the collection of the results
		Collection<T> collection = new ArrayList<T>(K);
		for (KdNode kdNode : results) {
			collection.add((T) kdNode.id);
		}
		return collection;
	}

	private static final <T extends ComparablePoint2D> void searchNode(T value, KdNode node, int K,
			TreeSet<KdNode> results, Set<KdNode> examined) {
		examined.add(node);

		// Search node
		KdNode lastNode = null;
		Double lastDistance = Double.MAX_VALUE;
		if (results.size() > 0) {
			lastNode = results.last();
			lastDistance = lastNode.id.euclideanDistance(value);
		}
		Double nodeDistance = node.id.euclideanDistance(value);
		if (nodeDistance.compareTo(lastDistance) < 0) {
			if (results.size() == K && lastNode != null)
				results.remove(lastNode);
			results.add(node);
		} else if (nodeDistance.equals(lastDistance)) {
			results.add(node);
			//results.remove(node);TODO
		} else if (results.size() < K) {
			results.add(node);
		}
		lastNode = results.last();
		lastDistance = lastNode.id.euclideanDistance(value);
		//System.out.println(lastDistance );
		int axis = node.depth % node.k;
		KdNode lesser = node.lesser;
		KdNode greater = node.greater;

		// Search children branches, if axis aligned distance is less than
		// current distance
		if (lesser != null && !examined.contains(lesser)) {
			examined.add(lesser);

			double nodePoint = Double.MIN_VALUE;
			double valuePlusDistance = Double.MIN_VALUE;
			if (axis == X_AXIS) {
				nodePoint = node.id.get(0);
				valuePlusDistance = value.get(0) - lastDistance;
			} else {
				nodePoint = node.id.get(1);
				valuePlusDistance = value.get(1) - lastDistance;
			}
			boolean lineIntersectsCube = ((valuePlusDistance <= nodePoint) ? true : false);

			// Continue down lesser branch
			if (lineIntersectsCube)
				searchNode(value, lesser, K, results, examined);
		}
		if (greater != null && !examined.contains(greater)) {
			examined.add(greater);

			double nodePoint = Double.MIN_VALUE;
			double valuePlusDistance = Double.MIN_VALUE;
			if (axis == X_AXIS) {
				nodePoint = node.id.get(0);
				valuePlusDistance = value.get(0) + lastDistance;
			} else {
				nodePoint = node.id.get(1);
				valuePlusDistance = value.get(1) + lastDistance;
			}
			boolean lineIntersectsCube = ((valuePlusDistance >= nodePoint) ? true : false);

			// Continue down greater branch
			if (lineIntersectsCube)
				searchNode(value, greater, K, results, examined);
		}
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String toString() {
		return TreePrinter.getString(this);
	}

	protected static class EuclideanComparator implements Comparator<KdNode> {

		private ComparablePoint2D point = null;

		public EuclideanComparator(ComparablePoint2D point) {
			this.point = point;
		}

		/**
		 * {@inheritDoc}
		 */
		@Override
		public int compare(KdNode o1, KdNode o2) {
			Double d1 = point.euclideanDistance(o1.id);
			Double d2 = point.euclideanDistance(o2.id);
			if (d1.compareTo(d2) < 0)
				return -1;
			else if (d2.compareTo(d1) < 0)
				return 1;
			return o1.id.compareTo(o2.id);
		}
	};

	public static class KdNode implements Comparable<KdNode> {

		private int k = 2;
		private int depth = 0;
		private ComparablePoint2D id = null;
		private KdNode parent = null;
		private KdNode lesser = null;
		private KdNode greater = null;

		public KdNode(ComparablePoint2D id) {
			this.id = id;
		}

		public KdNode(int k, int depth, ComparablePoint2D id) {
			this(id);
			this.k = k;
			this.depth = depth;
		}

		public static int compareTo(int depth, int k, ComparablePoint2D o1, ComparablePoint2D o2) {
			int axis = depth % k;
			if (axis == X_AXIS)
				return ComparablePoint2D.X_COMPARATOR.compare(o1, o2);
			return ComparablePoint2D.Y_COMPARATOR.compare(o1, o2);
		}

		/**
		 * {@inheritDoc}
		 */
		@Override
		public boolean equals(Object obj) {
			if (obj == null)
				return false;
			if (!(obj instanceof KdNode))
				return false;

			KdNode kdNode = (KdNode) obj;
			if (this.compareTo(kdNode) == 0)
				return true;
			return false;
		}

		/**
		 * {@inheritDoc}
		 */
		@Override
		public int compareTo(KdNode o) {
			return compareTo(depth, k, this.id, o.id);
		}

		/**
		 * {@inheritDoc}
		 */
		@Override
		public String toString() {
			StringBuilder builder = new StringBuilder();
			builder.append("k=").append(k);
			builder.append(" depth=").append(depth);
			builder.append(" id=").append(id.toString());
			return builder.toString();
		}
	}


	protected static class TreePrinter {

		public static <T extends ComparablePoint2D> String getString(KDTree2D<T> tree) {
			if (tree.root == null)
				return "Tree has no nodes.";
			return getString(tree.root, "", true);
		}

		private static String getString(KdNode node, String prefix, boolean isTail) {
			StringBuilder builder = new StringBuilder();

			if (node.parent != null) {
				String side = "left";
				if (node.parent.greater != null && node.id.equals(node.parent.greater.id))
					side = "right";
				builder.append(prefix + (isTail ? "L__" : "|__ ") + "[" + side + "] " + "depth=" + node.depth + " id="
						+ node.id + "\n");
			} else {
				builder.append(prefix + (isTail ? "L__ " : "|__ ") + "depth=" + node.depth + " id=" + node.id + "\n");
			}
			List<KdNode> children = null;
			if (node.lesser != null || node.greater != null) {
				children = new ArrayList<KdNode>(2);
				if (node.lesser != null)
					children.add(node.lesser);
				if (node.greater != null)
					children.add(node.greater);
			}
			if (children != null) {
				for (int i = 0; i < children.size() - 1; i++) {
					builder.append(getString(children.get(i), prefix + (isTail ? "    " : "|   "), false));
				}
				if (children.size() >= 1) {
					builder.append(getString(children.get(children.size() - 1), prefix + (isTail ? "    " : "|   "),
							true));
				}
			}

			return builder.toString();
		}
	}
}




