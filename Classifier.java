//
// The last layer in the network. Used to classify input
//
public class Classifier{
// Input image from the subsampling layer
	image[] in_imgs;
// Input to this layer. Converted from 1*1 images	
	double[] in;
// Output of this layer. A weighted sum of all in[i]. Will be used to compute the Ecildean distance between out and Y to
// train the network, or the distance between out and all possible Y and pick the minimum as the class of original input
// image	
	double[] out;
// Weights that are used to compute the weighted sum. Dimension is in_dim*lb_dim. First in_dim corresponds to weights for
// out[0], next in_dim corresponds to out[1] etc
	double[] weights;
// Bias vector for each out[i]. Dimension equals lb_dim and every entry in this vector is the same
	double[] bias;		
// The label of the original input . If label is null, then this layer should output the class of original input image
	public int[] Y;
// Number of input images
	int in_dim;	
// Dimension of the label	
	int lb_dim;	
// Step size used for updating
	double step_size;		
		
//
// Constructor. 
//		
	public Classifier(image[] imgs, double[] w, double[] bs, int[] y, int label_dim, double ss){		
		in_dim=imgs.length;
		in_imgs=imgs;
		// Convert input images to doubles
		in=new double[in_dim];
		Y=y;
		lb_dim=label_dim;
		out=new double[lb_dim];
		weights=w;
		bias=bs;
		step_size=ss;
	}
	
//
// Forward propagation. Used to calculate the results in out	
//
	public void fprop(){
		for(int i=0;i<lb_dim;i++){
			fprop(i);
		}
	}

//
// Backward propagation	and updating, return the partial derivative vector with respect to X. Here dx is computed
// from the compute_energy_dx method
//
	public vector bprop(vector dx){
		Matrix g1=compute_partial_derivative_X();
		Matrix g2=compute_partial_derivative_weights();
		Matrix g3=compute_partial_derivative_bias();
		vector dx2=g1.product(dx);
		//update weights and bias
		vector t1=new vector(weights);
		vector t2=new vector(bias);
		vector tt1=g2.product(dx);
		vector tt2=g3.product(dx);
		tt1.multiply(-step_size);
		tt2.multiply(-step_size);
		t1.add(tt1);
		t2.add(tt2);
		weights=t1.arrayForm();
		bias=t2.arrayForm();
		return dx2;
	}
	

//
// Compute and return the dx vector from the energy function, i.e Eclidean distance function
//
	public vector compute_energy_dx(){
		vector temp=new vector(lb_dim);
		for(int i=0;i<lb_dim;i++){
			temp.set(i, 2*(out[i]-Y[i]));
		}
		return temp;
	}
	
//	
// Forward propagation for out[n]. Make sure imgs[i] is 1*1 pixel image!
//
	public void fprop(int n){
		for(int i=0;i<in_imgs.length;i++){
			in[i]=in_imgs[i].get(0, 0);
		}
		// The starting point for the weights
		int start_weight=in_dim*n;
		double sum=0.0;
		// Dot product of weights and in
		for(int i=0;i<in_dim;i++){
			sum=sum+weights[start_weight+i]*in[i];
		}
		// Add a bias, pass the result through a sigmoid, and store the result in out[n]
		out[n]=C_layer.hyperbolic_tangent_sigmoid(sum+bias[n]);
	}
	
//
// Compute and return the Jacobian matrix of loss function with respect to X
//	
	public Matrix compute_partial_derivative_X(){
		Matrix temp=new Matrix(lb_dim, in_dim, true);
		vector[] weights_vector=weights_vectorform();
		vector in_vector=new vector(in);
		for(int i=0;i<temp.getRow();i++){
			double mul=C_layer.hyperbolic_tangent_sigmoid_derivative(weights_vector[i].dot_product(in_vector)+bias[i]);
			weights_vector[i].multiply(mul);
			for(int j=0;j<weights_vector[i].dim;j++){
				temp.set(i, j, weights_vector[i].get(j));
			}			
		}
		return temp;
	}

//
// Convert the weights array into a vector form. Each weight vector have dimension in_dim	
//
	public vector[] weights_vectorform(){
		vector[] temp=new vector[lb_dim];
		int ctr=0;
		for(int i=0;i<lb_dim;i++){
			temp[i]=new vector(in_dim);
			for(int j=0;j<in_dim;j++){
				temp[i].set(j, weights[ctr]);
				ctr++;
			}
		}
		return temp;
	}

//
// Compute and return the Jacobian matrix of loss function with respect to weights matrix. This Jacobian matrix is 
// converted to an array of double so that it is easy to update weights
//
	public Matrix compute_partial_derivative_weights(){
		Matrix temp=new Matrix(lb_dim, lb_dim*in_dim, true);
		vector[] weights_vector=weights_vectorform();		
		for(int i=0;i<temp.getRow();i++){
			vector in_vector=new vector(in);
			int start_index=i*in_dim;
			double mul=C_layer.hyperbolic_tangent_sigmoid_derivative(weights_vector[i].dot_product(in_vector)+bias[i]);			
			in_vector.multiply(mul);		
			for(int j=start_index;j<start_index+in_dim;j++){
				temp.set(i, j, in_vector.get(j-start_index));
			}
		}
		return temp;
	}
	
//
// Compute and return the Jacobian for the bias
//	
	public Matrix compute_partial_derivative_bias(){
		Matrix temp=new Matrix(lb_dim, lb_dim, true);
		vector[] weights_vector=weights_vectorform();
		vector in_vector=new vector(in);
		for(int i=0;i<lb_dim;i++){
			double mul=C_layer.hyperbolic_tangent_sigmoid_derivative(weights_vector[i].dot_product(in_vector)+bias[i]);
			temp.set(i, i, mul);
		}
		return temp;
	}
	
//
// Compute the loss using the Eclidean distance squared. Make sure Y is not null!!!!!!
//
	public double compute_loss(){
		return compute_dist(Y);
	}
	
//
// Classify the input image by finding the minimum Eclidean distance. Return the class of minimum distance to out
//
	public int[] classify(){
		// Create and initialize the class	
		int[] temp=new int[lb_dim];
		temp[0]=1;
		for(int i=1;i<lb_dim;i++){
			temp[i]=-1;
		}
		// Stores the min distance and the corresponding class. min_class store the position where +1 is stored
		double min_dist=compute_dist(temp);
		int min_class=0;
		// Compute all Ecildean distances and pick the minimum
		for(int i=1;i<lb_dim;i++){
			temp[i]=1;
			temp[i-1]=-1;
			double dist=compute_dist(temp);
			if(dist<min_dist){
				min_dist=dist;
				min_class=i;
			}
		}
		// Now temp will store the closest class. Return temp
		temp[lb_dim-1]=-1;
		temp[min_class]=1;
		return temp;
	}

//
// Compute the Ecildean distance squared between out and target
//	
	public double compute_dist(int[] target){
		double dist=0.0;
		for(int i=0;i<lb_dim;i++){
			double dif=Math.pow(out[i]-(double)(target[i]), 2.0);
			dist=dist+dif*dif;
		}
		return dist;
	}
	
	public static void print(double[] a){
		for(int i=0;i<a.length;i++){
			System.out.print(a[i]+" ");
		}
		System.out.println();
	}
}