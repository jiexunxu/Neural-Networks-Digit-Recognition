//
// The subsampling layer
//
public class S_layer{
//Input images. Make sure each entry is NOT null!!!!!!!!!	
	image[] in;
//Output images. Make sure each entry is NOT null!!!!!!!!!	
	image[] out;
//An array of trainable biases, dim(bias)=dim_in		
	double[] bias;
//An array of the scalar multiplied to the subsampled images. Dim(dev)=dim_in
	double[] dev;	
//The fraction to shrink the images
	int step;	
//Number of input images, which equals the number of output images		
	int dim_in;
//Dimension of the input and output image	
	int dim_img_in;
	int dim_img_out;
// The step size used to updating the dev and bias
	double step_size;

//
// Constructor
//
	public S_layer(image[] input, image[] output, double[] dv, double[] bs, double ss){
		in=input;
		out=output;
		dev=dv;
		bias=bs;
		dim_in=in.length;
		dim_img_in=in[0].dim;
		dim_img_out=out[0].dim;
		step=dim_img_in/dim_img_out;
		step_size=ss;
	}

//
// Forward propagation
//
	public void fprop(){
		for(int i=0;i<dim_in;i++){
			fprop(i);
		//	System.out.println("Fprop result for subsample layer for output "+i);
		//	System.out.println(out[i]);
		}
	}
	
//
// Backward propagation and updating all bias and scalar multipliers in this layer. 
// In this layer, there are dim_in input images and output images. For each input image, dimension is dim_img_in,
// so X is a dim_img_in*dim_img_in*dim_in dimensional vector, and output Y is dim_img_out*dim_img_out*dim_in dimensional
// vector. X is ordered as follows: First dim_img_in entries strating at X[0] corresponds to first row of in[0], aonther 
// dim_img_in entries starting at X[dim_img_in] corresponds to second row of in[0], ...... first dim_img_in entries
// entries starting at X[dim_img_in*dim_img_in] corresponds to first row in in[1]...... Same for Y. 
//  
	public vector bprop(vector dx){
		//compute Jacobian
		Matrix g1=compute_partial_derivative_X();
		Matrix g2=compute_partial_derivative_dev();
		Matrix g3=compute_partial_derivative_bias();
		vector dx2=g1.product(dx);
		//update dev and bias
		vector t1=new vector(dev);
		vector t2=new vector(bias);
		vector tt1=g2.product(dx);
		vector tt2=g3.product(dx);
		tt1.multiply(-step_size);
		tt2.multiply(-step_size);
		t1.add(tt1);
		t2.add(tt2);
		dev=t1.arrayForm();
		bias=t2.arrayForm();	
		return dx2;
	}

//
// Compute the Jacobian matrix of the partial derivative of the scalar multipliers. Since there are dim_img_in*
// dim_img_in*dim_in X, so the Jacobian matrix is dim_img_in*dim_img_in*dim_in columns and 
// dim_img_out*dim_img_out*dim_in rows. Note that for one particular y, there are step number of x associated with it.
// so the derivatives of these x with respect to this y are non zero, others are all zero. FOr a particular dev, all 
// derivatives with respect to x is the same.
//
	public Matrix compute_partial_derivative_X(){
		Matrix temp=new Matrix(dim_img_out*dim_img_out*dim_in, dim_img_in*dim_img_in*dim_in, true);
		for(int i=0;i<dim_img_out*dim_img_out*dim_in;i++){
			//image_index range from 0 to dim_in-1
			int image_index=(int)(i/(dim_img_out*dim_img_out));	
			int offset_total=i-image_index*dim_img_out*dim_img_out;
			// the x, y position for y_index in out[image_index];
			int offset_x=offset_total%dim_img_out;
			int offset_y=(int)(offset_total/dim_img_out);
			// Get the start position (upper left corner) for the square (of size step) in in[image_index] to average
			int offset_in_x=offset_x*step;
			int offset_in_y=offset_y*step;
			//compute the correct entry (i.e derivative)
			double x_ave=bprop_compute_average_x(i, image_index);
			double result=C_layer.hyperbolic_tangent_sigmoid_derivative(dev[image_index]*x_ave+bias[image_index])*dev[image_index];
			//set the value to be result(i.e non zero) for the correct entries in X
			for(int p=0;p<step;p++){
				for(int q=0;q<step;q++){
					temp.set(i, get_X_index(offset_in_x+p, offset_in_y+q, image_index), result);
				}
			}			
		}
		return temp;
	}
	
//
// Compute the Jacobian matrix of the partial derivative of the scalar multipliers. Since there are dim_in scalar 
// multipliers, so the Jacobian matrix is dim_in columns and dim_img_out*dim_img_out*dim_in rows. Note that for dev[i],
// only image_index==i entries are nonzero	
//
	public Matrix compute_partial_derivative_dev(){
		Matrix temp=new Matrix(dim_img_out*dim_img_out*dim_in, dim_in, true);
		for(int i=0;i<temp.getRow();i++){
			//image_index range from 0 to dim_in-1
			int image_index=(int)(i/(dim_img_out*dim_img_out));
			for(int j=0;j<temp.getColumn();j++){
				if(image_index!=j){
					continue;
				}
				// compute the average of some pixels in in[image_index] at particular pixel in out[image_index]
				double x_ave=bprop_compute_average_x(i, image_index);				
				temp.set(i, j, C_layer.hyperbolic_tangent_sigmoid_derivative(dev[j]*x_ave+bias[j])*x_ave);
			}
		}
		return temp;
	}
	
//
// Compute and return Jacobian matrix of the bias. This matrix is same dimension as Jacobian for dev.
//	
	public Matrix compute_partial_derivative_bias(){
		Matrix temp=new Matrix(dim_img_out*dim_img_out*dim_in, dim_in, true);
		for(int i=0;i<temp.getRow();i++){
			//image_index range from 0 to dim_in-1
			int image_index=(int)(i/(dim_img_out*dim_img_out));
			for(int j=0;j<temp.getColumn();j++){
				if(image_index!=j){
					continue;
				}
				// compute the average of some pixels in in[image_index] at particular pixel in out[image_index]
				double x_ave=bprop_compute_average_x(i, image_index);				
				temp.set(i, j, C_layer.hyperbolic_tangent_sigmoid_derivative(dev[j]*x_ave+bias[j]));
			}
		}
		return temp;
	}
	
//
// returns the average of some pixels in in[image_index] at particular pixel y_index in out[image_index]	
//
	public double bprop_compute_average_x(int y_index, int img_index){
		int offset_total=y_index-img_index*dim_img_out*dim_img_out;
		// the x, y position for y_index in out[image_index];
		int offset_x=offset_total%dim_img_out;
		int offset_y=(int)(offset_total/dim_img_out);
		// Get the start position (upper left corner) for the square (of size step) in in[image_index] to average
		int offset_in_x=offset_x*step;
		int offset_in_y=offset_y*step;
		double sum=0.0;
		for(int i=0;i<step;i++){
			//image_index range from 0 to dim_in-1
			int image_index=(int)(i/(dim_img_out*dim_img_out));
			for(int j=0;j<step;j++){
				sum=sum+in[image_index].get(offset_in_x+i, offset_in_y+j);
			}
		}
		return sum/(step*step);
	}

//
// Given the x, y offset and img_index, compute the corresponding index in X
//
	public int get_X_index(int x_offset, int y_offset, int img_index){
		return img_index*dim_img_in*dim_img_in+y_offset*dim_img_in+x_offset;
	}
		
//
// Forward propagation for in[n]
//	
	public void fprop(int n){
// Subsample window start at (i, j), end at (i+step-1, j+step-1)		
		for(int i=0;i<dim_img_in;i+=step){
			for(int j=0;j<dim_img_in;j+=step){
				double sum=0.0;
// p, q are iterators inside the window				
				for(int p=0;p<step;p++){
					for(int q=0;q<step;q++){
						sum=sum+in[n].get(i+p, j+q);					
					}
				}
				
// Average the sum, multiply by a trainable scalar, add the bias and pass the result through a sigmoid function				
				sum=C_layer.hyperbolic_tangent_sigmoid(dev[n]*(sum/(step*step))+bias[n]);
// Set the corresponding entry in out[n]
				out[n].set(i/step, j/step, sum);
			}
		}
	}
	
	public static void print(double[] a){
		for(int i=0;i<a.length;i++){
			System.out.print(Math.round(a[i]*1000)/1000.0+" ");
		}
		System.out.println();
	}	
}