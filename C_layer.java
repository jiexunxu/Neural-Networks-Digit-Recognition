//
// The convolution layer
//
public class C_layer{
//Input images. Make sure each entry is NOT null!!!!!!!!!	
	public image[] in;
//Output images. Make sure each entry is NOT null!!!!!!!!!	
	public image[] out;
//An array of kernel vectors. dim(kernels)=dim_in*dim_out. The dimension of each kernel vector is conv_n*conv_n.
//Rule for convolution: First dim_in kernel vectors map to out[0], second dim_in vectors map to out[1] etc
	vector[] kernels; 
//An array of trainable biases, dim(bias)=dim(out)		
	double[] bias;	
//Convolution window size		
	int conv_n; 
//Number of input and output images		
	int dim_in;
	int dim_out;
//Dimension of the input and output image	
	int dim_img_in;
	int dim_img_out;
//Step size used for back propagation
	double step_size;

//
//Constructor
//	
	public C_layer(image[] input, image[] output, vector[] ker, double[] bs, double ss){
		in=input;
		out=output;
		kernels=ker;
		bias=bs;
		conv_n=(int)(Math.sqrt(kernels[0].dim));
		dim_in=in.length;
		dim_out=out.length;
		dim_img_in=in[0].dim;
		dim_img_out=out[0].dim;
		step_size=ss;
	}

//
// Forward propagation 
//	
	public void fprop(){
		image[][] images=unfold_all_input_images();
		for(int i=0;i<dim_out;i++){
			fprop(images, i);
		
		//		System.out.println("Fprop result for conv layer for output "+i);
		//		System.out.println(out[i]);
			
			
		}
	}

//
// Backward propagation
//
	public vector bprop(vector dx){
		//compute Jacobian		
		Matrix g1=compute_partial_derivative_X();
		Matrix g2=compute_partial_derivative_kernels();
		Matrix g3=compute_partial_derivative_bias();
		vector dx2=g1.product(dx);
		//update kernels and bias
		vector t1=new vector(kernels);
		vector t2=new vector(bias);
		vector tt1=g2.product(dx);
		vector tt2=g3.product(dx);
		tt1.multiply(-step_size);
		tt2.multiply(-step_size);
		t1.add(tt1);
		t2.add(tt2);
		kernels=t1.arrayForm(kernels[0].dim);
		bias=t2.arrayForm();
		return dx2;
	}
	
//
// Matrix dimension is #rows: dim_img_out*dim_img_out*dim_out
// #columns: dim_img_in*dim_img_in*dim_in
//
	public Matrix compute_partial_derivative_X(){
		Matrix temp=new Matrix(dim_img_out*dim_img_out*dim_out, dim_img_in*dim_img_in*dim_in, true);
		// For each function yi(i.e for each row)
		for(int i=0;i<temp.getRow();i++){
			InfoStorer ifs=bprop_get_yindex_info(i);
			//First compute the sum of dot product
			double sum_dproduct=0.0;
			for(int j=0;j<dim_in;j++){
				vector x_vector=imageToVector(j, ifs.offset_in_x, ifs.offset_in_y, conv_n);
				vector ker_vector=kernels[ifs.kers[j]];
				sum_dproduct=sum_dproduct+x_vector.dot_product(ker_vector);
			}
			//Then the secant squared term
			double der_mul=hyperbolic_tangent_sigmoid_derivative(sum_dproduct+bias[(int)(i/(dim_img_out*dim_img_out))]);
			//For each of the images in in[j]
			for(int j=0;j<dim_in;j++){
				vector x_vector=imageToVector(j, ifs.offset_in_x, ifs.offset_in_y, conv_n);
				vector ker_vector=kernels[ifs.kers[j]];
				//For the particular convolution window corresponding to yi in image in[j], the start position
				//is ifs.offset_in_x and ifs.offset_in_y, and the associated kernel vector is kernels[ifs.kers[j]]
				for(int p=ifs.offset_in_y;p<ifs.offset_in_y+conv_n;p++){
					for(int q=ifs.offset_in_x;q<ifs.offset_in_x+conv_n;q++){
						//get the column index for current pixel
						int x_index=j*dim_img_in*dim_img_in+p*dim_img_in+q;
						temp.set(i, x_index, der_mul*ker_vector.get((p-ifs.offset_in_y)*conv_n+q-ifs.offset_in_x));						
					}
				}				
			}
		}
		return temp;
	}

//
// Matrix dimension is #rows: dim_img_out*dim_img_out*dim_out
// #columns: dim_in*dim_out*conv_n*conv_n
//
	public Matrix compute_partial_derivative_kernels(){
		Matrix temp=new Matrix(dim_img_out*dim_img_out*dim_out, dim_in*dim_out*conv_n*conv_n, true);
		// For each function yi(i.e for each row)
		for(int i=0;i<temp.getRow();i++){
			InfoStorer ifs=bprop_get_yindex_info(i);
			//First compute the sum of dot product
			double sum_dproduct=0.0;
			for(int j=0;j<dim_in;j++){
				vector x_vector=imageToVector(j, ifs.offset_in_x, ifs.offset_in_y, conv_n);
				vector ker_vector=kernels[ifs.kers[j]];
				sum_dproduct=sum_dproduct+x_vector.dot_product(ker_vector);
			}
			//Then the secant squared term
			double der_mul=hyperbolic_tangent_sigmoid_derivative(sum_dproduct+bias[(int)(i/(dim_img_out*dim_img_out))]);
			//For this one Y pixel value, there are dim_in kernel vectors associated with it, and these vectors are 
			//kernels[ifs.kers[w]], for all 0<=w<dim_in, where each kernel vector is associated with in[w]
			for(int j=0;j<dim_in;j++){
				vector x_vector=imageToVector(j, ifs.offset_in_x, ifs.offset_in_y, conv_n);
				vector ker_vector=kernels[ifs.kers[j]];
				//For the particular convolution window corresponding to yi in image in[j], the start position
				//is ifs.offset_in_x and ifs.offset_in_y, and the associated kernel vector is kernels[ifs.kers[j]],
				//which starts at (0,0) of course
				for(int p=0;p<conv_n;p++){
					for(int q=0;q<conv_n;q++){
						//get the column index for current pixel
						int ker_index=ifs.image_index*dim_in*conv_n*conv_n+j*conv_n*conv_n+p*conv_n+q;
						temp.set(i, ker_index, der_mul*x_vector.get(p*conv_n+q));							
					}
				}				
			}
		}
		return temp;
	}

//
// Matrix dimension is #rows: dim_out
// #columns: dim_in*dim_out*conv_n*conv_n
//
	public Matrix compute_partial_derivative_bias(){
		Matrix temp=new Matrix(dim_img_out*dim_img_out*dim_out, dim_out, true);
		// For each function yi(i.e for each row)
		for(int i=0;i<temp.getRow();i++){
			InfoStorer ifs=bprop_get_yindex_info(i);
			//First compute the sum of dot product
			double sum_dproduct=0.0;
			for(int j=0;j<dim_in;j++){
				vector x_vector=imageToVector(j, ifs.offset_in_x, ifs.offset_in_y, conv_n);
				vector ker_vector=kernels[ifs.kers[j]];
				sum_dproduct=sum_dproduct+x_vector.dot_product(ker_vector);
			}
			//Then the secant squared term
			double der_mul=hyperbolic_tangent_sigmoid_derivative(sum_dproduct+bias[(int)(i/(dim_img_out*dim_img_out))]);
			//Now set the value for column bias[ifs.image_index] at row i to be der_mul
			temp.set(i, ifs.image_index, der_mul);
		}
		return temp;
	}
	
//
// Convert in[img_index] starting at (p, q) and window size l to a vector
//	
	public vector imageToVector(int img_index, int p, int q, int l){
		vector temp=new vector(l*l);
		int ctr=0;
		for(int i=0;i<l;i++){
			for(int j=0;j<l;j++){
				temp.set(ctr, in[img_index].get(q+i, p+j));
				ctr++;
			}
		}
		return temp;
	}
	
	public InfoStorer bprop_get_yindex_info(int y_index){
		InfoStorer ifs=new InfoStorer();
		ifs.image_index=(int)(y_index/(dim_img_out*dim_img_out));
		int temp=y_index-ifs.image_index*dim_img_out*dim_img_out;
		ifs.offset_out_x=temp%dim_img_out;
		ifs.offset_out_y=(int)(temp/dim_img_out);
		ifs.offset_in_x=ifs.offset_out_x;
		ifs.offset_in_y=ifs.offset_out_y;
		int[] ker_indices=new int[dim_in];
		int start_index=ifs.image_index*dim_in;
		for(int i=0;i<ker_indices.length;i++){
			ker_indices[i]=start_index+i;
		}
		ifs.kers=ker_indices;
		return ifs;
	}
//
//Unfold all images in the input image vector and return a array of array of images
//
	public image[][] unfold_all_input_images(){
		image[][] temp=new image[dim_in][];
		for(int i=0;i<dim_in;i++){
			temp[i]=in[i].unfold(conv_n);
		}
		return temp;
	}
	
//
// Forward propagation on out[n]. imgs is the unfolded images for all in. image[i][] is unfolded image for in[i]
//	
	public void fprop(image[][] imgs, int n){
//Clear all the entries in out[n]
		for(int p=0;p<dim_img_out;p++){
			for(int q=0;q<dim_img_out;q++){				
				out[n].set(p, q, 0.0);				
			}
		}	
//			
//Start of the kernel vectors used for the fprop for this output slot		
//
		int ker_start=n*dim_in;			
		double sum=0.0;				
		for(int i=0;i<dim_in;i++){
			vector kernel=kernels[ker_start+i];	
			int j=0;	
			for(int p=0;p<dim_img_out;p++){
				for(int q=0;q<dim_img_out;q++){
					vector temp=bulidVectorFromImage(imgs[i][j]);
					out[n].acc(p, q, temp.dot_product(kernel));
					j++;
				}
			}			
		}
		
//		
//Add the bias to each entry in out[n], then 			
//
		for(int p=0;p<dim_img_out;p++){
			for(int q=0;q<dim_img_out;q++){				
				out[n].acc(p, q, bias[n]);	
				out[n].set(p, q, hyperbolic_tangent_sigmoid(out[n].get(p, q)));								
			}
		}	
	}

//
//Build a vector from a image. Vector is in	row major order
//
	public vector bulidVectorFromImage(image im){
		int dim=im.dim;
		vector temp=new vector(dim*dim);
		int ctr=0;
		for(int i=0;i<dim;i++){
			for(int j=0;j<dim;j++){
				temp.set(ctr, im.get(i, j));
				ctr++;
			}
		}
		return temp;
	}

//
//The hyperbolic tangent sigmoid function
//	
	public static double hyperbolic_tangent_sigmoid(double x){
		double etox=Math.pow(Math.E, x);
		double etongx=Math.pow(Math.E, -x);
		return (etox-etongx)/(etox+etongx);
	}
	
//
// The derivative hyperbolic tangent sigmoid function
//
	public static double hyperbolic_tangent_sigmoid_derivative(double x){
		return 1-Math.pow(hyperbolic_tangent_sigmoid(x), 2.0);
	}
	
	public static void print(double[] a){
		for(int i=0;i<a.length;i++){
			System.out.print(Math.round(a[i]*1000)/1000.0+" ");
		}
		System.out.println();
	}
}