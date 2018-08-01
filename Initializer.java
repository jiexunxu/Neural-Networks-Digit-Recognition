import java.io.*;
import java.util.*;
//
// Initialize the entire project by assigning random values between 0 and 1 to all kernels, biases and multipliers
// Output this initial file
//
public class Initializer{
// The number of convolution layers	
	int num_of_conv_layers;
// The dimension of the label vector
	int dim_label;
// The number of images for each convolution layer
	int[] num_of_images;
// The dimension for each image layer
	int[] image_dim;
// The maximum number a random number generator will create
	double bound;
	
//
// Constructor
//
	public Initializer(int nocl, int dl, int[] noi, int[] id, double b){
		num_of_conv_layers=nocl;
		dim_label=dl;
		num_of_images=noi;
		image_dim=id;
		bound=b;
	}

//
// Creates an output .dat file
//	
	public void init(String filename) throws IOException{
		PrintWriter output=new PrintWriter(new File(filename));
		output.println(1);
		output.println(dim_label);
		output.println(num_of_conv_layers);
		// Print information related to creating images
		for(int i=0;i<num_of_conv_layers;i++){
			output.print(num_of_images[i]+" ");
		}
		output.println();
		for(int i=0;i<num_of_conv_layers*2+1;i++){
			output.print(image_dim[i]+" ");
		}
		output.println();
		// Print out all the kernels
		for(int i=0;i<num_of_conv_layers;i++){
			int kernel_count=num_of_images[i];
			if(i>0){
				kernel_count=kernel_count*num_of_images[i-1];
			}
			for(int j=0;j<kernel_count;j++){
				int kernel_dim=(image_dim[i*2]-image_dim[i*2+1]+1)*(image_dim[i*2]-image_dim[i*2+1]+1);
				for(int k=0;k<kernel_dim;k++){
					output.print(rand()+" ");
				}				
				output.println();
			}
			output.println();
		}
		output.println();
		//Print out all the bias vectors
		for(int i=0;i<num_of_conv_layers*2;i++){
			int bias_dim;
			if(i==num_of_conv_layers*2){
				bias_dim=num_of_images[(i-1)/2];
			}else{
				bias_dim=num_of_images[i/2];
			}			
			for(int j=0;j<bias_dim;j++){
				output.print(rand()+" ");
			}
			output.println();
		}
		for(int i=0;i<dim_label;i++){
			output.print(rand()+" ");
		}
		output.println();
		//Print out all the scalar multipliers
		for(int i=0;i<num_of_images.length;i++){
			for(int j=0;j<num_of_images[i];j++){
				output.print(rand()+" ");
			}
			output.println();
		}
		output.println();
		//Print out all the weights
		for(int i=0;i<dim_label*num_of_images[num_of_images.length-1];i++){
			output.print(rand()+" ");
		}
		output.close();
	}
	
	public double rand(){
		return Math.random()*bound*0.8-0.4;
	}
}