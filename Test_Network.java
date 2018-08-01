import java.io.*;
/*
This is the main program.

Current this program serves for digit recognition. But this can be changed easily.
Currently the network configuration is: 3 convolution + 3 subsampling layers. Input is 36*36 grayscale image.
The dimension of images per layer:  36 -C-> 32 -S-> 16 -C-> 12 -S-> 6 -C-> 2 -S-> 1 -Classify->
The number of images per layer:     1       5       5      20      20     60     60
where -C-> means convolution, -S-> means subsampling

All input text files for training are expected to be named as 0_0, 0_1, 0_2 (i.e the images for 0) 1_0, 1_1, 2_3 etc
Also expect a file that stores the parameters for the network. File must be named parameters.dat.
The parameter file is assumed to be in the same path as this project. The data file is assumed to be in 0_data/0_0, 
1_data/1_5 etc

Training result will be output to the file parameters.dat, overwritting the existing data
*/
public class Test_Network{
	public static void main(String args[]) throws IOException {
	// The number of convolution layers	
		int num_of_conv_layers=3;
	// The dimension of the label vector
		int dim_label=10;
	// The number of images for each convolution layer
		int[] num_of_images={6, 16, 60};
	// The dimension for each image layer
		int[] image_dim={36, 32, 16, 12, 6, 2, 1};
	// The maximum number a random number generator will create
		double bound=1.0;
				
		double step_size=0.01;
	//	initializeParameterFile(num_of_conv_layers, dim_label, num_of_images, image_dim, bound, "parameters.dat");
		Network nk=buildNetwork("full_para.dat", step_size);
		System.out.println("Test network. Result is "+nk.test(args[0]));
	}
	
	public static void initializeParameterFile(int nocl, int dl, int[] noi, int[] id, double b, String filename) throws IOException {
		Initializer in=new Initializer(nocl, dl, noi, id, b);
		in.init(filename);
	}
	
	public static Network buildNetwork(String filename, double step_size){
		InfoReader ir=new InfoReader(filename);
		return new Network(ir, step_size);
	}
	
//
// Train the network and output the parameters to an output file. train_count determines the number of iterations to train
//
	public static void train(Network nk, int train_count, String output_filename) throws IOException {
		//Input files are i_data/i_j
		int k=0;
		for(int i=1;i<=train_count;i++){
			int j=(int)(Math.random()*5000);
			k=(k+1)%10;
			nk.train("digit_data_081127\\train_set_081127\\"+k+"_data\\b_"+k+"_"+j+".txt");		
		}
		nk.outputParameters(output_filename, 1);
	}

//
// Test the network and output the parameters to an output file. test_count determines the number of iterations to train
//	
	public static void test(Network nk, int test_count) throws IOException{
		int[] correct=new int[10];
		for(int i=1;i<=test_count;i++){
			int j=(int)(Math.random()*800);
			if(i%100==0){
				System.out.println("Finish some samples");
			}
			for(int k=0;k<=9;k++){	
				int type=nk.test("digit_data_081127\\test_set_081127\\"+k+"_data\\b_"+k+"_"+j+".txt");
				if(type==k){
					correct[k]++;
				}			
			}
		}
		for(int i=0;i<10;i++){
			System.out.println("Correct percentage for digit "+i+" is "+(double)correct[i]/test_count);
		}
	}

	public static void print(double[] x){
		for(int i=0;i<x.length;i++){
			System.out.print(x[i]+" ");
		}
		System.out.println();
	}
}