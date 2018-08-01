import java.util.*;
import java.io.*;
/*
The class that reads all the information required to build a network from a .dat file

A sample .dat file:(parentathis are used for notes, not included in the real .dat file)

1 (This time we load the data from this file and build the network for for training. If this number is 0, then this time
we will test).
10 (The dimension of the final output and the dimension for the label. In this example we are training with digits 0-9, so
this dimension is 10)
3 (number of convolution layers. In this case, 3 convolution layer, 3 subsample layer for a total of 6)
5 20 60 (The number of images for the outputs of each convolution layer. This determines the dimension of the kernel, bias,
dev and weights vectors or matrices. In this example the first kernel will have dimension 5, second 20, third 60)
36 32 16 12 6 2 1 (The dimension of input image for each layer. In this case 6 layers plus a classifier, so 7 numbers)

(We have 3 convolutional layers, we need for each layer, the kernel vectors below. Each line corresponds to a kernel 
vector. Note that each line may have different numbers because kernel vectors in these 3 layers aren't necessarily the same
dimension. In this example, there are 3 sets of kernel vectors. Set 1 for convolution layer 1. Since conv layer 1 has 5 
output, so it has 1*5 lines. Set 2 has 5*20 lines and set 3 has 20*60 lines.)
.1 .2 .1 .3 ......  
.2 .5 .8 .1 ......
...... 

.1 .2 .1 .3 ......  
.2 .5 .8 .1 ......
...... 

.1 .2 .1 .3 ......  
.2 .5 .8 .1 ......
......   

(We have 6 layers plus a classifier, so we need for each of the seven objects, a bias vector. Since 7 objects, so 7 lines.
Now bias for conv layer 1 has dimension 5 because output of layer 1 has dimension 5, so first line has five numbers; 
bias for subsample layer 1 has 5 dimension as well; bias for conv and subsample layer 2 has dimension 20; bias for conv and
subsample layer 3 has dimension 60. Bias for classfier has dimension 60 as well).
.3 .7 .4 .2 .1
.5 .8 .1 .7 .6
......

(Following is the scaler multiplier for each subsampling layer. The format is the same as the bias layer above, but with
only 3 lines: line 1 has 5 numbers, line 2 20, line 3 60)
.2 .5 .7 .8 .1
......

(Following is the weights for the classifier. It's just a line of numbers. Dimension is 60*10)
.3 .4 .1 ......

*/
public class InfoReader{
// Sepcify a scanner to read information
	Scanner reader;
// See if this file is for training or testing
	boolean isTraining;
// The dimension of the label vector
	int dim_label;
// The number of convolution layers	
	int num_of_conv_layers;
// The number of images for each convolution layer
	int[] num_of_images;
// The images for all layers. images_all[0] is input for convolution layer 1, images_all[1] is output for convolution layer
// 1 and input for subsample layer 1, images_all[2] is output for subsample layer 1 and input for convolution layer 2 etc
	image[][] images_all;
// The kernel vectors for all convolutional layers. kernels_all[i][] is the kernel vectors for convolution layer i
	vector[][] kernels_all;
// The bias vector for all layers (convolution, subsample and classify)
	double[][] bias_all;
// The scalar multiplier for all subsampling layers
	double[][] dev_all;
// The weights vector for the classifier
	double[] weights;		

	public InfoReader(String filename){
		try{
			reader=new Scanner(new File(filename));
		}catch(IOException ex){
			System.out.println("The parameter file does not exist!");
			System.exit(1);
		}		
		readCritialInfo();		
		createImages();
		createKernels();
		createBiases();
		createMultipliers();
		createWeights();
	}

//
//  Read the critial informations
//	
	public void readCritialInfo(){
		int temp=reader.nextInt();
		isTraining=false;
		if(temp==1){
			isTraining=true;
		}
		dim_label=reader.nextInt();
		num_of_conv_layers=reader.nextInt();
		num_of_images=new int[num_of_conv_layers];
	}
	
//
// Create the images for the network
//	
	public void createImages(){
		images_all=new image[num_of_conv_layers*2+1][];
		images_all[0]=new image[1];
		int dim;
		for(int i=0;i<num_of_conv_layers;i++){
			dim=reader.nextInt();
			images_all[i*2+1]=new image[dim];
			images_all[i*2+2]=new image[dim];
			num_of_images[i]=dim;
		}
		for(int i=0;i<num_of_conv_layers*2+1;i++){
			dim=reader.nextInt();
			for(int j=0;j<images_all[i].length;j++){				
				images_all[i][j]=new image(dim);
			}
		}
	}
	
//
// Create the kernels
//
	public void createKernels(){
		kernels_all=new vector[num_of_conv_layers][];
		for(int i=0;i<num_of_conv_layers;i++){
			if(i==0){
				kernels_all[i]=new vector[num_of_images[i]];
			}else{
				kernels_all[i]=new vector[num_of_images[i]*num_of_images[i-1]];
			}			
			for(int j=0;j<kernels_all[i].length;j++){
				String line=reader.nextLine();
				while((line==null)||(line.equals(""))||(line.equals(" "))){
					line=reader.nextLine();
				}
				kernels_all[i][j]=new vector(line);
			}
		}
	}
	
//
// Create the biases
//
	public void createBiases(){
		bias_all=new double[num_of_conv_layers*2+1][];
		for(int i=0;i<num_of_conv_layers*2+1;i++){
			String line=reader.nextLine();
			while((line==null)||(line.equals(""))||(line.equals(" "))){
				line=reader.nextLine();
			}
			vector temp=new vector(line);
			bias_all[i]=temp.arrayForm();
		}
	}
	
//
// Create the multipliers
//
	public void createMultipliers(){
		dev_all=new double[num_of_conv_layers][];
		for(int i=0;i<num_of_conv_layers;i++){
			String line=reader.nextLine();
			while((line==null)||(line.equals(""))||(line.equals(" "))){
				line=reader.nextLine();
			}
			vector temp=new vector(line);
			dev_all[i]=temp.arrayForm();
		}
	}
	
//
// Create the weights for the classifier level
//
	public void createWeights(){
		weights=new double[num_of_images[num_of_images.length-1]*dim_label];
		for(int i=0;i<weights.length;i++){
			weights[i]=reader.nextDouble();
		}
	}	
}