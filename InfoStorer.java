public class InfoStorer{
	// Image index
	public int image_index;
	// Offsets of a particular pixel in out[image_index]
	public int offset_out_x;
	public int offset_out_y;
	// Offsets of the upper left corner of each convolution window in in[i] corresponding to this particular pixel
	public int offset_in_x;
	public int offset_in_y;
	// Indices of kernel vectors associated with the convolution of this particular pixel
	public int[] kers;
}