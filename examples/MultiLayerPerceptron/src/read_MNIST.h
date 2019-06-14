#ifndef READ_MNIST
#define READ_MNIST
 
#include <iostream>
#include <fstream>
#include <cstdint>
#include <string>
#include <endian.h>
#include <stdexcept>
#include <vector>

#define MNIST_ROWS 28
#define MNIST_COLS 28

using namespace std;

class MNISTEntry {
public:
	MNISTEntry(int size) {
		this->size = size;
		image = new uint8_t [size];
	}
	
	MNISTEntry(const MNISTEntry& rhs) : MNISTEntry(rhs.size){
		label=rhs.label;
		for(int i=0; i<size; ++i) {
			image[i] = rhs.image[i];
		}
	}

	~MNISTEntry(){
		delete [] image;
	}
	
	MNISTEntry& operator=(const MNISTEntry& rhs) {
		size=rhs.size;
		label=rhs.label;
		for(int i=0; i<size; ++i) {
			image[i] = rhs.image[i];
		}
		return *this;
	}
	
	void print() {
		for(int y=0; y<MNIST_COLS; ++y) {
			for(int x=0; x<MNIST_ROWS; ++x) {
				if(image[y*MNIST_ROWS+x]<128)
					cout<<".";
				else
					cout<<"#";
			}
			cout<<endl;
		}
	}
	
	uint32_t get_label() {
		return (uint32_t)label;
	}
	
	uint32_t size;
	uint8_t* image;
	uint8_t label;
};


class MNISTData {
public:
	MNISTData(const string data_path, const string label_path) {
		fstream data_file;
		fstream label_file;
		data_file.open(data_path, ios::binary | ios::in);
		if(data_file.fail())
			throw runtime_error("Failed to open file for data!");
		label_file.open(label_path, ios::binary | ios::in);
		if(label_file.fail())
			throw runtime_error("Failed to open file for labels!");
		
		data_file.seekg(4, ios::beg); //skip magic
		uint32_t rows;
		uint32_t cols;
		data_file.read((char*) &num_images, 4);
		data_file.read((char*) &rows, 4);
		data_file.read((char*) &cols, 4);
		// MNIST files saved in big-endian
		num_images=be32toh(num_images);
		rows=be32toh(rows);
		cols=be32toh(cols);
		int size=rows*cols;
        this->height = rows;
        this->width = cols;
		
		label_file.seekg(4, ios::beg); //skip magic
		uint32_t num_images_l;
		label_file.read((char*) &num_images_l, 4);
		num_images_l=be32toh(num_images_l);
		if(num_images != num_images_l) {
			cout<<num_images<<" "<<num_images_l<<endl;
			throw length_error("Number of items in MNIST data and label files do not match!");
		}
		
		for(int i=0; i<num_images; ++i) {
			data.push_back(MNISTEntry(size));
			data_file.read((char*) data[i].image, size);
			label_file.read((char*) &(data[i].label), 1);
		}
		
		data_file.close();
		label_file.close();
	}
	/*
	~MNISTData(){
		delete [] data;
	} */
	
	int get_num_images(){
		return num_images;
	}

	int get_height(){
		return height;
	}

	int get_width(){
		return width;
	}

    int get_num_classes() {return 10;}
	
	MNISTEntry* get_entry(int index) {
		return &data[index];
	} 

private:
	uint32_t num_images;
    uint32_t height, width;
	vector<MNISTEntry> data;
};
 
 #endif
