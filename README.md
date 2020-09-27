# p2pSSE
p2pSSE is a program for assigning the secondary structure elements (SSEs) of a protein with a PDB file as its input. It consists of two sub-programs. The first one is a C++ program, p2p, that computes a pc-polyline for a protein structure and then extracts the geometrical features from the pc-polyline to be used for SSE assignment. The second one is a python script, p2pAssign.py, that assigns SSE using a CNN model trained with the same type of geometrical features.
Please see the file "Instruction4p2p.pdf" for details. 

The list of files:
(1) 2FR3FH.pdb: a PDB structure file used as an example. The protons in the structure has been added by REDUCE program.
(2) 2FR3FH_A_p2p.dat: the output of running p2p program on "2FR3FH.pdb",  it has the geometrical features extracted from the pc-polyline for the structure
(3) 2FR3FH_A_p2p.dat: the output of running p2pAssign.py on "2FR3FH_A_p2p.dat", it has the SSE assignments for each residue of the structure
(4) Instructione4p2p: a brief description of the p2pSSE program itself and the detailed instructions for the usages of the programs p2p and p2pAssign.py
(5) p2p       : the C++ program for the computation of a pc-polyline and the extraction of its geometrical features.
(6) p2pAssign.py:  the python script for SSE assignment using the geometrical features of a pc-polyline and a trained CNN model. It requires TensorFlow. 
(7) par_all27_prot_na.prm: a Charmm force field required by the p2p program
(8) top_all27_prot_na_correct.top: a Charmm force field required by the p2p program

The trained CNN model, best_asg0921_8A5z.h5,  is relative large ( ~61MB ) and could not be uploaded directly to Github so I upload it to Google Drive with the following link "https://drive.google.com/file/d/1etVXwmxp5FjMRIlS8NYnLLJ6dDeg7IGy/view?usp=sharing". The model was trained using Tensorflow1.4. But the python script itself ( p2pAssign.py ) that uses the model for assigning SSEs could be run using TensorFlow2.3. This script has been tested on both Ubuntu 18.04 and Ubuntu 20.04. 
