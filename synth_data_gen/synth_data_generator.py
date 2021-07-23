import argparse
import logging
import os
import sys

import numpy as np
from PIL import Image
import timeit

#import tomophantom
from tomophantom import TomoP3D
from tomophantom.supp.qualitymetrics import QualityTools
from tomophantom.supp.flatsgen import synth_flats
from tomobar.supp.suppTools import normaliser
from tomobar.methodsDIR import RecToolsDIR
from tomobar.methodsIR import RecToolsIR
from tomophantom.supp.artifacts import _Artifacts_
import random
from tomophantom.TomoP3D import Objects3D
# python synth_generator.py -i RECON -m MASKS -n NO_DATASETS -s RECON_SIZE -a ANGLES_NUMBER

def normalise_im(im):
    return (im - im.min())/(im.max() - im.min())

def get_phantom_params(N_size):
    a_el1_min = 0.7
    a_el1_max = 0.95
    a_el1 = random.uniform(a_el1_min, a_el1_max)
    b_el1_min = 0.6
    b_el1_max = 0.75
    b_el1 = random.uniform(b_el1_min, b_el1_max)
    c_el1_min = 0.6
    c_el1_max = 0.85
    c_el1 = random.uniform(c_el1_min, c_el1_max)

    el1 = {'Obj': Objects3D.ELLIPSOID,
          'C0' : 0.7,
          'x0' : 0.0,
          'y0' : 0.0,
          'z0' : 0.0,
          'a'  : a_el1,
          'b'  : b_el1,
          'c'  : c_el1,
          'phi1': 0.0}


    a_el2_min = 0.6
    a_el2_max = a_el1
    a_el2 = random.uniform(a_el2_min, a_el2_max)
    b_el2_min = 0.6
    b_el2_max = b_el1
    b_el2 = random.uniform(b_el2_min, b_el2_max)
    c_el2_min = 0.6
    c_el2_max = c_el1
    c_el2 = random.uniform(c_el2_min, c_el2_max)

    el2 = {'Obj': Objects3D.ELLIPSOID,
          'C0' : -0.4,
          'x0' : 0.0,
          'y0' : 0.0,
          'z0' : 0.0,
          'a'  : a_el2,
          'b'  : b_el2,
          'c'  : c_el2,
          'phi1' : 0.0}

    C0_min = 0.01
    C0_max = 0.2
    C_0 = random.uniform(C0_min, C0_max)
    a_el3_min = 0.05
    a_el3_max = 0.5
    a_el3 = random.uniform(a_el3_min, a_el3_max)
    b_el3_min = 0.05
    b_el3_max = 0.5
    b_el3 = random.uniform(b_el3_min, b_el3_max)
    c_el3_min = 0.05
    c_el3_max = 0.5
    c_el3 = random.uniform(c_el3_min, c_el3_max)
    x0_rand = random.uniform(-0.1, 0.1)
    y0_rand = random.uniform(-0.1, 0.1)
    z0_rand = random.uniform(-0.1, 0.1)
    phi_min = 0.0
    phi_max = 180.0
    phi1 = random.uniform(phi_min, phi_max)

    el3 = {'Obj': Objects3D.CUBOID,
          'C0' : C_0,
          'x0' : x0_rand,
          'y0' : y0_rand,
          'z0' : z0_rand,
          'a'  : a_el3,
          'b'  :  b_el3,
          'c'  :  c_el3,
          'phi1': phi1}

    myObjects=[el1,el2,el3]
    return myObjects

def create_sample(dataset, N_size, output_path_recon, output_path_gt):
    print ("Building 3D phantom using TomoPhantom software")
    myObjects=get_phantom_params(N_size)

    PHANTOM3D = TomoP3D.Object(N_size, myObjects)
    PHANTOM3D[PHANTOM3D == 0.7] = 2
    PHANTOM3D[(PHANTOM3D > 0.2999) & (PHANTOM3D <= 0.3)] = 3
    PHANTOM3D[(PHANTOM3D > 0.3) & (PHANTOM3D < 0.7)] = 1

    # Projection geometry related parameters:
    Horiz_det = int(np.sqrt(2)*N_size)
    Vert_det = N_size # detector row count (vertical) (no reason for it to be > N)
    angles_num = int(0.5*np.pi*N_size); # angles number
    angles = np.linspace(0.0,179.9,angles_num,dtype='float32') # in degrees
    angles_rad = angles*(np.pi/180.0)

    print ("Building 3D analytical projection data with TomoPhantom")
    projData3D_analyt = TomoP3D.ObjectSino(N_size, Horiz_det, Vert_det, angles, myObjects)


    print ("Simulate synthetic flat fields, add flat field background to the projections and add noise")
    _fresnel_propagator_ = {'fresnel_dist_observation' : 40,
                            'fresnel_scale_factor' : 10,
                            'fresnel_wavelenght' : 0.007}

    projection_data3D_fresnel = _Artifacts_(projData3D_analyt, **_fresnel_propagator_)

    #%%
    I0  = 35000; # Source intensity
    flatsnum = 40 # the number of the flat fields required
    intensity_fluct=random.uniform(25000, 40000)
    source_fluctuation=random.uniform(0.1, 0.25)
    [projData3D_noisy, flatsSIM] = synth_flats(projection_data3D_fresnel,
                                               source_intensity = intensity_fluct, source_variation=source_fluctuation,\
                                               arguments_Bessel = (1,10,10,12),\
                                               specklesize = 15,\
                                               kbar = 0.3,
                                               jitter = 0.1,
                                               sigmasmooth = 3, flatsnum=flatsnum)

    print ("Normalise projections using ToMoBAR software")
    # normalise the data, the required format is [detectorsX, Projections, detectorsY]
    projData3D_norm = normaliser(projData3D_noisy, flatsSIM, darks=None, log='true', method='mean')

    print ("Reconstructing...")
    # set parameters and initiate a class object
    Rectools = RecToolsIR(DetectorsDimH = Horiz_det,     # Horizontal detector dimension
                        DetectorsDimV = N_size,        # Vertical detector dimension (3D case)
                        CenterRotOffset = None,          # Center of Rotation scalar or a vector
                        AnglesVec = angles_rad,          # A vector of projection angles in radians
                        ObjSize = N_size,                # Reconstructed object dimensions (scalar)
                        datafidelity='LS',               # Data fidelity, choose from LS, KL, PWLS
                        device_projector='gpu')

    _data_ = {'projection_norm_data' : projData3D_norm,
              'OS_number' : 6} # data dictionary

    lc = Rectools.powermethod(_data_) # calculate Lipschitz constant (run once to initialise)

    # Run FISTA reconstrucion algorithm without regularisation
    _algorithm_ = {'iterations' : 20,
                   'lipschitz_const' : lc}

    # adding regularisation using the CCPi regularisation toolkit
    _regularisation_ = {'method' : 'PD_TV',
                        'regul_param' :0.00001,
                        'iterations' : 80,
                        'device_regulariser': 'gpu'}

    # Run FISTA reconstrucion algorithm with 3D regularisation
    Recon = Rectools.FISTA(_data_, _algorithm_, _regularisation_)
    """
    RectoolsD = RecToolsDIR(DetectorsDimH = int(Horiz_det),  # DetectorsDimH # detector dimension (horizontal)
                        DetectorsDimV = int(N_size),  # DetectorsDimV # detector dimension (vertical) for 3D case only
                        CenterRotOffset = 0.0, # Center of Rotation (CoR) scalar (for 3D case only)
                        AnglesVec = angles_rad, # array of angles in radians
                        ObjSize = int(N_size), # a scalar to define reconstructed object dimensions
                        device_projector = 'gpu')

    Recon=RectoolsD.FBP(projData3D_norm) # FBP reconstruction
    """
    print ("Saving the images")
    for n in range(Recon.shape[1]):
        # save image
        filename = output_path_recon + str(dataset) + "_recon_" + str(n).zfill(5) + ".tif"
        im = Recon[n,:,:].astype(np.float64)
        im = normalise_im(im)
        im = Image.fromarray(im)
        im.save(filename)

    for n in range(PHANTOM3D.shape[1]):
        # save image
        filename = output_path_gt + str(dataset) + "_gt_" + str(n).zfill(5) + ".tif"
        im = PHANTOM3D[n,:,:]
        im = Image.fromarray(im.astype(np.uint8))
        im.save(filename)

def get_args():
    parser = argparse.ArgumentParser(description='Create the folder with the training data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--images', metavar='IMAGES', dest='dir_img',
                        help='Folder where the ouput images will be stored')
    parser.add_argument( '-m', '--masks', metavar='MASKS', dest='dir_mask',
                        help='Folder where the ouput masks will be stored')
    parser.add_argument('-n', '--n_datasets', dest='n_datasets',
                        help='Number of datasets to be generated')
    parser.add_argument('-s', '--size', dest='size', default=900,
                        help='Size of the data generated')
    parser.add_argument('-a', '--angles', dest='angles', default=800,
                        help='Number of angles')

    return parser.parse_args()



if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()

    logging.info(f'Simulating data:\n'
                 f'\tImage output directory : {args.dir_img}\n'
                 f'\tMask output directory : {args.dir_mask}\n'
                 f'\tNumber of datasets : {args.n_datasets}\n'
                 f'\tSize : {args.size}\n')

    dataset = "00000"
    N_datasets = int(args.n_datasets)
    N_size = int(args.size)
    output_path_recon = args.dir_img
    output_path_gt = args.dir_mask

    for i in range(N_datasets):
        print("Creating dataset", dataset)
        create_sample(dataset, N_size, output_path_recon, output_path_gt)
        dataset = str(i+1).zfill(5)
