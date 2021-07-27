"""
Script to generate synthetic data and then reconstruct it  using various imaging artifacts
The data generated is to replicate X-ray macromolecular data
Usage >>>> python synth_data_generator.py -i OUTPUT_PATH_TO_RECON -m OUTPUT_PATH_TO_MASKS -n NUMBER_of_DATASETS -s RECON_SIZE -a TOTAL_PROJECTIONS_NUMBER
Authors:
Daniil Kazantsev
Gerard Jover Pujol
"""
import argparse
import logging

import numpy as np
from PIL import Image

from tomophantom import TomoP3D
from tomophantom.supp.flatsgen import synth_flats
from tomophantom.supp.artifacts import _Artifacts_
from tomophantom.TomoP3D import Objects3D

from tomobar.supp.suppTools import normaliser
from tomobar.methodsDIR import RecToolsDIR
from tomobar.methodsIR import RecToolsIR

import random

def normalise_im(im):
    return (im - im.min())/(im.max() - im.min())

def create_sample(dataset, N_size, total_angles, output_path_recon, output_path_gt):
    print ("Building 3D phantom using TomoPhantom software")

    el1 = {'Obj': Objects3D.ELLIPSOID,
          'C0' : 0.2,
          'x0' : 0.0,
          'y0' : 0.0,
          'z0' : 0.0,
          'a'  : 0.8,
          'b'  : 0.65,
          'c'  : 0.6,
          'phi1': 0.0}

    el2 = {'Obj': Objects3D.ELLIPSOID,
          'C0' : -0.2,
          'x0' : 0.0,
          'y0' : 0.0,
          'z0' : 0.0,
          'a'  : 0.7,
          'b'  : 0.5,
          'c'  : 0.5,
          'phi1': 0.0}

    myObjects=[el1,el2]
    GROUND_TRUTH = TomoP3D.Object(N_size, myObjects)

    midslice = int(0.5*N_size)
    GROUND_TRUTH[:,midslice+(int(0.1*midslice)):-1,:] = 0.0
    GROUND_TRUTH[:,0:midslice-(int(0.1*midslice)),:] = 0.0

    el3 = {'Obj': Objects3D.ELLIPSOID,
          'C0' : 0.3,
          'x0' : 0.0,
          'y0' : 0.0,
          'z0' : 0.0,
          'a'  : 0.9,
          'b'  : 0.7,
          'c'  : 0.65,
          'phi1': 0.0}

    myObjects2=[el3]
    GROUND_TRUTH2 = TomoP3D.Object(N_size, myObjects2)
    GROUND_TRUTH+=GROUND_TRUTH2

    C0_min = 0.025
    C0_max = 0.15
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

    el4 = {'Obj': Objects3D.CUBOID,
          'C0' : C_0,
          'x0' : x0_rand,
          'y0' : y0_rand,
          'z0' : z0_rand,
          'a'  : a_el3,
          'b'  :  b_el3,
          'c'  :  c_el3,
          'phi1': phi1}

    myObjects3=[el4]
    GROUND_TRUTH3 = TomoP3D.Object(N_size, myObjects3)
    GROUND_TRUTH+=GROUND_TRUTH3

    # Projection geometry related parameters:
    Horiz_det = int(np.sqrt(2)*N_size)
    Vert_det = N_size # detector row count (vertical) (no reason for it to be > N)
    angles = np.linspace(0.0,179.9,total_angles,dtype='float32') # in degrees
    angles_rad = angles*(np.pi/180.0)

    print ("Forward project the resulting 3D phantom")
    RectoolsDIR = RecToolsDIR(DetectorsDimH = Horiz_det,  # DetectorsDimH # detector dimension (horizontal)
                  DetectorsDimV = Vert_det,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                  CenterRotOffset = 0.0,   # Center of Rotation (CoR) scalar
                  AnglesVec = angles_rad, # array of angles in radians
                  ObjSize = N_size, # a scalar to define reconstructed object dimensions
                  device_projector='gpu')


    GROUND_TRUTH[GROUND_TRUTH == 0.5] = 0.05
    projData3D = RectoolsDIR.FORWPROJ(GROUND_TRUTH)

    GROUND_TRUTH[GROUND_TRUTH == 0.05] = 2
    GROUND_TRUTH[(GROUND_TRUTH > 0.2999) & (GROUND_TRUTH <= 0.3)] = 3
    GROUND_TRUTH[(GROUND_TRUTH > 0.3) & (GROUND_TRUTH < 0.5)] = 1


    print ("Simulate synthetic flat fields, add flat field background to the projections and add noise")
    _fresnel_propagator_ = {'fresnel_dist_observation' : 40,
                            'fresnel_scale_factor' : 10,
                            'fresnel_wavelenght' : 0.007}

    projection_data3D_fresnel = _Artifacts_(projData3D, **_fresnel_propagator_)

    flatsnum = 40 # the number of the flat fields required
    intensity_fluct=random.uniform(20000, 40000)
    source_fluctuation=random.uniform(0.01, 0.02)
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

    """
    RectoolsD = RecToolsDIR(DetectorsDimH = int(Horiz_det),  # DetectorsDimH # detector dimension (horizontal)
                        DetectorsDimV = int(N_size),  # DetectorsDimV # detector dimension (vertical) for 3D case only
                        CenterRotOffset = 0.0, # Center of Rotation (CoR) scalar (for 3D case only)
                        AnglesVec = angles_rad, # array of angles in radians
                        ObjSize = int(N_size), # a scalar to define reconstructed object dimensions
                        device_projector = 'gpu')

    Recon=RectoolsD.FBP(projData3D_norm) # FBP reconstruction
    """

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

    Recon=normalise_im(Recon)*65535
    print ("Saving the images")
    for n in range(Recon.shape[1]):
        # save image
        filename = output_path_recon + str(dataset) + "_recon_" + str(n).zfill(5) + ".tif"
        im = Recon[n,:,:].astype(np.uint16)
        im = Image.fromarray(im)
        im.save(filename)

    for n in range(GROUND_TRUTH.shape[1]):
        # save image
        filename = output_path_gt + str(dataset) + "_gt_" + str(n).zfill(5) + ".tif"
        im = GROUND_TRUTH[n,:,:]
        im = Image.fromarray(im.astype(np.uint8))
        im.save(filename)

def get_args():
    parser = argparse.ArgumentParser(description='Create the folder with the training data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--images', metavar='IMAGES', dest='dir_img', type=str, default='recon/',
                        help='Folder where the ouput images will be stored')
    parser.add_argument( '-m', '--masks', metavar='MASKS', dest='dir_mask', type=str, default='gt/',
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
    total_angles = int(args.angles)
    output_path_recon = args.dir_img
    output_path_gt = args.dir_mask

    for i in range(N_datasets):
        print("Creating dataset", dataset)
        create_sample(dataset, N_size, total_angles, output_path_recon, output_path_gt)
        dataset = str(i+1).zfill(5)
