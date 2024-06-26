#!/usr/bin/env python
"""
Calculating thermodynamic perperties using Auto3D output
"""
import sys
import os
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)

import torch
import numpy as np

from tqdm import tqdm # It is for progress bar
import ase
from ase import Atoms
from ase.optimize import BFGS
# from ase.vibrations import Vibrations
from ase.vibrations import VibrationsData, Vibrations
from ase.thermochemistry import IdealGasThermo
import ase.calculators.calculator
from functools import partial
from typing import Optional
import torchani
from Auto3D.batch_opt.batchopt import EnForce_ANI
from Auto3D.batch_opt.ANI2xt_no_rep import ANI2xt
from Auto3D.utils import hartree2ev

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
ev2hatree = 1/hartree2ev  

class Calculator(ase.calculators.calculator.Calculator):
    """ASE calculator interface for AIMNET and ANI2xt"""
    implemented_properties = ['energy', 'forces']
    def __init__(self, model, charge=0):
        super().__init__()
        self.model = model 
        for p in self.model.parameters():
            p.requires_grad_(False)
        a_parameter = next(self.model.parameters())
        self.device = a_parameter.device
        self.dtype = a_parameter.dtype
        self.charge = torch.tensor([charge], dtype=torch.float, device=self.device)
        self.species = {'H':1, 'C':6, 'N':7, 'O':8, 'F':9, 'Si':14, 'P':15,
                        'S':16, 'Cl':17, 'As':33, 'Se':34, 'Br':35, 'I':53,
                        'B':5}

    def set_charge(self, charge:int):
        self.charge = torch.tensor([charge], dtype=torch.float, device=self.device)

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=ase.calculators.calculator.all_changes):
        super().calculate(atoms, properties, system_changes)

        species = torch.tensor([self.species[symbol] for symbol in self.atoms.get_chemical_symbols()],
                               dtype=torch.long, device=self.device)
        coordinates = torch.tensor(self.atoms.get_positions()).to(self.device).to(self.dtype)
        coordinates = coordinates.requires_grad_(True)

        species = species.unsqueeze(0)
        coordinates = coordinates.unsqueeze(0)
        
        energy, forces = self.model(coordinates, species, self.charge)
        self.results['energy'] = energy.item()
        self.results['forces'] = forces.squeeze(0).to('cpu').numpy()

def model_name2model_calculator(model_name: str, device=torch.device('cpu'), charge=0):
    """Return a model and the ASE calculator object for a molecule"""
    if model_name == "ANI2xt":
        model = EnForce_ANI(ANI2xt(device, periodic_table_index=True), model_name).double()
        calculator = Calculator(model, charge)
    elif model_name == "AIMNET":
        # Using the ensemble AIMNet2 model for computing energy and forces
        aimnet = torch.jit.load(os.path.join(root, "models/aimnet2_wb97m_ens_f.jpt"), map_location=device)
        model = EnForce_ANI(aimnet, model_name)
        calculator = Calculator(model, charge)
    elif model_name == "ANI2x":
        ani2x = torchani.models.ANI2x(periodic_table_index=True).to(device).double()
        model = EnForce_ANI(ani2x, model_name)
        calculator = ani2x.ase()
    else:
        raise ValueError("model has to be 'ANI2x', 'ANI2xt' or 'AIMNET'")
    return model, calculator

def aimnet_hessian_helper(coord:torch.tensor, 
                          numbers:Optional[torch.Tensor]=None,
                          charge: Optional[torch.Tensor]=None,
                          model: Optional[torch.nn.Module]=None,
                          model_name='AIMNET'):
    '''coord shape: (1, num_atoms, 3)
    numbers shape: (1, num_atoms)
    charge shape: (1,)'''
    if model_name == 'AIMNET':
        dct = dict(coord=coord, numbers=numbers, charge=charge)
        return model(dct)['energy']  # energy unit: eV
    elif model_name == 'ANI2xt':
        device = coord.device
        periodict2idx = {1:0, 6:1, 7:2, 8:3, 9:4, 16:5, 17:6}
        numbers2 = torch.tensor([periodict2idx[num.item()] for num in numbers.squeeze()], device=device).unsqueeze(0)
        e = model(numbers2, coord)
        return e  # energy unit: eV
    elif model_name == 'ANI2x':
        e = model((numbers, coord)).energies * hartree2ev
        return e  # energy unit: eV

def xyz2aimnet_input(species, coord, charge, device=torch.device('cpu'), model_name='AIMNET') -> dict:
    """Converts sdf to aimnet input, assuming the sdf has only 1 conformer."""
    coord_tensor = torch.tensor(coord, device=device).unsqueeze(0)
    numbers_tensor = torch.tensor([Atoms(a).get_atomic_numbers()[0] for a in species],
                            device=device).unsqueeze(0)
    charge_tensor = torch.tensor([charge], device=device, dtype=torch.float)
    return dict(coord=coord_tensor, numbers=numbers_tensor, charge=charge_tensor)

def vib_hessian_xyz(species, coord, charge, ase_calculator, model,
                    device=torch.device('cpu'), model_name='AIMNET'):
    '''return a VibrationsData object
    model: ANI2xt or AIMNet2 or ANI2x that can be used to calculate Hessian'''
    # get the ASE atoms object
    atoms = Atoms(species, coord)
    atoms.set_calculator(ase_calculator)

    # get the Hessian
    aimnet_input = xyz2aimnet_input(species, coord, charge, device, model_name=model_name)
    coord = aimnet_input['coord'].requires_grad_(True)
    num_atoms = coord.shape[1]
    numbers = aimnet_input['numbers']
    charge = aimnet_input['charge']

    hess_helper = partial(aimnet_hessian_helper,
                          numbers=numbers,
                          charge=charge,
                          model=model,
                          model_name=model_name)
    hess = torch.autograd.functional.hessian(hess_helper,
                                             coord)
    hess = hess.detach().cpu().view(num_atoms, 3, num_atoms, 3).numpy()  

    # get the VibrationsData object
    vib = VibrationsData(atoms, hess)
    return vib

# This function relies on using gitlab sourced ASE implementation
# conda version does not work due to lack of ignore_imag_modes option
def do_mol_thermo_xyz(species, coord, charge,
                      atoms: ase.Atoms,
                      model: torch.nn.Module,
                      device=torch.device('cpu'),
                      T=298.0, model_name='AIMNET'):
    """For a RDKit mol object, calculate its thermochemistry properties.
    model: ANI2xt or AIMNet2 or ANI2x that can be used to calculate Hessian"""
    vib = vib_hessian_xyz(species, coord, charge, atoms.get_calculator(), model, device, model_name=model_name)
    vib_e = vib.get_energies()
    e = atoms.get_potential_energy()
    thermo = IdealGasThermo(vib_energies=vib_e,
                            potentialenergy=e,
                            atoms=atoms,
                            geometry='nonlinear',
                            symmetrynumber=1, spin=0, natoms=None,
                            ignore_imag_modes=True) # ignore_imag_modes avoid optimization failures due to imaginary frequencies being supplied
    H = thermo.get_enthalpy(temperature=T) * ev2hatree
    S = thermo.get_entropy(temperature=T, pressure=101325) * ev2hatree
    G = thermo.get_gibbs_energy(temperature=T, pressure=101325) * ev2hatree
    
    result = {}
    result["H_hartree"] = str(H)
    result["S_hartree"] = str(S)
    result["T_K"]       = str(T)
    result["G_hartree"] = str(G)
    result["E_hartree"] = str(e * ev2hatree)
    
    #Updating ASE atoms coordinates
    new_coord = atoms.get_positions()
    return (new_coord, result)

def parse_xyz(xyzname):

    """
    Parse the coordinates from an xyz file.
    """

    f = open(xyzname, 'r')
    text = f.read().split('\n')
    natoms = int(text[0])
    text = text[2:2+natoms]
    atoms, coords_all = [], []
    for line in text:
        array = line.split()
        assert len(array) == 4
        atom = array[0]
        coords = np.array([float(array[i]) for i in range(1,4)])
        atoms.append(atom)
        coords_all.append(coords)
    coords_all = np.array(coords_all)

    return atoms, coords_all

def parse_xyz_list(xyzname):
    # The input file should be a list of xyz structured molecules, for example crest_clustered.xyz
    f = open(xyzname, 'r')
    text = f.read().split('\n')
    natoms = int(text[0])
    xyz_list = []

    total_length = int(len(text))
    section_length = int(natoms + 2)
    for i in range(total_length // section_length):
        begin_line = 2 + i * section_length
        xyz_section = text[begin_line:begin_line+natoms]
        atoms, coords_all = [], []
        for line in xyz_section:
            array = line.split()
            assert len(array) == 4
            atom = array[0]
            coords = np.array([float(array[i]) for i in range(1,4)])
            atoms.append(atom)
            coords_all.append(coords)
        coords_all = np.array(coords_all)

        xyz_list.append((atoms, coords_all))

    return xyz_list

def print_xyz(molecule_data, filename):
    (species, coord, result) = molecule_data
    molecule_length = len(coord)
    free_energy = result['G_hartree']

    f = open(filename, "w")
    f.write(str(molecule_length) + '\n')
    f.write(str(free_energy) + '\n') 
    for j in range(len(species)):
        # astype converts the result in numpy.fp64 to string
        output_line_str = species[j] + '\t' + str(coord[j][0]) + '\t' + str(coord[j][1]) +'\t' + str(coord[j][2]) + '\n'
        f.write(output_line_str)
    
    f.close()

def calc_thermo_scl(species_coords_list_list, model_name: str, temperature=298.15, gpu_idx=0, opt_tol=0.0002, opt_steps=5000):
    """
    ASE interface for calculation thermo properties using ANI2x, ANI2xt or AIMNET.

    :param species_coords_list_list: A list of list for input structures, each structure fed from a parser in (species, coords, charge) format
    :type species_coords_list_list: list list
    :param path: Input sdf file
    :type path: str
    :param model_name: ANI2x, ANI2xt or AIMNET
    :type model_name: str
    :param get_mol_idx_t: A function that returns (idx, T) from a pybel mol object, by default using the 298 K temperature, defaults to None
    :type get_mol_idx_t: function, optional
    :param gpu_idx: GPU cuda index, defaults to 0
    :type gpu_idx: int, optional
    :param opt_tol: Convergence_threshold for geometry optimization, defaults to 0.0002
    :type opt_tol: float, optional
    :param opt_steps: Maximum geometry optimization steps, defaults to 5000
    :type opt_steps: int, optional
    """
    #Prepare output name
    out_mols_list, mols_failed = [], []

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_idx}")
    else:
        device = torch.device("cpu")
        print("Warning: Running on CPU")

    if model_name == 'AIMNET':
        aimnet0_path = os.path.join(root, "models/aimnet2_wb97m-d3_0.jpt")
        hessian_model = torch.jit.load(aimnet0_path, map_location=device)
    elif model_name == 'ANI2xt':
        hessian_model = ANI2xt(device).double()
    elif model_name == 'ANI2x':
        hessian_model = torchani.models.ANI2x(periodic_table_index=True).to(device).double()
    model, calculator = model_name2model_calculator(model_name, device)

    for species_coords_list in species_coords_list_list:
        out_mols = []
        total_molecule_count = len(species_coords_list)
        for i in tqdm(range(total_molecule_count)):
            (species, coord, charge) = species_coords_list[i]
            atoms = Atoms(species, coord)

            if model_name == 'AIMNET':
                calculator.set_charge(charge)
            atoms.set_calculator(calculator)        

            idx = i
            T = temperature
            print(idx)

            try:
                try:
                    enForce_in = xyz2aimnet_input(species, coord, charge, device, model_name=model_name)
                    _, f_ = model(enForce_in['coord'].requires_grad_(True),
                                    enForce_in['numbers'],
                                    enForce_in['charge'])
                    fmax = f_.norm(dim=-1).max(dim=-1)[0].item()
                    assert fmax <= 3e-3
                    (new_coord, result) = do_mol_thermo_xyz(species, coord, charge, atoms, hessian_model, device, T, model_name=model_name)
                    out_mols.append((species, new_coord, result))
                except AssertionError:
                    print('optiimize the input geometry')
                    opt = BFGS(atoms)
                    opt.run(fmax=3e-3, steps=opt_steps)
                    (new_coord, result) = do_mol_thermo_xyz(species, coord, charge, atoms, hessian_model, device, T, model_name=model_name)
                    out_mols.append((species, new_coord, result))
            except ValueError:
                print('use tighter convergence threshold for geometry optimization')
                opt = BFGS(atoms)
                opt.run(fmax=opt_tol, steps=opt_steps)
                (new_coord, result) = do_mol_thermo_xyz(species, coord, charge, atoms, hessian_model, device, T, model_name=model_name)
                out_mols.append((species, new_coord, result))
        
        print("Number of successful thermo calculations: ", len(out_mols), flush=True)
        out_mols_list.append(out_mols)


    return out_mols_list

def calc_thermo_filelist(input_file_list, model_name: str, output_dir: str, temperature=298.15, gpu_idx=0, opt_tol=0.0002, opt_steps=5000):
    # input_file_list should supply a list of xyz files, one xyz structure in each file.
    input_data_list = []
    for input_file in input_file_list:
        (species, coord) = parse_xyz(input_file)
        charge = 0
        input_data_list.append([(species, coord, charge)])

    # Do the computation
    out_mols = calc_thermo_scl(input_data_list, model_name, temperature, gpu_idx, opt_tol, opt_steps)

    # Check if the directory exists. If it does not exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Output to result xyz files
    for i in range(len(out_mols)):
        out_file_path = os.path.join(output_dir, "output_" + str(i) + ".xyz")
        print_xyz(out_mols[i], out_file_path)

def calc_thermo_clustered(input_file: str, model_name: str, output_dir: str, output_file_basename: str, temperature=298.15, gpu_idx=0, opt_tol=0.0002, opt_steps=5000):
    # input file should supply a concartenation of xyz files, like crest_clustered.xyz
    input_data_list_tmp = parse_xyz_list(input_file)
    input_data_list = []
    for (species, coord) in input_data_list_tmp:
        charge = 0
        input_data_list.append((species, coord, charge))

    # Do the computation
    out_mols_list = calc_thermo_scl([input_data_list], model_name, temperature, gpu_idx, opt_tol, opt_steps)

    # Check if the directory exists. If it does not exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Output to result xyz files
    out_mols = out_mols_list[0]
    for i in range(len(out_mols)):
        out_file_path = os.path.join(output_dir, output_file_basename + "_" + str(i) + ".xyz")
        print_xyz(out_mols[i], out_file_path)

def calc_thermo_crest_clustered_in_directory(input_base_directory: str, model_name: str, output_dir: str, temperature=298.15, gpu_idx=0, opt_tol=0.0002, opt_steps=5000):
    # input_base_directory is the directory, such that it contains "name_of_structure"/crest_clustered.xyz
    clustered_file_name = 'crest_clustered.xyz'
    input_dir_list = []
    input_data_list_list = []
    for this_dir in os.scandir(input_base_directory):
        if os.path.isdir(this_dir):
            dir_name = this_dir.name
            print('Adding: ' + dir_name + ' to the run list')
            input_file = os.path.join(input_base_directory, dir_name, clustered_file_name)
            input_data_list_tmp = parse_xyz_list(input_file)
            input_data_list = []

            for (species, coord) in input_data_list_tmp:
                charge = 0
                input_data_list.append((species, coord, charge))

            input_dir_list.append(dir_name)
            input_data_list_list.append(input_data_list)

    # Do the computation
    out_mols_list = calc_thermo_scl(input_data_list_list, model_name, temperature, gpu_idx, opt_tol, opt_steps)

    # Output to result xyz files
    for (output_file_basename, out_mols) in zip(input_dir_list, out_mols_list):
        # Check if the directory exists. If it does not exist, create it
        if not os.path.exists(os.path.join(output_dir, output_file_basename)):
            os.makedirs(os.path.join(output_dir, output_file_basename))
        for i in range(len(out_mols)):
            out_file_path = os.path.join(output_dir, output_file_basename, output_file_basename + "_" + str(i) + ".xyz")
            print_xyz(out_mols[i], out_file_path)

if __name__ == "__main__":
    # path = '/home/jack/run_auto3d/20231030-101405-214461_methane/imaginary/methane_out.sdf'
    # path = '/home/jack/Auto3D_pkg/tests/files/cyclooctane.sdf'
    # out = calc_thermo(path, 'AIMNET', gpu_idx=1)
    # out = calc_thermo(path, 'ANI2xt', gpu_idx=1)
    # out = calc_thermo(path, 'ANI2x', gpu_idx=1)
    print("Lorem ipsum")
