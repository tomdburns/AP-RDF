#!/usr/bin/env python

"""
Generates AP-RDF Descriptors for a MOF

Version 0.2.2 - The Major update in this version is the addition
                of the charge weighted RDF - Using specific atom
                charges to weight the RDF calculation
"""

import os
import math
import time
import rdf_calc
import warnings
import itertools
import numpy as np
import pandas as pd
import pickle as pkl
from sys import argv
from config import Options
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")
LOC = '/'.join(os.path.realpath(__file__).split('/')[:-1])

try:
    ofile = argv[1].split('.cif')[0] + '.rdf'
    infile = argv[1]
    if '.cif' not in infile:
        infile += '.cif'
except:
    print "\nError: No cif file specified.\n"
    print "Submission line: $ ap-rdf structure.cif\n"
    exit()


OPTIONS = Options(input=ofile)

# ===============================================
# The RDF Code Options
# ===============================================
# RDF Calculation Parameters
# For details on these options see defaults.ini
# -----------------------------------------------
RERUN  = OPTIONS.get_bool('rerun')
AP     = OPTIONS.get_str('AP')
CUTOFF = OPTIONS.get_float('cutoff')
STEP   = OPTIONS.get_float('step')
MINR   = OPTIONS.get_float('minr')
SIGMA  = OPTIONS.get_float('sigma')
MINCV  = OPTIONS.get_bool('mincv')
ALT    = OPTIONS.get_bool('alt')
B      = OPTIONS.get_float('B')
F      = OPTIONS.get_float('F')
PLOT   = OPTIONS.get_bool('plot')
SMTH   = OPTIONS.get_float('smooth_sig')
RESO   = OPTIONS.get_int('reso')
NORM   = OPTIONS.get_bool('norm')
SMOOTH = OPTIONS.get_bool('smooth')
SPEC   = OPTIONS.get_bool('spec')
PAIRS  = OPTIONS.get_strlist('pair')
CMPLD  = OPTIONS.get_bool('compiled')
# ===============================================
# Other Stuff. Do Not Edit.
# -----------------------------------------------
VERSION = [0, 2, 2]
SUFFIX  = ''
DOPRP   = True
PROP    = {}
if 'None' in AP:
    DOPRP = False
BINS = int(math.ceil((CUTOFF - MINR) / STEP))
# ===============================================


def mkdir(name):
    try:
        os.mkdir(name)
    except OSError:
        pass


def import_properties():
    """Imports the atomic properties"""
    if not os.path.exists(LOC + '/Properties/' + AP + '.csv'):
        print "Property file for", AP, "Missing!"
        exit()
    data = pd.read_csv(LOC + '/Properties/' + AP + '.csv')
    props = {}
    for idx in data.index:
        props[data['atom'][idx]] = data['value'][idx]
    return props


def structural_data():
    """Extracts the atomic positions and bonding information from
    the cif
    """
    data = open(infile, 'r').readlines()
    head = data[:26]
    cell, atoms, bonds, shead = {}, False, False, False
    pos, bds = {}, {}
    fill = []
    for line in data:
        if '_cell' in line[:5]:
            line = line.split()
            cell[line[0]] = float(line[1].strip())
            continue
        if '_atom_type_partial_charge' in line:
            shead = True
            atoms = True
            continue
        if 'loop_' in line:
            if 'loop_\n' not in fill:
                fill.append(line)
            atoms = False
            continue
        if '_ccdc_geom_bond_type' in line:
            fill.append(line)
            bonds = True
            continue
        if atoms:
            nline   = line.split()
            if len(nline) == 0:
                continue
            try:
                tag    = nline[0]
                type   = nline[1]
                x_frac = float(nline[3])
                y_frac = float(nline[4])
                z_frac = float(nline[5])
                charge = float(nline[6])
            except IndexError:
                continue
            pos[tag] = {'Type'  : type,
                        'x_frac': x_frac,
                        'y_frac': y_frac,
                        'z_frac': z_frac,
                        'charge': charge,
                        'Line'  : line}
        elif bonds:
            nline = line
            line = line.split()
            if len(line) == 0:
                continue
            src, dst = line[0], line[1]
            if src not in bds:
                bds[src] = {}
            length = float(line[2])
            misc   = line[3]
            type   = line[4].strip()
            bds[src][dst] = {'Length': length,
                             'Type'  : type,
                             'Misc'  : misc,
                             'Line'  : nline}
        elif shead:
            if line not in fill:
                fill.append(line)
    return {'Cell': cell, 'Positions': pos, 'Bonds': bds, 'Head': head, 'Fill': fill}


def determine_ncells(cell):
    """Calculates the number of cells needed"""
    a = cell['_cell_length_a']
    b = cell['_cell_length_b']
    c = cell['_cell_length_c']
    lengths = [a, b, c]
    cnts = [0, 0, 0]
    for i, length in enumerate(lengths):
        done, val = False, length
        while not done:
            if val > CUTOFF:
                done = True
                break
            val += length
            cnts[i] += 1
    return cnts


def setup_carconv(cell):
    """Sets up the matrix to convert fractional to cartesian coords"""
    a_cell = cell['_cell_length_a']
    b_cell = cell['_cell_length_b']
    c_cell = cell['_cell_length_c']

    alpha  = cell['_cell_angle_alpha']
    alpha  = np.deg2rad(alpha)
    beta   = cell['_cell_angle_beta']
    beta   = np.deg2rad(beta)
    gamma  = cell['_cell_angle_gamma']
    gamma  = np.deg2rad(gamma)

    cosa   = np.cos(alpha)
    sina   = np.sin(alpha)
    cosb   = np.cos(beta)
    sinb   = np.sin(beta)
    cosg   = np.cos(gamma)
    sing   = np.sin(gamma)

    volume = 1.0 - cosa ** 2.0 - cosb ** 2.0 - cosg ** 2.0 + 2.0 * cosa * cosb * cosg
    volume = volume ** 0.5

    # Generate the Conversion Matrix
    r = np.zeros((3, 3))
    r[0, 0] = a_cell
    r[0, 1] = b_cell * cosg
    r[0, 2] = c_cell * cosb
    r[1, 1] = b_cell * sing
    r[1, 2] = c_cell * (cosa - cosb * cosg) / sing
    r[2, 2] = c_cell * volume / sing

    return r


def frac_to_cart(coord, r):
    """Converts a fractional coord to cartesian"""
    frac = np.zeros((3, 1))
    frac[0, 0] = coord[0]
    frac[1, 0] = coord[1]
    frac[2, 0] = coord[2]
    cart = np.matmul(r, frac)
    x_cart = cart[0, 0]
    y_cart = cart[1, 0]
    z_cart = cart[2, 0]
    return [x_cart, y_cart, z_cart]


def possible_coords(mof, cell, multi):
    """Generate possible coordinates"""
    r_f2c = setup_carconv(cell)
    fcoords = {}
    ccoords = {}
    ocoords = {}
    charges = []
    for nid, atom in enumerate(mof):
        fracs = (mof[atom]['x_frac'], mof[atom]['y_frac'], mof[atom]['z_frac'])
        charges.append(mof[atom]['charge'])
        new_fracs = []
        for i, frac in enumerate(fracs):
            val = multi[i]
            new = [frac]
            #if nid == 0:
            #    print(i, val, frac)
            for j in range(0, val + 1):
                d = j + 1
                new.append(frac + d)
                new.append(frac - d)
            new_fracs.append(new)
        #if nid == 0:
        #    for thing in new_fracs:
        #        print(len(thing), thing)
        refracs = []
        for i, frac in enumerate(fracs):
            new = [frac]
            for val in new_fracs[i]:
                new.append(val)
            refracs.append(new)
        temp = list(itertools.product(*refracs))
        fcoords[atom] = []
        for coord in temp:
            if coord in fcoords[atom]:
                continue
            fcoords[atom].append(coord)
        ccoords[atom] = []
        for fcoord in fcoords[atom]:
            ccoord = frac_to_cart(fcoord, r_f2c)
            ccoords[atom].append(ccoord)
            if fcoord == fracs:
                ocoords[atom] = ccoord
    return (ccoords, fcoords, ocoords, charges)


def rdf_component(r, R, a=1, b=1):
    """Gaussian Probability value for the combination at a given radius"""
    if not ALT:
        A = 1 / (SIGMA * (2*math.pi) ** 0.5)
        return a*b*A * np.exp(-1 * (((r - R)/SIGMA) ** 2))
    return a*b*F * np.exp(-1 * B * ((r - R) ** 2))


def build_bins():
    """sets up the bins for each rdf"""
    midl = (CUTOFF / BINS) / 2
    bins = {}
    val = midl
    while val < CUTOFF:
        bins[val] = 0
        val += (2 * midl)
    return bins


def build_bins_new():
    """sets up the bins for each rdf"""
    midl = (CUTOFF / BINS) / 2
    bins = []
    val = MINR + midl
    while val < CUTOFF:
        bins.append(val)
        val += (2 * midl)
    return np.array(bins), np.zeros(len(bins))


def distance(c1, c2):
    """Calcualtes distance"""
    return ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2 + (c1[2] - c2[2])**2) ** 0.5


def break_it_down(tags, vals):
    """Generates a breakdown for the components in the RDF"""
    breakdown = {}
    print "Sorting and Breaking Down Results"
    for id, tag in enumerate(tags):
        found = False
        if tag in breakdown:
            found = True
        elif tag.split('-')[1] + '-' + tag.split('-')[0] in breakdown:
            tag = tag.split('-')[1] + '-' + tag.split('-')[0]
            found = True
        if not found:
            breakdown[tag] = vals[id]
        else:
            for jd, r in enumerate(vals[id]):
                breakdown[tag][jd] += r
    print "Done."
    return breakdown


def calculate_rdf2(mof, atoms, coords, ocoords, charges):
    """Calculates the RDF"""
    #for iatom in tqdm(atoms):
    omit, eles, prop = [], [], []
    atm_id, iocoords, all_coords = [], [], []
    for aid, atom in enumerate(atoms):
        atm_id.append(aid)
        iocoords.append(ocoords[atom])
        iele = mof['Positions'][atom]['Type']
        all_coords.append(coords[atom])
        eles.append(iele)
        if DOPRP and 'Charge' not in AP:
            prop.append(PROP[iele])
        if iele not in PROP:
            prop.append(0)
    atm_id, iocoords = np.array(atm_id), np.array(iocoords)
    all_coords, prop = np.array(all_coords), np.array(prop)
    #combos = np.array(list(itertools.combinations_with_replacement(atoms, 2)))
    combos = np.array(list(itertools.combinations_with_replacement(atm_id, 2)))
    omit, eles = np.array(omit), np.array(eles)
    rbins, results = build_bins_new()
    print "Testing :", len(combos), "atom pairs"
    print "With    :", BINS, "Bins"
    print "Total   :", len(combos) * BINS, "Calculations\n"
    print "Min r   :", MINR
    print "Max r   :", CUTOFF
    print "Interval:", STEP, '\n'
    start = time.time()
    if 'Charge' in AP:
        prop = np.array(charges[:])
    if CMPLD:
        results, all, tags = rdf_calc.rdf_go(combos, results, rbins, iocoords, eles,
                                             all_coords, DOPRP, prop, SIGMA, F, B, ALT, CUTOFF)
    else:
        results, all, tags = rdf_go2(combos, results, rbins, iocoords, eles,
                                     all_coords, DOPRP, prop, SIGMA, F, B, ALT, CUTOFF)
    l_time = time.time() - start
    print "Calculation Completed in: %.2f" % l_time, "Seconds"
    print "Calculation Completed in: %.2f" % (l_time / 60), "Minutes"
    print '-' * 50
    breakdown = break_it_down(tags, all)
    return rbins, results, breakdown


def rdf_go2(combos, results, rbins, iocoords, eles,
           all_coords, doprp, prop, sigma, F, B, ALT, CUTOFF):
    """Compiled thing designed to work with numba"""
    tags, all = [], []
    for i, j in combos:
        icoord = iocoords[i]
        for jid, jcoord in enumerate(all_coords[j]):
            dcmp = (icoord - jcoord) ** 2
            dist = 0
            sub = []
            for d in dcmp:
                dist += d
            dist = dist ** 0.5
            if dist == 0 or dist > CUTOFF:
                continue
            else:
                for rid, rval in enumerate(rbins):
                    if doprp:
                        a, b = prop[i], prop[j]
                    else:
                        a, b = 1, 1
                    if not ALT:
                        A = 1 / (sigma * (2 * math.pi) ** 0.5)
                        rdf_bit = a * b * A * np.exp(-1 * (((rval - dist) / sigma) ** 2))
                    else:
                        rdf_bit = a * b * F * np.exp(-1 * B * ((rval - dist) ** 2))
                    results[rid] += rdf_bit
                    sub.append(rdf_bit)
            all.append(sub)
            tags.append(eles[i] + '-' + eles[j])
    return results, all, tags


def normalize_data(yvals, dx):
    """Normalizes the data so that the AUC is 1"""
    auc = 0.
    for val in yvals:
        auc += val * dx
    newy = []
    for val in yvals:
        newy.append(val / auc)
    return newy


def load_rdf(tag):
    """Loads previous rdf"""
    os.chdir('States')
    out = open(tag, 'rb')
    sts = pkl.load(out)
    out.close()
    os.chdir('..')
    return sts['rdfs'], sts['rbins'], sts['breakdown']


def dump_rdf(data, filetag):
    """Dumps the rdf data to prevent having to rerun"""
    mkdir('States')
    os.chdir('States')
    out = open(filetag, 'wb')
    pkl.dump(data, out)
    out.flush()
    out.close()
    os.chdir('..')


def get_dx(data):
    """gets the smallest dx"""
    rs = [r for r in data]
    dr = 999999.
    combos = list(itertools.combinations(rs, 2))
    for combo in combos:
        val = abs(combo[0] - combo[1])
        if val < dr:
            dr = val
    return dr


def write_output(tag, sums, radii, natoms):
    """writes an output file containing the results"""
    print "Writing csv file"
    out = open(tag + '.csv', 'w')
    out.write('Radius')
    for rad in radii:
        out.write(',%.3f' % rad)
    out.write('\n')
    out.write('RDF Value / Atom')
    for rad in radii:
        out.write(',%f' % (sums[rad] / natoms))
    out.flush()
    out.close()
    print "Done"


def main():
    """Main execution of script"""
    global PROP
    print "=" * 50
    print "Starting AP-RDF Code Version %i.%i.%i" % (VERSION[0], VERSION[1], VERSION[2]) + SUFFIX
    print "=" * 50 + '\n'
    mof = structural_data()
    if DOPRP and 'Charge' not in AP:
        PROP = import_properties()
    if MINCV:
        multi = [0, 0, 0]
    else:
        multi = determine_ncells(mof['Cell'])
    coords = possible_coords(mof['Positions'], mof['Cell'], multi)

    atoms = [atom for atom in mof['Positions']]
    natoms = len(atoms)
    nmof = argv[1].split('.cif')[0]
    print "Starting RDF Calculation for", nmof
    print "Number of Atom in Unit Cell:", len(atoms)
    sc, mult = [], 1
    for uc in multi:
        sc.append(3 + (2 * uc))
        mult *= (3 + (2 * uc))
    print "\nConsidering a %ix%ix%i Supercell" % (sc[0], sc[1], sc[2])
    print "Final Atom Count:", len(atoms) * mult
    print '-' * 50
    if '/' in nmof:
        nmof = nmof.split('/')[-1]
    if '_clean' in mof:
        nmof = nmof.split('_clean')[0]
    #tag = argv[1].split('.cif')[0] + '_' + AP
    tag = nmof + '_' + AP
    tail = ''
    if MINCV:
        tail += '-1'
    else:
        tail += '-0'
    if ALT:
        tail += '-1'
    else:
        tail +='-0'
    tail += '-%.1f-%i-%.1f-%.2f.pkl' % (MINR, BINS, CUTOFF, SIGMA)
    f_break = False
    tag += tail
    gen_file = 'States/%s_None%s' % (argv[1].split('.cif')[0], tail)
    #print "State:", 'States/' + tag
    #print "Gen  :", gen_file
    if os.path.exists('States//' + tag) and not RERUN:
        #rdfs, rbins, all, tags = load_rdf(tag)
        rdfs, rbins, breakdown = load_rdf(tag)
    elif os.path.exists(gen_file) and not RERUN and 'Charge' not in AP:
        f_break = True
        rdfs, rbins, breakdown = load_rdf('%s_None%s' % (nmof, tail))
        print "Previous General Calculation Found."
    else:
        rbins, rdfs, breakdown = calculate_rdf2(mof, atoms, coords[0], coords[2], coords[3])
        state = {'rdfs': rdfs, 'rbins': rbins, 'breakdown': breakdown}
        print "Calculations Completed. Storing Data."
        dump_rdf(state, tag)

    if PLOT or SPEC:
        plt.figure(figsize=(20, 10))
        plt.subplot(121)
        plt.title('AP-RDF Breakdown')
    sums = {}
    for jd, jtag in enumerate(breakdown):
        if f_break:
            a = PROP[jtag.split('-')[0]]
            b = PROP[jtag.split('-')[1]]
        else:
            a = 1
            b = 1
        ndis = a * b
        x, y = [], []
        for id, val in enumerate(rbins):
            x.append(val)
            y.append(ndis * breakdown[jtag][id])
            if val not in sums:
                sums[val] = 0
            sums[val] += ndis * breakdown[jtag][id]
        y = np.array(y) / natoms
        if PLOT or SPEC:
            if jd > 9:
                plt.plot(x, y, label=jtag, linestyle='--')
            else:
                plt.plot(x, y, label=jtag)
            plt.xlabel('Radius [Angstroms]')
            plt.ylabel('RDF Value / Atom')
    if PLOT or SPEC:
        plt.legend(loc='best')

    r_ix = sorted([val for val in sums])
    write_output(tag.split('.pkl')[0], sums, r_ix, len(atoms))

    if not PLOT and not SPEC:
        print('=' * 50)
        print("Program Terminated Normally.\n") 
        exit()

    xall, yall = [], []
    for ix in r_ix:
        #print ix, sums[ix]
        xall.append(ix)
        yall.append(sums[ix])
    xall = np.array(xall)
    yall = np.array(yall)
    plt.subplot(122)
    plt.title('AP-RDF For ' + nmof + " with Property: " + AP)
    #plt.plot(rbins, np.array(rdfs) / natoms)
    plt.plot(xall, yall / natoms)
    plt.xlabel('Radius [Angstroms]')
    plt.ylabel('RDF Value / Atom')
    plt.show()
    print '=' * 50
    print "Program Terminated Normally.\n"


if __name__ in '__main__':
    main()

