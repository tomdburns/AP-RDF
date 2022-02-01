#!/usr/bin/env python

"""
The actual compiled loop which calculates the AP-RDF
"""

import sys
import math
import itertools
import numpy as np


def build_bins(cutoff, bins):
    """sets up the bins for each rdf"""
    midl = (cutoff / bins) / 2
    bins = {}
    val = midl
    while val < cutoff:
        bins[val] = 0
        val += midl
    return bins


def printProgress (iteration, total, prefix='Progress', suffix='Complete ',
                   decimals=2, barLength=50):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
    """
    filledLength = int(round(barLength * iteration / float(total)))
    percents = round(100.00 * (iteration / float(total)), decimals)
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('%s [%s] %.2f%s %s\r' % (prefix, bar, percents, '%', suffix)),
    sys.stdout.flush()


def distance(c1, c2):
    """Calcualtes distance"""
    return ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2 + (c1[2] - c2[2])**2) ** 0.5


def rdf_component(r, R, B, F, sigma, a=1, b=1, alt=False):
    """Gaussian Probability value for the combination at a given radius"""
    if not alt:
        A = 1 / (sigma * (2*math.pi) ** 0.5)
        return a*b*A * np.exp(-1 * (((r - R)/sigma) ** 2))
    return a*b*F * np.exp(-1 * B * ((r - R) ** 2))


def rdf_go(combos, results, rbins, iocoords, eles,
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


def calculate_rdf(mof, atoms, coords, rdfs, done, mincv, doprp,
                  prop, cutoff, bins, B, F, sigma, alt, bar=False,
                  freq=100):
    """Calculates the RDF"""
    count, nruns = 0, 0
    if mincv:
        for i in range(len(atoms)):
            nruns += i
    else:
        for i in range(len(atoms) + 1):
            nruns += i
    print("Number of Loops: %i\n\n" % nruns)
    for jid, iatom in enumerate(atoms):
        #if bar:
        #    printProgress(count, nruns)
        if mincv:
            jatoms = atoms[jid + 1:]
        else:
            jatoms = atoms[jid:]
        iele = mof['Positions'][iatom]['Type']
        if doprp and iele not in prop:
            continue
        for jatom in jatoms:
            if bar:
                if count % freq == 0:
                    printProgress(count, nruns)
                count += 1
            jele = mof['Positions'][jatom]['Type']
            if doprp and jele not in prop:
                continue
            if iatom + '-' + jatom in done:
                continue
            if jatom + '-' + iatom in done:
                continue
            done.append(iatom + '-' + jatom)
            tag, found = iele + '-' + jele, False
            if tag in rdfs:
                found = True
            if jele + '-' + iele in rdfs and not found:
                tag = jele + '-' + iele
            if not found:
                rdfs[tag] = build_bins(cutoff, bins)
            all_dists = []
            for icoord in coords[iatom]:
                for jcoord in coords[jatom]:
                    dist = distance(icoord, jcoord)
                    if dist == 0 or dist > cutoff:
                        continue
                    if mincv:
                        all_dists.append(dist)
                    else:
                        if doprp:
                            for rval in rdfs[tag]:
                                rdfs[tag][rval] += rdf_component(rval, dist, B, F, sigma,
                                                                 a=prop[iele], b=prop[jele], alt=alt)
                        else:
                            for rval in rdfs[tag]:
                                rdfs[tag][rval] += rdf_component(rval, dist, B, F,  sigma, alt=alt)
            if mincv and len(all_dists) > 0:
                for rval in rdfs[tag]:
                    if doprp:
                        rdfs[tag][rval] += rdf_component(rval, min(all_dists), B, F, sigma,
                                                         a=prop[iele], b=prop[jele], alt=alt)
                    else:
                        rdfs[tag][rval] += rdf_component(rval, min(all_dists), B, F, sigma, alt=alt)
    if bar:
        printProgress(nruns, nruns)
    print('\n\n')
    return rdfs


def calculate_rdf2(mof, atoms, coords, rdfs, done, mincv, doprp,
                  prop, cutoff, bins, B, F, sigma, alt, ocoords,
                  bar=False, freq=100):
    """Calculates the RDF"""
    combos = np.array(list(itertools.combinations_with_replacement(atoms, 2)))
    count, nruns = 0, len(combos)
    print("Number of Loops: %i\n\n" % nruns)
    for jid, combo in enumerate(combos):
        iatom = combo[0]
        jatom = combo[1]
        iele = mof['Positions'][iatom]['Type']
        if doprp and iele not in prop:
            continue
        if bar:
            if count % freq == 0:
                printProgress(count, nruns)
            count += 1
        jele = mof['Positions'][jatom]['Type']
        if doprp and jele not in prop:
            continue
        if iatom + '-' + jatom in done:
            continue
        if jatom + '-' + iatom in done:
            continue
        done.append(iatom + '-' + jatom)
        tag, found = iele + '-' + jele, False
        if tag in rdfs:
            found = True
        if jele + '-' + iele in rdfs and not found:
            tag = jele + '-' + iele
        if not found:
            rdfs[tag] = build_bins(cutoff, bins)
        all_dists = []
        #for icoord in coords[iatom]:
        icoord = ocoords[iatom]
        for jcoord in coords[jatom]:
            dist = distance(icoord, jcoord)
            if dist == 0 or dist > cutoff:
                continue
            if mincv:
                all_dists.append(dist)
            else:
                if doprp:
                    for rval in rdfs[tag]:
                        rdfs[tag][rval] += rdf_component(rval, dist, B, F, sigma,
                                                         a=prop[iele], b=prop[jele], alt=alt)
                else:
                    for rval in rdfs[tag]:
                        rdfs[tag][rval] += rdf_component(rval, dist, B, F,  sigma, alt=alt)
        if mincv and len(all_dists) > 0:
            for rval in rdfs[tag]:
                if doprp:
                    rdfs[tag][rval] += rdf_component(rval, min(all_dists), B, F, sigma,
                                                     a=prop[iele], b=prop[jele], alt=alt)
                else:
                    rdfs[tag][rval] += rdf_component(rval, min(all_dists), B, F, sigma, alt=alt)
    if bar:
        printProgress(nruns, nruns)
    print('\n\n')
    return rdfs

