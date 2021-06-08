#!/usr/bin/env python
# Copyright (C) 2012 Ion Torrent Systems, Inc. All Rights Reserved

import os
from ion.utils.blockprocessing import printtime
import traceback
import subprocess
import json
from Bio import SeqIO
import commands
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def parse_fasta(file):
    amplicon = {}
    fasta_sequences = SeqIO.parse(open(file), "fasta")
    for r_fasta in fasta_sequences:
        r_name, r_sequence = r_fasta.id, r_fasta.seq.tostring()
        amplicon[r_name] = r_sequence
    return amplicon


def match_controls(amplicon):
    map_amp = {}
    for ampl, seq in amplicon.items():
        if ampl.find("ASC") >= 0:
            for ampl1, seq1 in amplicon.items():
                if ampl1 != ampl and len(seq1) == len(seq):
                    map_amp[ampl] = ampl1
    return map_amp


def merge_inlinecontrol_json(dirs, BASECALLER_RESULTS):
    printtime("Merging inline_control_stats.json files")
    print(dirs)
    try:
        inlinecontrolfiles = []
        for subdir in dirs:
            subdir = os.path.join(BASECALLER_RESULTS, subdir)
            printtime("DEBUG: %s:" % subdir)
            inlinecontroljson = os.path.join(subdir, "inline_control_stats.json")
            if os.path.exists(inlinecontroljson):
                inlinecontrolfiles.append(subdir)
            else:
                printtime(
                    "Warning: Merging inline_control_stats.json files: skipped %s"
                    % inlinecontroljson
                )

        merge(inlinecontrolfiles, BASECALLER_RESULTS)
    except Exception:
        traceback.print_exc()
        printtime("Merging inline_control_stats.json files failed")

    printtime("Finished merging inline control stats")


def merge(block_dirs, results_dir):
    """inlinecontrol.merge - Combine inline_control_stats.json metrics from multiple blocks"""
    combined_json = merge_counts(block_dirs)

    file = open(os.path.join(results_dir, "inline_control_stats.json"), "w")
    file.write(json.dumps(combined_json, indent=4))
    file.close()

    # Determine the maximum on ratio plots
    threshold_ratio = 5.0
    scale = 5.0
    for bam, bam_dict in combined_json.iteritems():
        try: 
            ratios = bam_dict["ratio"]
            for amplicon, ratio in ratios.iteritems():
                if "NA" not in ratio and float(ratio) >= 5.0 and float(ratio) > scale:
                    splits = amplicon.split("/")
                    spikeIn = splits[1]
                    if int(bam_dict["counts"][spikeIn]) >= 100:
                        scale = float(ratio)
        except:
            traceback.print_exc()
            printtime("Cannot find maximum inline control ratio " + bam)
            pass
    printtime("The maximum ratio for this run is: " + str(scale))

    # plot inline control ratios
    for bam, bam_dict in combined_json.iteritems():
        try:
            plot_ilc_ratios(bam_dict, results_dir, str(bam).replace(".basecaller.bam", "") + '.inline_control.png', scale)
        except Exception:
            traceback.print_exc()
            printtime("Plotting inline control ratio failed " + bam)
            pass

def merge_counts(block_dirs):
    ic = {"counts": {}, "ratio": {}}
    ic_perbam = {}
    mapping = {}
    count = 0
    for dir in block_dirs:
        try:
            file = open(os.path.join(dir, "inline_control_stats.json"), "r")
            block_json = json.load(file)
            # initialize merge elements
            if count == 0:
                c = iter(block_json.values()).next()["counts"]
                r = iter(block_json.values()).next()["ratio"]
                for item, temp in c.items():
                    ic["counts"][item] = "0"
                for item, temp in r.items():
                    ic["ratio"][item] = "0"
                    for item1, temp1 in c.items():
                        if item.endswith(item1):
                            splits = item.split("/")
                            mapping[item1] = splits[0]

            # adding the current block
            for bam, bam_dict in block_json.items():
                if count == 0 or bam not in ic_perbam.keys():
                    ic_perbam[bam] = {}
                    ic_perbam[bam]["ratio"] = {}
                    ic_perbam[bam]["counts"] = {}
                    if count > 0: # new bam file for the block
                        for item, temp in bam_dict["counts"].items():
                            ic_perbam[bam]["counts"][item] = bam_dict["counts"][item]

                for item, temp in bam_dict["counts"].items():
                    if (
                        "NA" not in ic["counts"][item]
                        and "NA" not in bam_dict["counts"][item]
                    ):
                        ic["counts"][item] = str(
                            int(ic["counts"][item]) + int(bam_dict["counts"][item])
                        )
                    else:
                        ic["counts"][item] = "NA"

                    if count == 0:
                        ic_perbam[bam]["counts"][item] = bam_dict["counts"][item]
                    else:
                        ic_perbam[bam]["counts"][item] = str(
                            int(bam_dict["counts"][item])
                            + int(ic_perbam[bam]["counts"][item])
                        )

            file.close()
            count = count + 1
        except Exception as e:
            printtime("merge_inlinecontrol_json.merge_filtering: skipping block " + dir + str(e))

    # ratios
    for bam, bam_dict in ic_perbam.items():
        for item, item_dict in mapping.items():
            if bam in ic_perbam.keys() and int(ic_perbam[bam]["counts"][item]) > 0:
                ic_perbam[bam]["ratio"][item_dict + "/" + item] = str(
                    float(ic_perbam[bam]["counts"][item_dict])
                    / float(ic_perbam[bam]["counts"][item])
                )
            else:
                ic_perbam[bam]["ratio"][item_dict + "/" + item] = "NA"

    for item, item_dict in mapping.items():
        if int(ic["counts"][item]) > 0:
            ic["ratio"][item_dict + "/" + item] = float(
                ic["counts"][item_dict]
            ) / float(ic["counts"][item])
        else:
            ic["ratio"][item_dict + "/" + item] = "NA"
    return ic_perbam


def inline_control(bcDir, ctrlRef, outDir):
    if not os.path.exists(bcDir):
        printtime("Warnings: Cannot find basecaller directory: " + bcDir)
        exit(1)

    if not os.path.exists(ctrlRef):
        printtime("Error: Cannot find control fasta file: " + ctrlRef)
        exit(1)
    printtime("Inline Control reference file : " + ctrlRef)
    if not os.path.exists(outDir):
        printtime("Warnings: Cannot find output directory: " + outDir)
        os.system("mkdir -p " + outDir)

    try:

        bamfiles = []
        for bam in os.listdir(bcDir):
            if bam.endswith("rawlib.basecaller.bam"):
                bamfile = os.path.join(bcDir, bam)
                if os.path.exists(bamfile):
                    bamfiles.append(bamfile)
                else:
                    print("Warnings: Cannot find bamfile " + bamfile)
        print(bamfiles)
    except Exception:
        printtime("Error: Cannot read bam files from directory: " + bcDir)
        exit(1)

    try:
        # parse control amplicon
        ref_ctrl = parse_fasta(ctrlRef)
        map_ctrl = match_controls(ref_ctrl)

        del_target_file = os.path.join(outDir, "del_target.tmp")
        if os.path.exists(del_target_file):
            os.remove(del_target_file)

        # with open(del_target_file, "w") as tmpfile:
        #     for target, control in map_ctrl.items():
        #         tmpfile.write(target + "\n")

                # ref_hk = parse_fasta(hkRef)

        stats_bam = {}
    except Exception:
        printtime("Error: Cannot parse control sequences")
        exit(1)
    try:
        stats_bam = {}

        for bam in bamfiles:
            printtime("Aligning bam file: " + bam)
            D, bamfile = os.path.split(bam)
            bamCtrl_output = os.path.join(
                outDir, os.path.splitext(bamfile)[0] + ".control.bam"
            )

            # control reads
            cmd = (
                "tmap mapall -n 12 -f %s  -r %s -i bam -v -Y stage1 map4 \
                | samtools view -h -Sb -q 10 -F 4 -o %s - 2>> /dev/null"
                % (ctrlRef, bam, bamCtrl_output)
            )
            subprocess.call(cmd, shell=True)
            bamCtrl_sorted = os.path.join(
                outDir, os.path.splitext(bamfile)[0] + ".control.sorted.bam"
            )
            cmd = "samtools sort -o {out_bam} {in_bam}".format(
                in_bam=bamCtrl_output, out_bam=bamCtrl_sorted
            )
            subprocess.call(cmd, shell=True)
            cmd = "samtools index %s" % bamCtrl_sorted
            subprocess.call(cmd, shell=True)

            printtime("Counting reads bam file: " + bam)
            stats_bam[bamfile] = {}
            stats_bam[bamfile]["counts"] = {}
            for ref in ref_ctrl:
                cmd = "samtools view -q 10 {sort_bam} | grep {ref_seq} | wc -l".format(
                    sort_bam=bamCtrl_sorted, ref_seq=ref
                )
                (status, count) = commands.getstatusoutput(cmd)
                if status == 0:
                    stats_bam[bamfile]["counts"][ref] = count

            # Ratios
            stats_bam[bamfile]["ratio"] = {}
            for spike, control in map_ctrl.items():
                if int(stats_bam[bamfile]["counts"][spike]) > 0:
                    stats_bam[bamfile]["ratio"][control + "/" + spike] = str(
                        float(stats_bam[bamfile]["counts"][control])
                        / float(stats_bam[bamfile]["counts"][spike])
                    )
                else:
                    stats_bam[bamfile]["ratio"][control + "/" + spike] = "NA"

            # Remove control reads from the original unmapped bam
            try:
                cmd = "samtools view %s | cut -f 1 > %s" %(bamCtrl_sorted, del_target_file)
                subprocess.call(cmd, shell=True)
                cmd = "samtools view -h %s | grep -vf %s | samtools view -h -bS -o %s -" % (
                    bam,
                    del_target_file,
                    os.path.join(outDir, os.path.splitext(bamfile)[0] + ".filtered.bam"),
                )
                subprocess.call(cmd, shell=True)
            except:
                print "Error in removing reads from bam"
                pass

        if os.path.exists(del_target_file):
            os.remove(del_target_file)

        with open(os.path.join(outDir, "inline_control_stats.json"), "w") as fp:
            json.dump(stats_bam, fp, indent=4)
    except Exception:
        printtime("Error: Alignment failed in inline control")
        exit(1)


def plot_ilc_ratios(combined, outDir, plotname, scale):
    plotBarChart(combined, os.path.join(outDir, plotname), scale)

def parsePairs(dict_ratio):
    mapping = {}
    for item, temp in dict_ratio.iteritems():
        splits = item.split("/")
        mapping[splits[0]] = splits[1]
    return mapping

def plotBarChart(data, fname, scale):
    mapping = {"chr5:112216009-112216224/ASC_Siz10:687-902": "266", "chr7:75618964-75618995/ASC_Siz10:51-82":"70",
    "chr2:10786354-10786673/ASC_Siz10:996-1315":"374",
    "chr2:10797506-10797568/ASC_Siz10:310-372":"125",
    "chr5:112222060-112222183/ASC_Siz10:473-596":"178",
    "chr2:178331262-178331318/ASC_Siz10:154-210":"100",
    }
    pairs = parsePairs(data["ratio"])
    labels = []
    ratio = []
    for amplicon in data["ratio"].keys():
        splits = amplicon.split("/")
        spikeIn = splits[1]
        labels.append(mapping[amplicon])
        if data["ratio"][amplicon] != 'NA' and float(data["ratio"][amplicon]) > 0.000001 and int(data["counts"][spikeIn]) >= 100:
            ratio.append(float(data["ratio"][amplicon]))
        else:
            ratio.append(0.0)

    ratio_reorder = [y for _,y in sorted(zip(map(int,labels),ratio))] 
    label_reorder = [x for x,_ in sorted(zip(map(int,labels),ratio))]
    
    # plot counts
    bar_w = 0.0
    index = np.arange(len(ratio_reorder))
    
    plt.figure()
    rects1 = plt.bar(index, ratio_reorder, # bar_w,
        alpha = 0.95,
        color = 'b',
        align='center',
        width = 0.35, 
        # linestyle='-',
        # marker = 'o',
        )
    plt.title('Read Ratio for Inline Controls (Endogenous to Spike-ins)')
    plt.xlabel('In-line Control Amplicon Size (bp)')
    plt.ylabel('Read Ratio')
    plt.xticks(index + bar_w, label_reorder)
    plt.ylim(bottom = 0.0)
    if scale <= 5.0:
        plt.yticks(np.arange(0.0, 5.05, 1.0))
    else:
        plt.yticks(np.arange(0.0, scale + round(scale/5), round(scale/5)))
    plt.margins(0.2)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(fname,dpi=1000)
    plt.close()