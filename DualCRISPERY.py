#%%
import os
from time import time
import csv
import numpy as np
import multiprocessing
from numba import njit
from numpy.core.arrayprint import _TimelikeFormat 
import tqdm as pb
import datetime
# %%
@njit
def binary_subtract(array1,array2,mismatch):
    
    """ Used for matching 2 sequences based on the allowed mismatches.
    Requires the sequences to be in numerical form"""
    
    miss=0
    for arr1,arr2 in zip(array1,array2):
        if arr1-arr2 != 0:
            miss += 1
        if miss>mismatch:
            return 0
    return 1

@njit
def sgrna_all_vs_all(binary_sgrna,read,mismatch):
    
    """ Runs the loop of the read vs all sgRNA comparison.
    Sends individually the sgRNAs for comparison.
    Returns the final mismatch score"""
    
    found = 0
    for guide in binary_sgrna:
        if binary_subtract(binary_sgrna[guide],read,mismatch):
            found+=1
            found_guide = guide
            if found>=2:
                return
    if found==1:
        return found_guide
    return
def seq2bin(sequence):
    
    """ Converts a string to binary, and then to 
    a numpy array in int8 format"""
    
    byte_list = bytearray(sequence,'utf8')
    return np.array((byte_list), dtype=np.int8)

def binary_converter(sgrna):

    """ Parses the input sgRNAs into binary dictionaries. Converts all DNA
    sequences to their respective binary array forms. This gives some computing
    speed advantages with mismatches."""
    
    from numba import types
    from numba.typed import Dict
    
    container = Dict.empty(key_type=types.unicode_type,
                            value_type=types.int8[:])

    for sequence in sgrna:
        container[sequence] = seq2bin(sequence)
    return container


# %%

def guides_loader(guides):
    
    """ parses the sgRNA names and sequences from the indicated sgRNA .csv file.
    Creates a dictionary using the sgRNA sequence as key. If duplicated sgRNA sequences exist, 
    this will be caught in here"""
    
    
    if not os.path.isfile(guides):
        input("\nCheck the path to the sgRNA file.\nNo file found in the following path: {}\nPress any key to exit".format(guides))
        raise Exception
    
    sgrna = {}
    with open(guides) as current:
        for i, line in enumerate(current):
            line = line[:-1].split(",")
            sequence = line[1].upper()
            sequence = sequence.replace(" ", "")
            
            if sequence not in sgrna:
                sgrna[sequence] = {"name":line[0].upper(), "index":i}
                
            #else:
            #    print("\nWarning!!\n{} and {} share the same sequence. Only {} will be considered valid.\n".format(sgrna[sequence], line[0],sgrna[sequence]))

    return sgrna
def cpu_counter():
    
    """ counts the available cpu cores, required for spliting the processing
    of the files """
    
    cpu = multiprocessing.cpu_count()
    if cpu >= 2:
        cpu -= 1
    if cpu >= 8:
        return 8
    return cpu

# %%

def getCombinations(f1, f2, guides):
    """ read both R1 and R2 files simultaneously, and if both R1 and R2 sequences match a sgRNA, add 1 to the count of the combination """
    tempo = time()
    mat = np.zeros(((len(guides)+1), (len(guides))+1), np.int32)
    guides_binary = binary_converter(guides)
    c = 0
    current = multiprocessing.current_process()
    pos = current._identity[0]-1
    total_file_size = min(os.path.getsize(f1), os.path.getsize(f2))
    filename = str(os.path.basename(f1)).replace("R1_", "paired")
    filename = filename.replace(".fastq", "")
    tqdm_text = "Processing file " + filename
    pbar = pb.tqdm(total=total_file_size,desc=tqdm_text, position=pos,colour="green",leave=False)
    # n = set("N")
    mismatch = [n for n in range(1, 4)] #1, or 2, or 3 missmatches
    readCounter, readCounterExact, readCounterMis = 0, 0, 0

    failReads = {}

    #open both R1 and R2 files at the same time
    with open(f1) as file1, open(f2) as file2:
        lineCounter = 1
        for line1, line2 in zip(file1, file2):
            er = False
            #if the current line is the sequence, and "er" is not true.
            if lineCounter == 2 and not er:
                readCounter += 1
                read1 = line1[:-1]
                read2 = line2[:-1]
                #both read1 and read2 have perfect match with one sgRNA
                if read1 in guides and read2 in guides:
                    mat[guides[read1]["index"], guides[read2]["index"]] += 1
                    readCounterExact += 1                
                # elif len(n.intersection(read1)) <= 1 and (len(n.intersection(read2)) <= 0):
                elif read1 not in failReads and read2 not in failReads:                    
                    #mismatch only if the read1 and the read2 does not contain N
                    read2bin=seq2bin(read2)
                    for miss in mismatch:
                        finder2 = sgrna_all_vs_all(guides_binary, read2bin, miss)
                        if finder2 is not None:
                            break
                    if finder2 is not None:
                        read1bin=seq2bin(read1)
                        for miss in mismatch:
                            finder1 = sgrna_all_vs_all(guides_binary, read1bin, miss)
                            if finder1 is not None:
                                break
                        if finder1 is not None:
                            mat[guides[finder1]["index"], guides[finder2]["index"]] += 1
                            readCounterMis += 1
                        else:
                            failReads[read1] = True
                    else:
                        failReads[read2] = True

                    
            lineCounter = (lineCounter + 1) % 4
            pbar.update(len(line1))
            c += 1 
    tempo = time() - tempo
    if tempo > 60:
        timing = str(round(tempo / 60, 2)) + " minutes"   
    else:
        timing = str(round(tempo, 2)) + " seconds"
    with open("log.txt", "a") as logFile:
        logFile.write(f"\n#Read counts ran in {timing} for file {filename}\n\t{readCounterExact+readCounterMis} reads out of {readCounter} were considered valid ({(readCounterExact+readCounterMis)/readCounter*100}%)\n\t{readCounterExact} were perfectly aligned\n\t{readCounterMis} were aligned with mismatch")
    failReads = {}

    return [filename, mat]



# %%
dir = "./Fastq"
files = []
guides = guides_loader("D39V_guides_869.csv")
for i, filename in enumerate(os.listdir(dir)):
    if filename.endswith(".fastq") or filename.endswith(".fq"):
        files.append(os.path.join(dir, filename))

result_objs = []
#check if even number of files
#check if file have R1/R2
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open("log.txt", "a") as logFile:
    logFile.write(f"Starting read counter with {len(files)} files at {current_time}\n")
tempo = time()
results = []
pool = multiprocessing.Pool(processes = cpu_counter(), initargs=(multiprocessing.RLock(),),initializer=pb.tqdm.set_lock)
for i in range(0, len(files), 2):
    result=pool.apply_async(getCombinations, args=((files[i], files[i+1], guides)))
    result_objs.append(result)
pool.close()
pool.join()
results = [res.get() for res in result_objs]
# %%
mats = {}
for obj in results:
    mats[obj[0]] = np.subtract(np.add(np.triu(obj[1]), np.tril(obj[1]).T), np.diag(np.diag(obj[1])))

# %%
with open("raw869_NextSeq_NovaSeq.csv", "w", encoding='UTF8', newline='') as f:
    ar = ["SG1", "SG2"]
    ar.extend([n for n in mats.keys()])
    csv.writer(f).writerow(ar)
    for i in range(len(guides)+1):
        for j in range(i, len(guides)+1):
            row = ["sgRNA" + str(i+1), "sgRNA" + str(j+1)]
            for obj in mats:
                row.append(mats[obj][i, j])
            csv.writer(f).writerow(row)

tempo = time() - tempo
if tempo > 60:
    timing = str(round(tempo / 60, 2)) + " minutes"   
else:
    timing = str(round(tempo, 2)) + " seconds"
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open("log.txt", "a") as logFile:
    logFile.write(f"\nProgram finished at {current_time} in {timing}\n------------------------------------------------------------\n")




# %%
