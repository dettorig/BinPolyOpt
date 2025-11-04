import re
import numpy as np
import os
from shutil import copyfile

def subsets(mySet):
    result = [[]]
    for x in mySet:
        result.extend([subset + [x] for subset in result])
    return result

def rename_and_write(filename, varnames):
    elem_reg = re.compile(r'^\(')
    with open(filename + '.poi.ieq', 'r') as r, open(filename + '.inequalities', 'w') as w:
        for line in r:
            if elem_reg.match(line):
                for k in sorted(varnames.keys(), reverse=True):
                    line = line.replace("x" + str(k), varnames[k])
            w.write(line)

def main(incidence_file):
    inc_matrix = np.loadtxt(incidence_file, dtype=int)

    with open(incidence_file, 'r') as file_handle:
        lines_list = file_handle.readlines()

    n = sum(1 for ch in lines_list[0] if ch not in {' ', '\n'})
    m = len(lines_list)

    vars = {}
    varnames = {}

    for i in range(1, n + 1):
        loop = "x" + str(i)
        varnames[i] = loop
        vars[i] = {loop}

    vary = n
    for line in lines_list:
        vary += 1
        edge = set()
        edgevar = 0
        value = 'y'
        for val in line.split():
            edgevar += 1
            if val != '0':
                edge.add("x" + str(edgevar))
                value += str(edgevar)
        vars[vary] = edge
        varnames[vary] = value

    nodes = {"x" + str(i) for i in range(1, n + 1)}
    allpoints = subsets(nodes)

    porta_string = f"DIM = {n + m}\n\nCONV_SECTION\n"
    for idx, point in enumerate(allpoints, start=1):
        porta_string += f"({idx}) "
        for i in range(1, n + m + 1):
            porta_string += " 1 " if vars[i].issubset(point) else " 0 "
        porta_string += "\n"
    porta_string += "END\n"

    filename = f"MPG{n},{m}"
    copyfile(incidence_file, filename + '.im')

    with open(filename + '.poi', 'w') as f:
        f.write(porta_string)

    porta_path = os.path.join(os.getcwd(), 'xporta.exe')
    result = os.system(f'"{porta_path}" -T {filename}.poi')
    if result != 0:
        raise RuntimeError(f"PORTA execution failed with error code {result}")

    rename_and_write(filename, varnames)

    return filename + ".poi.ieq", n, m

def generate_ieq_from_incidence(base_name=None):
    incidence_file = base_name + ".txt"
    
    if not os.path.exists(incidence_file):
        print(f"File '{incidence_file}' not found.")
        return None

    main(incidence_file)

    with open(incidence_file, 'r') as f:
        lines = [line for line in f.readlines() if line.strip()]

    n = sum(1 for ch in lines[0] if ch not in {' ', '\n'})
    m = len(lines)

    filename = f"MPG{n},{m}"
    return filename + ".poi.ieq", n, m

if __name__ == "__main__":
    filename, n, m = generate_ieq_from_incidence()
    print(f"Generated file: {filename}")
    print(f"Vertices: {n}, Edges: {m}")
