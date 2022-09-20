
from paths import *
from utils import read_file

suffix = ''
# suffix = '_q0_35'

name_list = read_file(name='high_fill_list.csv').name
if suffix == '':
    PPN2 = open(output_path + '/total_chain_PPN2.txt', 'w')
    PPO3N2 = open(output_path + '/total_chain_PPO3N2.txt', 'w')
D16 = open(output_path + '/total_chain_D16' + suffix + '.txt', 'w')
for name in name_list:
    file_path = output_path + '/output' + suffix + '/total_chain_' + name + '.txt'
    f = open(file_path, 'r')
    if suffix == '':
        PPN2.write("%s " %(name))
        PPO3N2.write("%s " %(name))
    D16.write("%s " %(name))
    for i, line in enumerate(f.readlines()):
        if suffix == '':
            if i == 0:
                PPN2.write(line)
            if i == 1:
                PPO3N2.write(line)
            if i == 2:
                D16.write(line)
        else:
            if i == 0:
                D16.write(line)

    f.close()

if suffix == '':
    PPN2.close()
    PPO3N2.close()
D16.close()
