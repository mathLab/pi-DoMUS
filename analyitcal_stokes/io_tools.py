#import PetscBinaryIO
from scipy.sparse import *
from scipy import *
import numpy as np
from sympy import *

def function_to_cpp_source(func,directory,filename,classname):
    dim = func.shape[0]
    template = '<'+str(dim)+','+str(dim)+','+'1'+'>'
    out_file = open (directory+filename+'.cpp', 'w')
    out_file.write('/*\n')
    out_file.write(' * stk_force_3d.cpp\n')
    out_file.write(' *\n')
    out_file.write(' *  Created on: Mar 24, 2014\n')
    out_file.write(' *      Author: nicola\n')
    out_file.write(' */\n')
    out_file.write('#include "'+filename+'.h"\n')
    
    out_file.write('void '+classname+'::evaluate(const std::vector< typename Function'+template+'::PointType > & 	points,\n')
    out_file.write('                   std::vector< typename Function'+template+'::ValueType > & values) const\n')
    out_file.write('{\n')
    out_file.write('    for (uint i =0; i<points.size(); i++)\n')
    out_file.write('    {\n')
    out_file.write('        double x = points[i][0];\n')
    out_file.write('        double y = points[i][1];\n')
    out_file.write('        double z = points[i][2];\n')
    for i in range (0,dim):
        out_file.write('        values[i]['+str(i)+'] = '+ccode(func[i,0])+';\n')
    out_file.write('    }\n')
    out_file.write('};\n')
    return

def function_to_cpp_header(dim_domain,func,directory,filename,classname):
    dim = func.shape[0]
    template = '<'+str(dim_domain)+','+str(dim)+','+'1'+'>'
    out_file = open (directory+filename+'.h', 'w')
    out_file.write('/*\n')
    out_file.write(' * '+filename+'.h\n')
    out_file.write(' *\n')
    out_file.write(' *  Created on: Mar 24, 2014\n')
    out_file.write(' *      Author: nicola\n')
    out_file.write(' */\n')
    out_file.write('\n')
    out_file.write('#ifndef '+filename+'_h_\n')
    out_file.write('#define '+filename+'_h_\n')
    out_file.write('\n')
    out_file.write('#include <igatools/base/function.h>\n')
    out_file.write('\n')
    out_file.write('using namespace iga;\n')
    out_file.write('using namespace std;\n')
    out_file.write('\n')
    out_file.write('class '+classname+' : public Function'+template+'\n')
    out_file.write('{\n')
    out_file.write('\n')
    out_file.write('public:\n')
    out_file.write('    '+classname+' () : Function'+template+'() {}\n')
    out_file.write('\n')
    out_file.write('    void evaluate(const ValueVector< typename Function'+template+'::Point > &points,\n')
    out_file.write('                  ValueVector< typename Function'+template+'::Value > &values)		 const\n')
    out_file.write('    {\n')
    out_file.write('        for (uint i =0; i<points.size(); i++)\n')
    out_file.write('        {\n')
    out_file.write('            double x = points[i][0];\n')
    out_file.write('            double y = points[i][1];\n')
    out_file.write('            double z = points[i][2];\n')
    out_file.write('\n')
    for i in range (0,dim):
        out_file.write('            values[i]['+str(i)+'] = '+ccode(func[i,0])+';\n')
    out_file.write('        }\n')
    out_file.write('    };\n')
    out_file.write('    static shared_ptr<const Function'+template+'>\n')
    out_file.write('                                            create()\n')
    out_file.write('    {\n')
    out_file.write('        return(shared_ptr<const Function'+template+'>(new '+classname+'())) ;\n')
    out_file.write('    }\n')
    out_file.write('};\n')
    out_file.write('\n')
    out_file.write('#endif /* '+filename+'_h_ */\n')
    out_file = open (directory+filename+'.cpp', 'w')
    out_file.write('/*\n')
    out_file.write(' * '+filename+'.cpp\n')
    out_file.write(' *\n')
    out_file.write(' *  Created on: Mar 24, 2014\n')
    out_file.write(' *      Author: nicola\n')
    out_file.write(' */\n')
    out_file.write('#include "'+filename+'.h"\n')
    return

def write_time_accuracy_table(filename,base,esponente,\
    p_error_l2,prex_l2_order,\
    u_error_l2,vel_l2_order):
    # write the accuracy table 
    # now it is prepared for L2 prex and L2 velocity
    # prex_l2_order should be dimenasion-1 with respect 
    # p_error_l2, same thing for velocity

    midrule = '\\cmidrule{1-1}'
    midrule += '\\cmidrule{3-5}'
    midrule += '\\cmidrule{7-9}'
    
    out_file = open(filename, "w")
    riga = '\\begin{tabular}{cc cc cc cc cc}\\toprule'
    out_file.write(riga)
    out_file.write("\n")
    riga = '$\Delta t$ & & $||\mathbf{X}_{\mathit{ex}}-\mathbf{X}_h||_{L^2}$ & & $L^2$-rate & & $||\mathbf{u}_{\mathit{ex}}-\mathbf{u}_h||_{L^2}$ & & $L^2$-rate & \\\\'
    riga += midrule
    out_file.write(riga)
    out_file.write("\n")
    riga = str(int(base[0]))+'$\cdot 10^{-'+str(int(esponente[0]))+'}$& & '
    riga += "%7.5e" % (p_error_l2[0])
    riga += '&  &'
    riga += ' - '
    riga += '&  &'
    riga += "%7.5e" % (u_error_l2[0])
    riga += '&  &'
    riga += ' - '
    riga += '&  \\\\'
    riga += midrule
    out_file.write(riga)
    out_file.write("\n")
    for line in np.arange(1,len(base)):
        riga = str(int(base[line]))+'$\cdot 10^{-'+str(int(esponente[line]))+'}$& & '
        riga += "%7.5e" % (p_error_l2[line])
        riga += '&  &'
        riga += "%7.2f" % (prex_l2_order[line-1])
        riga += '&  &'
        riga += "%7.5e" % (u_error_l2[line])
        riga += '&  &'
        riga += "%7.2f" % (vel_l2_order[line-1])
        riga += '&  \\\\'
        if line != len(base)-1:
            riga += midrule
        else:
            riga += '\\bottomrule'
        out_file.write(riga)
        out_file.write("\n")
    out_file.write('\\end{tabular}\n')
    out_file.close()
    return

def write_accuracy_table(filename,mesh_ref,p_error_l2,prex_l2_order,\
    u_error_l2,vel_l2_order):
    # write the accuracy table 
    # now it is prepared for L2 prex and L2 velocity
    # prex_l2_order should be dimenasion-1 with respect 
    # p_error_l2, same thing for velocity

    midrule = '\\cmidrule{1-1}'
    midrule += '\\cmidrule{3-5}'
    midrule += '\\cmidrule{7-9}'
    
    out_file = open(filename, "w")
    riga = '\\begin{tabular}{cc cc cc cc cc}\\toprule'
    out_file.write(riga)
    out_file.write("\n")
    riga = '$h_x$ & & $||p-p_h||_{L^2}$ & & $L^2$-rate & & $||\mathbf{u}-\mathbf{u}_h||_{L^2}$ & & $L^2$-rate & \\\\'
    riga += midrule
    out_file.write(riga)
    out_file.write("\n")
    riga = '$1/'+str(int(mesh_ref[0]))+'$& & '
    riga += "%7.5f" % (p_error_l2[0])
    riga += '&  &'
    riga += ' - '
    riga += '&  &'
    riga += "%7.5f" % (u_error_l2[0])
    riga += '&  &'
    riga += ' - '
    riga += '&  \\\\'
    riga += midrule
    out_file.write(riga)
    out_file.write("\n")
    for line in np.arange(1,mesh_ref.shape[0]):
        riga = '$1/'+str(int(mesh_ref[line]))+'$& & '
        riga += "%7.5f" % (p_error_l2[line])
        riga += '&  &'
        riga += "%7.2f" % (prex_l2_order[line-1])
        riga += '&  &'
        riga += "%7.5f" % (u_error_l2[line])
        riga += '&  &'
        riga += "%7.2f" % (vel_l2_order[line-1])
        riga += '&  \\\\'
        if line != mesh_ref.shape[0]-1:
            riga += midrule
        else:
            riga += '\\bottomrule'
        out_file.write(riga)
        out_file.write("\n")
    out_file.write('\\end{tabular}\n')
    out_file.close()
    return

def read_dealii_block_vector(filename,n_blocks):
    f = open(filename,"r")    
    vec = np.zeros((0,))

    for line in f:
        (header,sep,data) = line.partition(':')
        block_row = int(header[1:])
        data = [float(n) for n in data.split()]
        data = np.array(data)
        vec = np.append(vec,data)
    
    assert (n_blocks == block_row+1),\
    "n blocks mismatch between input and data file: "+filename 
    
    return vec

def read_dealii_block_matrix(filename,n_block_rows,n_block_cols):
    f = open(filename,"r") 
    
    matrix_list = []
    for i in np.arange(0,n_block_rows):
        row = []
        for j in np.arange(0,n_block_cols):
            row.append(np.array([]))
        matrix_list.append(row)
                     
    for line in f:
        if line[:9] == "Component":
            print line[:-1]
            (component,sep,blocks) = line.partition(' ')
            blocks = blocks[1:-2]
            (row_id,sep,cln_id) = blocks.partition(',')
            n_rows = 0
            block_row = int(row_id)
            block_col = int(cln_id)
        else:
            data = [float(n) for n in line.split()]
            data = np.array(data)
            if n_rows == 0:
                matrix = np.zeros((0,data.shape[0]))
                n_cols = data.shape[0]
            #
            data = np.reshape(data,(1,data.shape[0]))
            matrix = np.vstack([matrix,data])
            matrix_list[block_row][block_col] = np.copy(matrix)
            n_rows+=1
    
    f.close()
    
    n_cols = 0
    for i in np.arange(0,n_block_cols):
        n_cols+= matrix_list[0][i].shape[1]
    
    sys_mat = np.zeros((0,n_cols))
    
    
    for i in np.arange(0,n_block_rows):
        row = np.zeros((matrix_list[i][0].shape[0],0))
        for j in np.arange(0,n_block_cols):
            row = np.hstack([row,matrix_list[i][j]])
        sys_mat = np.vstack([sys_mat,row])
    
    print 'Matrix shape = ' + str(sys_mat.shape)
    return sys_mat


def read_dealii_vector(filename):
    header = ''
    with open(filename,'r') as f:
        byte = f.read(1)
        #print byte
        #print f.read(1)
        #print f.read(1)
        while (byte!='['):        
            header += byte
            byte = f.read(1)
        #header = header[:-1]
        print header
        max_len = int(header)
        #brakets = f.read(1)
        val = np.fromfile(f, dtype=np.float64, count=max_len)
    return val

def read_dealii_sparsity_pattern(filename):
    header = ''
    with open(filename,'r') as f:
        byte = f.read(1)
        while (byte!=']'):        
            header += byte
            byte = f.read(1)
        header = header[1:]
        data = [int(n) for n in header.split()]
        [max_dim,\
        rows,\
        cols,\
        max_vec_len,\
        max_row_length,\
        compressed,\
        store_diagonal_first_in_row] = data    
        brakets = f.read(1)
        rowstart = np.fromfile(f, dtype=np.uint64, count=rows+1)
        brakets = f.read(2)
        columns =  np.fromfile(f, dtype=np.uint64, count=max_vec_len)
    return rows,cols,rowstart, columns
    
def read_dealii_matrix_values(filename):
    header = ''
    with open(filename,'r') as f:
        byte = f.read(1)
        while (byte!=']'):        
            header += byte
            byte = f.read(1)
        header = header[1:]
        max_len = int(header)
        #brakets = f.read(1)
        val = np.fromfile(f, dtype=np.float64, count=max_len)
    return val
    
def read_dealii_matrix(filename):
    
    (rows,cols,rowstart, columns) = read_dealii_sparsity_pattern(filename+'_sp')
    val = read_dealii_matrix_values(filename+'_vl')
    
    matrix = csr_matrix((val, columns, rowstart), (rows, cols))
    
    return matrix


def petsc_to_scipy_mat(filename):
    io = PetscBinaryIO.PetscBinaryIO()
    objects = io.readBinaryFile(filename)
    shape = objects[0][0]
    data = objects[0][1][2]
    indices = objects[0][1][1]
    indptr = objects[0][1][0]
    A = csr_matrix((data, indices, indptr), shape)
    return A

def read_bc_map(filename):
    dof_name = filename + '_dof'
    f = open(dof_name,"r")
    bc_dof = np.fromfile(f,dtype=np.int32)
    f.close()

    val_name = filename + '_val'
    f = open(val_name,"r")
    bc_val = np.fromfile(f,dtype=np.float)
    f.close()
    return bc_dof, bc_val

def petsc_to_numpy_vec(filename):
    io = PetscBinaryIO.PetscBinaryIO()
    objects = io.readBinaryFile(filename)
    v = np.array(objects[0])
    return v
