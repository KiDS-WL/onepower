import numpy as np
from cosmosis.datablock import option_section, names as section_names

def setup(options):
    filename = options[option_section, "filepath"]
    des_fmt = options.get_bool(option_section,"des_fmt",default=False)
    histogram = options.get_bool(option_section,"histogram",default=False)

    output_section = options.get_string(option_section, "output_section", default=section_names.wl_number_density)
    single_bin = options.get_int(option_section, "single_bin", default=-666)
    from_fits_header = options.get_string(option_section, "from_fits_header", default="")

    if not from_fits_header:
        data_full = np.loadtxt(filename).T
        if des_fmt:
            z=0.5*(data_full[0]+data_full[1])
            nz=len(z)
            nbin=len(data_full)-3
            n_of_z=data_full[2:-1]
        else:
            nz = len(data_full[0])
            nbin = len(data_full)-1
            z = data_full[0]
            if single_bin!=-666:
                print ("Using nontomographic mode (bin %d)" %single_bin)
                n_of_z = data_full[single_bin]
                nbin = 1
            else:
                n_of_z = data_full[1:]
            if histogram:
            #in this case the sample z values are lower edges of
            #histogram bins.  So to turn them into samples we need to
            #shift everything.  This assumes equal sized bins
                dz = (z[1]-z[0])/2.0
                print ("n(z) set to histogram mode. Bin centers are %f higher than edges." %dz)
                z += dz

        if des_fmt:
            z=0.5*(data_full[0]+data_full[1])
            nz=len(z)
            nbin=len(data_full)-3
            n_of_z=data_full[2:-1]
        else:
            nz = len(data_full[0])
            nbin = len(data_full)-1
            z = data_full[0]
            if single_bin!=-666:
                n_of_z = data_full[single_bin]
                nbin = 1
            else:
                n_of_z = data_full[1:]
    
            #check first z is zero, if not add some
            if z[0]>0.00000001:
                z_new=np.zeros(len(z)+1)
                z_new[1:]=z
                n_of_z_new=np.zeros((nbin,len(z)+1))
                n_of_z_new[:,1:]=n_of_z
                z,n_of_z=z_new,n_of_z_new
                
               #Normalize n(z)
            for col in n_of_z:
                norm = np.trapz(col, z)
                col/=norm

        print ("Found %d samples and %d bins in redshift in file %s" % (nz, nbin, filename))
        return (nz, nbin, z, n_of_z), output_section, from_fits_header
    else:
        return (0, 0, [0.], [0]), output_section, from_fits_header

def execute(block, config):
    (nz, nbin, z, n_of_z), output_section, from_fits_header = config

    output_section="nz_"+output_section
    if from_fits_header:
        #block = rename_section(block, "nz_%s"%from_fits_header, output_section)
        nbin = block["nz_"+from_fits_header, "nbin"]
        block.put_int(output_section, "nzbin", block["nz_"+from_fits_header, "nbin"])
        block.put_int(output_section, "nbin", block["nz_"+from_fits_header, "nbin"])
        block.put_double_array_1d(output_section, "z", block["nz_%s"%from_fits_header, "z"])

        for n in xrange(nbin):
            block[output_section,"bin_%d"%(n+1)] = block["nz_%s"%from_fits_header,"bin_%d"%(n+1)] 

    else:
        block[output_section, 'nz'] = nz
        block[output_section, 'nbin'] = nbin
        block[output_section, 'z'] = z

        if output_section==section_names.wl_number_density:
            bin_name = "nbin"
        else:
            bin_name = "nzbin"
        print (bin_name)
        print (nbin)
        block[output_section, 'nz'] = nz
        block[output_section, bin_name] = nbin
        block[output_section, 'z'] = z
        for (bin, bin_n_of_z) in enumerate(n_of_z):
            name = "bin_%d"%(bin+1)
            block[output_section, name] =  bin_n_of_z

    return 0

def rename_section(block, oldname, newname):
    #print ("Renaming section %s (calling it %s)"%(oldname, newname))
    for key in block.keys(oldname):
        block[newname, key[1] ] = block[oldname, key[1]]
    return block


def cleanup(config):
    #nothing to do here!  We just include this 
    # for completeness
    return 0
