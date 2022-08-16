def get_kernel_from_basename(base_name, column_names):
    kernel = [(_, int(_.split('_')[-1])) for _ in column_names if base_name+'_' == _[:len(base_name)+1]]
    kernel = sorted(kernel, key=lambda x: x[-1])
    kernel = [_[0] for _ in kernel]
    return kernel

def shorten_col_name(col_name, renames={'CenterIn': 'CI',
                                        'CenterOut': 'CO',
                                        'SideIn': 'SI',
                                        'SideOut': 'SO',
                                        'photometry': '',
                                        'Index': ''}):
    out_col_name = col_name
    for base in renames:
        out_col_name = out_col_name.replace(base, renames[base])
    return out_col_name