def imp_bin_data(fn1, fn2, encoding="utf-8"):
    import numpy
    no_sa, no_ch, no_se, no_gr, no_mo = imp_hdr_param(fn2, encoding=encoding)
    data = numpy.fromfile(fn1, dtype=numpy.float32)
    nl = float(len(data)) / float(no_ch)
    nl = int(nl)
    data = data.reshape(nl, no_ch)
    return data


def imp_hdr_param(fn, encoding="utf-8"):
    num_lines = sum(1 for line in open(fn, encoding=encoding))

    search = 'number_of_samples='
    i = 0
    F = open(fn, 'r', encoding=encoding)
    while i < num_lines:
        f = F.readline()
        if search in f:
            break
        i = i + 1
    no_sa = int(''.join(list(filter(str.isdigit, f))))

    search = 'number_of_channels='
    i = 0
    F = open(fn, 'r', encoding=encoding)
    while i < num_lines:
        f = F.readline()
        if search in f:
            break
        i = i + 1
    no_ch = int(''.join(list(filter(str.isdigit, f))))

    search = 'number_of_sensors='
    i = 0
    F = open(fn, 'r', encoding=encoding)
    while i < num_lines:
        f = F.readline()
        if search in f:
            break
        i = i + 1
    no_se = int(''.join(list(filter(str.isdigit, f))))

    search = 'number_of_groups='
    i = 0
    F = open(fn, 'r', encoding=encoding)
    while i < num_lines:
        f = F.readline()
        if search in f:
            break
        i = i + 1
    no_gr = int(''.join(list(filter(str.isdigit, f))))

    search = 'number_of_modules='
    i = 0
    F = open(fn, 'r', encoding=encoding)
    while i < num_lines:
        f = F.readline()
        if search in f:
            break
        i = i + 1
    no_mo = int(''.join(list(filter(str.isdigit, f))))

    return no_sa, no_ch, no_se, no_gr, no_mo


def imp_samp_freq(fn, encoding="utf-8"):
    num_lines = sum(1 for line in open(fn, encoding=encoding))

    search = 'sampling_exponent='
    i = 0
    F = open(fn, 'r', encoding=encoding)
    while i < num_lines:
        f = F.readline()
        if search in f:
            break
        i = i + 1
    sa_ex = float(f.partition("=")[2])

    search = 'sampling_step='
    i = 0
    F = open(fn, 'r', encoding=encoding)
    while i < num_lines:
        f = F.readline()
        if search in f:
            break
        i = i + 1
    sa_st = float(f.partition("=")[2])

    return sa_st * 10 ** (sa_ex)
