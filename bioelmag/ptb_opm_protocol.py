def read_sens_info(sens_pos_fn, num_sens=None):
    # full path to the measurement directory
    # the files have to be exported to .csv
    # number of sensors you want to import
    # returns "sens_hold_idx" which corresponds to the
    # holes on OPM sensor holders built at PTB and
    # "con_position" which corresponds to # of channel
    # in con file

    import numpy as np
    sens_info = {}

    # filename = f"{meas_dir}Positions/SensorPositions_{block_name}.csv"
    with open(sens_pos_fn) as f:
        # sens_pos = np.loadtxt((x.replace(',', "\t") for x in f),
        sens_pos = np.loadtxt((x for x in f), skiprows=1,
                              delimiter="\t", max_rows=num_sens,
                              dtype=str)

    file_sens_idx = sens_pos[:, 0]
    file_sens_hold_idx = sens_pos[:, 1]
    file_sens_id = sens_pos[:, 2]
    file_sens_dataarr_pos = sens_pos[:, 3]
    file_sens_type = sens_pos[:, 4]

    directions = ["z", "y", "x"]
    sens_names = []
    sens_datapos = []
    sens_holdpos = []
    sens_type = []
    for i, j in enumerate(file_sens_idx):
        if file_sens_type[i] == "oMEG":
            for k, l in enumerate(file_sens_dataarr_pos[i].split(", ")):
                sens_type.append("mag")
                sens_datapos.append(int(l))
                sens_names.append(
                    file_sens_type[i] + file_sens_id[i] + directions[k])
                sens_holdpos.append(file_sens_hold_idx[i] + directions[k])
        else:
            sens_type.append(file_sens_type[i])
            sens_datapos.append(int(file_sens_dataarr_pos[i]))
            sens_names.append(file_sens_type[i] + file_sens_id[i])
            sens_holdpos.append("")

    sens_info["sens_names"] = sens_names
    sens_info["sens_datapos"] = sens_datapos
    sens_info["sens_type"] = sens_type
    sens_info["sens_holdpos"] = sens_holdpos
    #
    return sens_info


def read_sens_info_v2(sens_pos_fn, num_sens=None):
    # full path to the measurement directory
    # the files have to be exported to .csv
    # number of sensors you want to import
    # returns "sens_hold_idx" which corresponds to the
    # holes on OPM sensor holders built at PTB and
    # "con_position" which corresponds to # of channel
    # in con file

    import numpy as np
    sens_info = {}

    # filename = f"{meas_dir}Positions/SensorPositions_{block_name}.csv"
    with open(sens_pos_fn) as f:
        # sens_pos = np.loadtxt((x.replace(',', "\t") for x in f),
        sens_pos = np.loadtxt((x for x in f), skiprows=1,
                              delimiter="\t", max_rows=num_sens,
                              dtype=str)

    file_sens_idx = sens_pos[:, 0]
    file_sens_hold_idx = sens_pos[:, 1]
    file_sens_name = sens_pos[:, 3]
    file_sens_type = sens_pos[:, 4]

    directions = ["x", "y", "z"]
    sens_holdpos = []
    sens_type = []
    sens_names = []
    for i, j in enumerate(file_sens_idx):
        if file_sens_type[i] == "oMEG":
            for k, l in enumerate(file_sens_name[i].split(", ")):
                sens_names.append(l)
                sens_type.append("mag")
                sens_holdpos.append(file_sens_hold_idx[i] + directions[k])
        else:
            sens_type.append(file_sens_type[i])
            sens_names.append(file_sens_name[i])
            sens_holdpos.append("")

    sens_info["sens_names"] = sens_names
    sens_info["sens_type"] = sens_type
    sens_info["sens_holdpos"] = sens_holdpos
    #
    return sens_info


def read_fnirs_chan_info(sens_pos_fn, num_sens=None):
    # I have to include this in bioelmag.ptb_opm_protocol
    # full path to the measurement directory
    # the files have to be exported to .csv
    # number of sensors you want to import
    # returns "sens_hold_idx" which corresponds to the
    # holes on OPM sensor holders built at PTB and
    # "con_position" which corresponds to # of channel
    # in con file

    import numpy as np
    chan_info = {}

    # filename = f"{meas_dir}Positions/SensorPositions_{block_name}.csv"
    with open(sens_pos_fn) as f:
        # sens_pos = np.loadtxt((x.replace(',', "\t") for x in f),
        sens_pos = np.loadtxt((x for x in f), skiprows=1,
                              delimiter="\t", max_rows=num_sens,
                              dtype=str)

    file_sens_idx = sens_pos[:, 0]
    file_sens_hold_idx = sens_pos[:, 1]
    file_sens_id = sens_pos[:, 2]
    file_sens_dataarr_pos = sens_pos[:, 3]
    file_sens_type = sens_pos[:, 4]

    sens_names_sources = []
    sens_names_detectors = []
    sources_dataarr_poss = []
    detectors_dataarr_poss = []
    sources_hold_idx = []
    detectors_hold_idx = []

    for i, j in enumerate(file_sens_idx):
        if "S" in file_sens_type[i]:
            sens_names_sources.append(file_sens_type[i] + file_sens_id[i])
            sources_dataarr_poss.append(int(file_sens_dataarr_pos[i]))
            sources_hold_idx.append(file_sens_hold_idx[i])
        elif "D" in file_sens_type[i]:
            sens_names_detectors.append(file_sens_type[i] + file_sens_id[i])
            detectors_dataarr_poss.append(int(file_sens_dataarr_pos[i]))
            detectors_hold_idx.append(file_sens_hold_idx[i])

    ch_names = []
    ch_types = []
    ch_datapos = []
    ch_holdpos = []

    for ind_i, i in enumerate(sens_names_detectors):
        for ind_j, j in enumerate(sens_names_sources):
            ch_names.append(i + "_" + j + " hbo")
            ch_types.append("hbo")
            ch_datapos.append((sources_dataarr_poss[ind_j],
                               detectors_dataarr_poss[ind_i]))
            ch_holdpos.append((detectors_hold_idx[ind_i],
                               sources_hold_idx[ind_j]))
            ch_names.append(i + "_" + j + " hbr")
            ch_types.append("hbr")
            ch_datapos.append((sources_dataarr_poss[ind_j],
                               detectors_dataarr_poss[ind_i]))
            ch_holdpos.append((detectors_hold_idx[ind_i],
                               sources_hold_idx[ind_j]))

    chan_info["ch_names"] = ch_names
    chan_info["ch_datapos"] = ch_datapos
    chan_info["ch_types"] = ch_types
    chan_info["ch_holdpos"] = ch_holdpos

    return chan_info


def import_flt_hdr(header_name, value_name):
    import bioelmag.ptb_flthdr as pfhr
    import numpy as np

    data = np.transpose(pfhr.imp_bin_data(value_name, header_name,
                                          encoding="ISO-8859-1"))
    data = data * (10 ** -6)
    sfreq = 1/(pfhr.imp_samp_freq(header_name, encoding="ISO-8859-1"))

    return data, sfreq


def read_meas_info(meas_info_fn, block_name):
    # full path of the SensorPositions file
    # the files have to be exported to .csv
    # number of sensors you want to import
    # block name can either be str or int
    import numpy as np
    import sys

    measurements = np.loadtxt(meas_info_fn, delimiter=";", skiprows=1,
                              dtype=str)

    if isinstance(block_name, str):
        meas_idx = np.where(measurements[:, 1] == block_name)
        if len(meas_idx[0]) == 0:
            print("this block name does not exists")
        else:
            meas_idx = meas_idx[0][0]
    elif isinstance(block_name, int):
        meas_idx = block_name
    else:
        print("block name is not a string nor list")
        sys.exit(1)

    # now we calculate the sensors to reorient
    reorient_idx = []
    reorient_idx_string = measurements[meas_idx, 11]
    if len(reorient_idx_string) > 0:
        reorient_idx_string = reorient_idx_string.split(",")
        for i in reorient_idx_string:
            reorient_idx.append(int(i))

    # now we calculate the bads sensors indexes
    bads_idx = []
    bads_idx_string = measurements[meas_idx, 8]
    if len(bads_idx_string) > 0:
        bads_idx_string = bads_idx_string.split(",")
        for i in bads_idx_string:
            bads_idx.append(i)

    meas_info = {
                 "meas_state": int(measurements[meas_idx, 0]),
                 "gain": float(measurements[meas_idx, 4]),
                 # 'sens_num': int(measurements[meas_idx, 7]),
                 "geom_name": measurements[meas_idx, 12],
                 "protocol": measurements[meas_idx, 13],
                 "gen12": int(measurements[meas_idx, 3]),
                 "importgeometry": int(measurements[meas_idx, 5]),
                 "t_delay": float(measurements[meas_idx, 6]),
                 # "trigger_ch": int(measurements[meas_idx, 10]),
                 "dataformat": measurements[meas_idx, 9],
                 "bads_idx": bads_idx,
                 "block_name": measurements[meas_idx, 1],
                 # "meas_dir": meas_dir,
                 "reorient_idx": reorient_idx,
                 "additional_block_name": measurements[meas_idx, 14],
                 "bad_epochs":measurements[meas_idx, 15]
                 }

    # sens_hold, con_pos = read_sensor_positions(meas_dir, block_name,
    #                                            meas_info["sens_num"])
    # meas_info["con_pos"] = con_pos
    # meas_info["sens_hold"] = sens_hold

    return meas_info


def create_mne_raw(sens_info, meas_info, data_path, channel_factor=(10 ** -9),
                   sens_hol_path="", subject_dir="", opm_trans_path="",
                   geom_name="", move_fn="", return_pos_dict=False):
    import mne
    import numpy as np
    import bioelmag.vector_functions as vfun

    if meas_info["dataformat"] == "flt":
        header_name = data_path[0]
        value_name = data_path[1]
        data, sfreq = import_flt_hdr(header_name, value_name)

    if meas_info["importgeometry"] == 1:
        pos_dict = create_sens_pos_dict(sens_hol_path, "all")
        if len(move_fn) > 0:
            pos_dict = move_pos_dict(pos_dict, move_fn)
        pos_dict = rotate_translate_pos_dict(
            pos_dict, opm_trans_path, name=geom_name, subject_dir=subject_dir,
            gen12=meas_info["gen12"])

    info = mne.create_info(ch_names=sens_info["sens_names"], sfreq=sfreq,
                           ch_types=sens_info["sens_type"])

    unit_v = np.array([0.0, 0.0, 1.0])
    for i, j in enumerate(info.ch_names):
        if "oMEG" in j:
            info['chs'][i]['coil_type'] = 9999
            info['chs'][i]['scanno'] = i + 1
            info['chs'][i]['logno'] = i + 1
            info['chs'][i]['kind'] = 1
            info['chs'][i]['range'] = 1.0
            info['chs'][i]['cal'] = 3.7000000285836165e-10
            info['chs'][i]['unit'] = 112
            sens_hold_idx = sens_info['sens_names'].index(j)
            if meas_info["importgeometry"] == 1:
                sens_hold_name = sens_info['sens_holdpos'][
                    sens_hold_idx]
                holders = pos_dict[sens_hold_name]
                rot_mat = vfun.create_rot_matrix(holders[3:6], unit_v)
                info['chs'][i]['loc'] = np.array(
                    [holders[0], holders[1], holders[2], rot_mat[0, 0],
                     rot_mat[0, 1], rot_mat[0, 2], rot_mat[1, 0],
                     rot_mat[1, 1], rot_mat[1, 2], rot_mat[2, 0],
                     rot_mat[2, 1], rot_mat[2, 2]])
            data[sens_info['sens_datapos'][sens_hold_idx]] = data[
                sens_info['sens_datapos'][sens_hold_idx]] * channel_factor

    raw = mne.io.RawArray(data[sens_info['sens_datapos']], info)

    if return_pos_dict:
        return raw, pos_dict

    else:
        return raw


def move_pos_dict(pos_dict, move_fn):
    # This function has to be moved to ptb_opm_protocol
    import numpy as np
    import bioelmag.vector_functions as vfun

    try:
        with open(move_fn) as fp:
            for line in fp:
                line_list = line.split(" ")
                line_list[-1] = line_list[-1].strip()
                # print(line_list)

                # Very good find see: https://stackoverflow.com/questions/38974
                # 168/finding-entries-containing-a-substring-in-a-numpy-array
                # np.flatnonzero(np.core.defchararray.find(
                #   list(pos_dict.keys()), line_list[1]) != -1)

                # affect_list is the all channels that have to be
                # manipulated
                idx = np.array([s.find(line_list[1]) for s in list(pos_dict.keys())])
                idx = np.where(idx >= 0)[0].tolist()
                # idx = np.flatnonzero(np.core.defchararray.find(
                #     list(pos_dict.keys()), line_list[1]) != -1)
                affect_list = np.array(list(pos_dict.keys()))[idx].tolist()

                if line_list[0] == "mv":
                    # mv_vec_idx = np.flatnonzero(np.core.defchararray.find(
                    #     affect_list, line_list[2]) != -1)
                    mv_vec_idx = np.array([s.find(line_list[2]) for s in affect_list])
                    mv_vec_idx = np.where(mv_vec_idx > 0)[0].tolist()

                    mv_vec_name = affect_list[mv_vec_idx[0]]
                    mv_vector = pos_dict[mv_vec_name][3:6]
                    mv_vector = (mv_vector/np.linalg.norm(mv_vector)) * \
                        float(line_list[3])
                    for i in affect_list:
                        pos_dict[i][0:3] = pos_dict[i][0:3] + mv_vector

                if line_list[0] == "rt":
                    # rt_vec_idx = np.flatnonzero(np.core.defchararray.find(
                    #     affect_list, line_list[2]) != -1)
                    rt_vec_idx = np.array([s.find(line_list[2]) for s in affect_list])
                    rt_vec_idx = np.where(rt_vec_idx > 0)[0].tolist()

                    rt_vec_name = affect_list[rt_vec_idx[0]]
                    for i in affect_list:
                        # pos_dict[i][3:6] = rove.rotateAbout(
                        #     pos_dict[rt_vec_name][3:6], pos_dict[i][3:6],
                        #     line_list[3])
                        rota = vfun.rotation_matrix(pos_dict[rt_vec_name][3:6],
                                                    float(line_list[3]))
                        pos_dict[i][3:6] = np.dot(rota, pos_dict[i][3:6])
    except IOError:
        print("No SensorManipulation.txt file")

    return pos_dict


def create_sens_pos_dict(fn, fileformat):
    # fileformat can be either "all" or "occupied"
    import re
    import numpy as np

    pos_dict = {}

    with open(fn, "r") as F:
        num_lines = len(open(fn).readlines())

        i = 0
        while i < num_lines:
            f = F.readline()
            ff = f.replace('"', '')
            ff = ff.replace(':', ',')
            ff = ff.replace('\n', '')
            ff = list(filter(None, re.split(', ', ff)))
            i = i + 1

            if fileformat == "occupied":
                if int(ff[4]) > 0:
                    f = F.readline()
                    ffz = f.replace('"', '')
                    ffz = ffz.replace(':', ',')
                    ffz = ffz.replace('\n', '')
                    ffz = list(filter(None, re.split(', ', ffz)))
                    i = i + 1
                    pos_dict[ff[4]+"z"] = np.array(ff[1:4] + ffz[1:4],
                                                   dtype=float)

                    f = F.readline()
                    ffy = f.replace('"', '')
                    ffy = ffy.replace(':', ',')
                    ffy = ffy.replace('\n', '')
                    ffy = list(filter(None, re.split(', ', ffy)))
                    i = i + 1
                    pos_dict[ff[4] + "y"] = np.array(ff[1:4] + ffy[1:4],
                                                     dtype=float)

                    dirx = np.cross(np.array(ffz[1:4], dtype=float),
                                    np.array(ffy[1:4], dtype=float))
                    pos_dict[ff[4] + "x"] = np.concatenate(
                        (np.array(ff[1:4], dtype=float), dirx))

            elif fileformat == "all":
                f = F.readline()
                ffz = f.replace('"', '')
                ffz = ffz.replace(':', ',')
                ffz = ffz.replace('\n', '')
                ffz = list(filter(None, re.split(', ', ffz)))
                i = i + 1
                pos_dict[ff[2] + "z"] = np.array(ff[3:6] + ffz[3:6],
                                                 dtype=float)

                f = F.readline()
                ffy = f.replace('"', '')
                ffy = ffy.replace(':', ',')
                ffy = ffy.replace('\n', '')
                ffy = list(filter(None, re.split(', ', ffy)))
                i = i + 1
                pos_dict[ff[2] + "y"] = np.array(ff[3:6] + ffy[3:64],
                                                 dtype=float)

                dirx = np.cross(np.array(ffz[3:6], dtype=float),
                                np.array(ffy[3:6], dtype=float))
                pos_dict[ff[2] + "x"] = np.concatenate(
                    (np.array(ff[3:6], dtype=float), dirx))

    return pos_dict


def rotate_translate_pos_dict(pos_dict, opm_trans_path, geom_name, subject_dir="",
                              gen12=2):
    import bioelmag.vector_functions as vfun
    import numpy as np
    import mne

    rotation, translation = import_opm_trans(opm_trans_path, geom_name)
    translation = translation / 1000.0

    for key in pos_dict:
        pos_dict[key][0:3] = pos_dict[key][0:3] / 1000.0

        if gen12 == 2:
            # !!!!! be very carefull what comes first.
            pos_dict[key][0] = pos_dict[key][0] - translation[0]
            pos_dict[key][1] = pos_dict[key][1] - translation[1]
            pos_dict[key][2] = pos_dict[key][2] - translation[2]

        pos_dict[key][1], pos_dict[key][2] = vfun.rotate_via_numpy(
            pos_dict[key][1], pos_dict[key][2], np.radians(-rotation[0]))
        pos_dict[key][0], pos_dict[key][2] = vfun.rotate_via_numpy(
            pos_dict[key][0], pos_dict[key][2], np.radians(-rotation[1]))
        pos_dict[key][0], pos_dict[key][1] = vfun.rotate_via_numpy(
            pos_dict[key][0], pos_dict[key][1], np.radians(-rotation[2]))
        pos_dict[key][4], pos_dict[key][5] = vfun.rotate_via_numpy(
            pos_dict[key][4], pos_dict[key][5], np.radians(-rotation[0]))
        pos_dict[key][3], pos_dict[key][5] = vfun.rotate_via_numpy(
            pos_dict[key][3], pos_dict[key][5], np.radians(-rotation[1]))
        pos_dict[key][3], pos_dict[key][4] = vfun.rotate_via_numpy(
            pos_dict[key][3], pos_dict[key][4], np.radians(-rotation[2]))

        if gen12 != 2:
            pos_dict[key][0] = pos_dict[key][0] - translation[0]
            pos_dict[key][1] = pos_dict[key][1] - translation[1]
            pos_dict[key][2] = pos_dict[key][2] - translation[2]

            surf = mne.read_surface(
                subject_dir + geom_name + "/surf/" + "lh.white",
                read_metadata=True)
            pos_dict[key][0:3] = pos_dict[key][0:3] - surf[2]['cras'] / 1000.0

    return pos_dict


def import_opm_trans(fn, name):
    import re
    import numpy as np
    F = open(fn, 'r')
    num_lines = sum(1 for line in open(fn))
    i = 0
    while i < num_lines:
        f = F.readline()
        f = f.replace('"', '')
        f = f.replace(':', ',')
        f = f.replace('\n', '')
        f = f.replace('\t\t', ' ')
        f = f.replace('\t', '')
        if f == name:
            a = F.readline()
            aa = list(filter(None, re.split(" ", a)))
            ad = np.array(aa, dtype=float)
            b = F.readline()
            bb = list(filter(None, re.split(" ", b)))
            bd = np.array(bb, dtype=float)
            i = num_lines
        i += 1
    return ad, bd
