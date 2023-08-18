def imp_sensor_occupied(fn):
    # STRUCTUR OF SENSOR HOLDERS
    # ECHO:
    #
    import re
    import numpy as np

    F = open(fn, 'r')
    sensors = []
    orientations = []
    num_lines = sum(1 for line in open(fn))

    i = 0
    j = 0
    while i < num_lines:
        f = F.readline()
        f = f.replace('"', '')
        f = f.replace(':', ',')
        f = f.replace('\n', '')
        ff = list(filter(None, re.split(", ", f)))
        if j == 0:
            sensors.append([ff[1], ff[2], ff[3]])
            sensors.append([ff[1], ff[2], ff[3]])
            j = 1
        elif j == 1:
            orientations.append([ff[1], ff[2], ff[3]])
            j = 2
        elif j == 2:
            orientations.append([ff[1], ff[2], ff[3]])
            j = 0
        i = i + 1

    holders = [a + b for a, b in zip(sensors, orientations)]

    return np.array(holders).astype(float)


def create_sens_pos_occupied(fn, sel_list):
    import re

    with open(fn, "r") as F:
        num_lines = len(open(fn).readlines())

        pos_occupied = ""

        all_lines = []
        sens_numb = []

        i = 0
        while i < num_lines:
            f = F.readline()
            ff = f.replace('"', '')
            ff = ff.replace(':', ',')
            ff = ff.replace('\n', '')
            ff = list(filter(None, re.split(', ', ff)))
            all_lines.append(ff[3:])
            sens_numb.append(ff[2])
            i = i + 1

    # print(all_lines)
    i = 0
    for jj in sel_list:
        for ii, kk in enumerate(sens_numb):
            if int(kk) == int(jj):
                if i % 3 == 0:
                    pos_occupied += f"ECHO: {all_lines[ii][0]},{all_lines[ii][1]}, {all_lines[ii][2]}, {str(jj)}\n"
                else:
                    pos_occupied += f"ECHO: {all_lines[ii][0]}, {all_lines[ii][1]}, {all_lines[ii][2]}, {str(0)}\n"
            i += 1
    pos_occupied = pos_occupied[:-1]
    return pos_occupied


def create_sens_pos_dict(fn):
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

            if int(ff[4]) > 0:
                f = F.readline()
                ffz = f.replace('"', '')
                ffz = ffz.replace(':', ',')
                ffz = ffz.replace('\n', '')
                ffz = list(filter(None, re.split(', ', ffz)))
                i = i + 1
                pos_dict[ff[4]+"z"] = np.array(ff[1:4] + ffz[1:4], dtype=float)

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
    return pos_dict


def find_events_aef(values, triger_range):
    import numpy as np

    values = -values
    values1 = np.zeros(len(values))

    for j in range(0, len(values)):
        if triger_range[0] < values[j] < triger_range[1]:
            values1[j] = 1

    spikes = []
    spike_count = 0
    for j in range(0, len(values1)):
        if values1[j] > 0 and spike_count < 3:
            spike_count += 1
            if spike_count == 3:
                spikes.append([j - 2, 0, 1])
        if values1[j] == 0 and spike_count != 0:
            spike_count = 0

    return spikes


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
                sens_holdpos.append(file_sens_hold_idx[i])
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


def read_sensor_positions(sens_pos_fn, num_sens=None):
    # full path to the measurement directory
    # the files have to be exported to .csv
    # number of sensors you want to import
    # returns "sens_hold_idx" which corresponds to the
    # holes on OPM sensor holders built at PTB and
    # "con_position" which corresponds to # of channel
    # in con file

    import numpy as np

    # filename = f"{meas_dir}Positions/SensorPositions_{block_name}.csv"
    with open(sens_pos_fn) as f:
        # sens_pos = np.loadtxt((x.replace(',', "\t") for x in f),
        sens_pos = np.loadtxt((x for x in f),
                              delimiter="\t", max_rows=num_sens,
                              dtype=str, skiprows=1)

    con_position = []
    for i, j in zip(sens_pos[:, 3].astype('int'),
                    sens_pos[:, 4].astype('int')):
        con_position.append(i - 1)
        con_position.append(j - 1)
    con_position = np.array(con_position)
    # print(con_position)

    return con_position


def read_meas_info(meas_info_fn, block_name):
    # full path of the SensorPositions file
    # the files have to be exported to .csv
    # number of sensors you want to import
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
            bads_idx.append(int(i))

    meas_info = {"gain": float(measurements[meas_idx, 4]),
                 # 'sens_num': int(measurements[meas_idx, 7]),
                 "gen12": int(measurements[meas_idx, 3]),
                 "importgeometry": int(measurements[meas_idx, 5]),
                 "t_delay": float(measurements[meas_idx, 6]),
                 # "trigger_ch": int(measurements[meas_idx, 10]),
                 "dataformat": measurements[meas_idx, 9],
                 "bads_idx": bads_idx,
                 "block_name": block_name,
                 # "meas_dir": meas_dir,
                 "reorient_idx": reorient_idx
                 }

    # sens_hold, con_pos = read_sensor_positions(meas_dir, block_name,
    #                                            meas_info["sens_num"])
    # meas_info["con_pos"] = con_pos
    # meas_info["sens_hold"] = sens_hold

    return meas_info


def import_sensor_pos_ori_selected(sensorholder_num, sensorholder_direction,
                                   sensorholder_path, name, subject_dir,
                                   gen12=1):
    import bioelmag.vector_functions as vfun
    import mne
    import numpy as np

    if gen12 == 2:
        sensorholder_file = sensorholder_path + name
        + '_sensor_pos_ori_occupied_v2.txt'
        opm_trans_path = sensorholder_path + "opm_trans_v2.txt"
    else:
        sensorholder_file = sensorholder_path + name
        + '_sensor_pos_ori_occupied.txt'
        opm_trans_path = sensorholder_path + "opm_trans.txt"

    rotation, translation = import_opm_trans(opm_trans_path, name[0:4])
    translation = translation / 1000.0

    holders = imp_sensor_occupied(sensorholder_file)
    holders[:, 0:3] = holders[:, 0:3] / 1000.0

    if gen12 == 2:
        # !!!!! be very carefull what comes first.
        holders[:, 0] = holders[:, 0] - translation[0]
        holders[:, 1] = holders[:, 1] - translation[1]
        holders[:, 2] = holders[:, 2] - translation[2]

    holders[:, 1], holders[:, 2] = vfun.rotate_via_numpy(
        holders[:, 1], holders[:, 2], np.radians(-rotation[0]))
    holders[:, 0], holders[:, 2] = vfun.rotate_via_numpy(
        holders[:, 0], holders[:, 2], np.radians(-rotation[1]))
    holders[:, 0], holders[:, 1] = vfun.rotate_via_numpy(
        holders[:, 0], holders[:, 1], np.radians(-rotation[2]))
    holders[:, 4], holders[:, 5] = vfun.rotate_via_numpy(
        holders[:, 4], holders[:, 5], np.radians(-rotation[0]))
    holders[:, 3], holders[:, 5] = vfun.rotate_via_numpy(
        holders[:, 3], holders[:, 5], np.radians(-rotation[1]))
    holders[:, 3], holders[:, 4] = vfun.rotate_via_numpy(
        holders[:, 3], holders[:, 4], np.radians(-rotation[2]))

    if gen12 != 2:
        holders[:, 0] = holders[:, 0] - translation[0]
        holders[:, 1] = holders[:, 1] - translation[1]
        holders[:, 2] = holders[:, 2] - translation[2]

        surf = mne.read_surface(
            subject_dir + name[0:4] + "/surf/" + "lh.white",
            read_metadata=True)
        holders[:, 0:3] = holders[:, 0:3] - (surf[2]['cras'] / 1000.0)

    return holders


def import_sensor_pos_ori_occupied(sensorholder_path, name, subject_dir, gen12=1):
    import bioelmag.vector_functions as vfun
    import mne
    import numpy as np

    if gen12 == 2:
        sensorholder_file = sensorholder_path + name + '_v2_fNIRS_oMEG_opm_pos_ori_occupied.txt'
        # sensorholder_file = sensorholder_path + name + '_sensor_pos_ori_occupied_v2.txt'
        opm_trans_path = sensorholder_path + "opm_trans_v2.txt"
    else:
        sensorholder_file = sensorholder_path + name + '_sensor_pos_ori_occupied.txt'
        opm_trans_path = sensorholder_path + "opm_trans.txt"

    rotation, translation = import_opm_trans(opm_trans_path, name[0:4])
    translation = translation / 1000.0

    holders = imp_sensor_occupied(sensorholder_file)
    holders[:, 0:3] = holders[:, 0:3] / 1000.0

    if gen12 == 2:
        # !!!!! be very carefull what comes first.
        holders[:, 0] = holders[:, 0] - translation[0]
        holders[:, 1] = holders[:, 1] - translation[1]
        holders[:, 2] = holders[:, 2] - translation[2]

    holders[:, 1], holders[:, 2] = vfun.rotate_via_numpy(holders[:, 1], holders[:, 2], np.radians(-rotation[0]))
    holders[:, 0], holders[:, 2] = vfun.rotate_via_numpy(holders[:, 0], holders[:, 2], np.radians(-rotation[1]))
    holders[:, 0], holders[:, 1] = vfun.rotate_via_numpy(holders[:, 0], holders[:, 1], np.radians(-rotation[2]))
    holders[:, 4], holders[:, 5] = vfun.rotate_via_numpy(holders[:, 4], holders[:, 5], np.radians(-rotation[0]))
    holders[:, 3], holders[:, 5] = vfun.rotate_via_numpy(holders[:, 3], holders[:, 5], np.radians(-rotation[1]))
    holders[:, 3], holders[:, 4] = vfun.rotate_via_numpy(holders[:, 3], holders[:, 4], np.radians(-rotation[2]))

    if gen12 != 2:
        holders[:, 0] = holders[:, 0] - translation[0]
        holders[:, 1] = holders[:, 1] - translation[1]
        holders[:, 2] = holders[:, 2] - translation[2]

        surf = mne.read_surface(subject_dir + name[0:4] + "/surf/" + "lh.white", read_metadata=True)
        holders[:, 0:3] = holders[:, 0:3] - (surf[2]['cras'] / 1000.0)

    return holders


def create_occupied_file_meas_info(meas_info):
    if int(meas_info["gen12"]) == 2:
        gen12_s = "_v2"
    else:
        gen12_s = ""

    sensor_all_fn = f"{meas_info['meas_dir']}Positions/Sensorholders/{meas_info['block_name'][0:4]}" \
                    f"_sensor_pos_ori_all{gen12_s}.txt"
    sensor_occupied_fn = f"{meas_info['meas_dir']}Positions/Sensorholders/{meas_info['block_name']}" \
                         f"_sensor_pos_ori_occupied{gen12_s}.txt"

    sensorholder = create_sens_pos_occupied(sensor_all_fn, meas_info["sens_hold"])

    with open(sensor_occupied_fn, "w") as f:
        print(sensorholder, file=f)

    return


def create_mne_raw(sens_info, meas_info, data_path, sens_hol_path="", subject_dir="", opm_trans_path="", geom_name=""):
    import mne
    import numpy as np
    import bioelmag.vector_functions as vfun

    if meas_info["dataformat"] == "flt":
        header_name = data_path[0]
        value_name = data_path[1]
        data, sfreq = import_flt_hdr(header_name, value_name)

    info = create_mne_info_opm(sens_info, sfreq)

    if meas_info["importgeometry"] == 1:
        pos_dict = create_sens_pos_dict(sens_hol_path)
        pos_dict = rotate_translate_pos_dict(
            pos_dict, opm_trans_path, name=geom_name, subject_dir=subject_dir, gen12=meas_info["gen12"])

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
                sens_hold_name = sens_info['sens_holdpos'][sens_hold_idx] + j[-1]
                holders = pos_dict[sens_hold_name]
                rot_mat = vfun.create_rot_matrix(holders[3:6], unit_v)
                info['chs'][i]['loc'] = np.array(
                    [holders[0], holders[1], holders[2], rot_mat[0, 0], rot_mat[0, 1],
                     rot_mat[0, 2], rot_mat[1, 0], rot_mat[1, 1], rot_mat[1, 2], rot_mat[2, 0],
                     rot_mat[2, 1], rot_mat[2, 2]])
            data[sens_info['sens_datapos'][sens_hold_idx]] = data[sens_info['sens_datapos'][sens_hold_idx]] * (10 ** -9)

    raw = mne.io.RawArray(data[sens_info['sens_datapos']], info)

    return raw


def rotate_translate_pos_dict(pos_dict, opm_trans_path, name, subject_dir="", gen12=2):
    import bioelmag.vector_functions as vfun
    import numpy as np
    import mne

    rotation, translation = import_opm_trans(opm_trans_path, name[0:4])
    translation = translation / 1000.0

    for key in pos_dict:
        pos_dict[key][0:3] = pos_dict[key][0:3] / 1000.0

        if gen12 == 2:
            # !!!!! be very carefull what comes first.
            pos_dict[key][0] = pos_dict[key][0] - translation[0]
            pos_dict[key][1] = pos_dict[key][1] - translation[1]
            pos_dict[key][2] = pos_dict[key][2] - translation[2]

        pos_dict[key][1], pos_dict[key][2] = vfun.rotate_via_numpy(pos_dict[key][1], pos_dict[key][2], np.radians(-rotation[0]))
        pos_dict[key][0], pos_dict[key][2] = vfun.rotate_via_numpy(pos_dict[key][0], pos_dict[key][2], np.radians(-rotation[1]))
        pos_dict[key][0], pos_dict[key][1] = vfun.rotate_via_numpy(pos_dict[key][0], pos_dict[key][1], np.radians(-rotation[2]))
        pos_dict[key][4], pos_dict[key][5] = vfun.rotate_via_numpy(pos_dict[key][4], pos_dict[key][5], np.radians(-rotation[0]))
        pos_dict[key][3], pos_dict[key][5] = vfun.rotate_via_numpy(pos_dict[key][3], pos_dict[key][5], np.radians(-rotation[1]))
        pos_dict[key][3], pos_dict[key][4] = vfun.rotate_via_numpy(pos_dict[key][3], pos_dict[key][4], np.radians(-rotation[2]))

        if gen12 != 2:
            pos_dict[key][0] = pos_dict[key][0] - translation[0]
            pos_dict[key][1] = pos_dict[key][1] - translation[1]
            pos_dict[key][2] = pos_dict[key][2] - translation[2]

            surf = mne.read_surface(
                subject_dir + name[0:4] + "/surf/" + "lh.white",
                read_metadata=True)
            pos_dict[key][0:3] = pos_dict[key][0:3] - (surf[2]['cras'] / 1000.0)

    return pos_dict


def import_flt_hdr(header_name, value_name):
    import megtools.pyread_biosig as pbio
    import numpy as np

    data = np.transpose(pbio.imp_bin_data(value_name, header_name,
                                          encoding="ISO-8859-1"))
    data = data * (10 ** -6)
    sfreq = 1/(pbio.imp_samp_freq(header_name, encoding="ISO-8859-1"))

    return data, sfreq


def create_mne_info(meas_info, sfreq, subject_dir):
    import mne
    import numpy as np
    import bioelmag.vector_functions as vfun

    ch_types = len(meas_info["con_pos"]) * ["mag"]

    ch_names = []
    for i in range(len(meas_info["con_pos"])):
        if i % 2 == 0:
            ch_names.append("rad" + str(i))
        else:
            ch_names.append("tan" + str(i))

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    unit_v = np.array([0.0, 0.0, 1.0])
    for i in range(len(meas_info["con_pos"])):
        info['chs'][i]['coil_type'] = 9999
        info['chs'][i]['scanno'] = i + 1
        info['chs'][i]['logno'] = i + 1
        info['chs'][i]['kind'] = 1
        info['chs'][i]['range'] = 1.0
        info['chs'][i]['cal'] = 3.7000000285836165e-10
        info['chs'][i]['unit'] = 112
        if meas_info["importgeometry"] == 1:
            sens_hol_path = meas_info["meas_dir"] + 'Positions/Sensorholders/'
            holders = import_sensor_pos_ori_occupied(sensorholder_path=sens_hol_path, name=meas_info["block_name"],
                                                     subject_dir=subject_dir, gen12=meas_info["gen12"])
            rot_mat = vfun.create_rot_matrix(holders[i, 3:6], unit_v)
            info['chs'][i]['loc'] = np.array([holders[i, 0], holders[i, 1], holders[i, 2], rot_mat[0, 0], rot_mat[0, 1],
                                              rot_mat[0, 2], rot_mat[1, 0], rot_mat[1, 1], rot_mat[1, 2], rot_mat[2, 0],
                                              rot_mat[2, 1], rot_mat[2, 2]])
    return info


def create_mne_info_opm(sens_info, sfreq):
    import mne

    ch_types = sens_info["sens_type"]
    ch_names = sens_info["sens_names"]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    return info


def import_sensor_pos_ori_all(sensorholder_path, name, subject_dir, gen12=1):
    import megtools.vector_functions as vfun
    import megtools.pymeg_biosig as pbio
    import mne
    import numpy as np

    if gen12 == 2:
        sensorholder_file = sensorholder_path + name + '_sensor_pos_ori_all_v2.txt'
        opm_trans_path = sensorholder_path + "opm_trans_v2.txt"
    else:
        sensorholder_file = sensorholder_path + name + '_sensor_pos_ori_all.txt'
        opm_trans_path = sensorholder_path + "opm_trans.txt"

    rotation, translation = pbio.import_opm_trans(opm_trans_path, name)
    translation = translation / 1000.0

    holders = pbio.imp_sensor_holders(sensorholder_file)
    holders[:, 0:3] = holders[:, 0:3] / 1000.0

    if gen12 == 2:
        # !!!!! be very carefull what comes first.
        holders[:, 0] = holders[:, 0] - translation[0]
        holders[:, 1] = holders[:, 1] - translation[1]
        holders[:, 2] = holders[:, 2] - translation[2]

    holders[:, 1], holders[:, 2] = vfun.rotate_via_numpy(
        holders[:, 1], holders[:, 2], np.radians(-rotation[0]))
    holders[:, 0], holders[:, 2] = vfun.rotate_via_numpy(
        holders[:, 0], holders[:, 2], np.radians(-rotation[1]))
    holders[:, 0], holders[:, 1] = vfun.rotate_via_numpy(
        holders[:, 0], holders[:, 1], np.radians(-rotation[2]))
    holders[:, 4], holders[:, 5] = vfun.rotate_via_numpy(
        holders[:, 4], holders[:, 5], np.radians(-rotation[0]))
    holders[:, 3], holders[:, 5] = vfun.rotate_via_numpy(
        holders[:, 3], holders[:, 5], np.radians(-rotation[1]))
    holders[:, 3], holders[:, 4] = vfun.rotate_via_numpy(
        holders[:, 3], holders[:, 4], np.radians(-rotation[2]))

    if gen12 != 2:
        holders[:, 0] = holders[:, 0] - translation[0]
        holders[:, 1] = holders[:, 1] - translation[1]
        holders[:, 2] = holders[:, 2] - translation[2]

        surf = mne.read_surface(
            subject_dir + name + "/surf/" + "lh.white", read_metadata=True)
        holders[:, 0:3] = holders[:, 0:3] - (surf[2]['cras'] / 1000.0)

    return holders
