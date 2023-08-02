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
                    pos_occupied += f"ECHO: {all_lines[ii][0]}, {all_lines[ii][1]}, {all_lines[ii][2]}, {str(jj)}\n"
                else:
                    pos_occupied += f"ECHO: {all_lines[ii][0]}, {all_lines[ii][1]}, {all_lines[ii][2]}, {str(0)}\n"
            i += 1
    pos_occupied = pos_occupied[:-1]
    return pos_occupied


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


def read_sensor_positions(meas_dir, block_name, num_sens):
    # full path to the measurement directory
    # the files have to be exported to .csv
    # number of sensors you want to import
    # returns "sens_hold_idx" which corresponds to the
    # holes on OPM sensor holders built at PTB and
    # "con_position" which corresponds to # of channel
    # in con file

    import numpy as np

    filename = f"{meas_dir}Positions/SensorPositions_{block_name}.csv"
    with open(filename) as f:
        sens_pos = np.loadtxt((x.replace(',', "\t") for x in f),
                              delimiter="\t", skiprows=1, max_rows=num_sens, dtype=str)

    sens_hold_idx = sens_pos[:, 1]

    con_position = []
    for i, j in zip(sens_pos[:, 3].astype('int'), sens_pos[:, 4].astype('int')):
        con_position.append(i - 1)
        con_position.append(j - 1)
    con_position = np.array(con_position)
    print(con_position)

    return sens_hold_idx, con_position


def read_sens_info(meas_dir, block_name):
    # full path of the SensorPositions file
    # the files have to be exported to .csv
    # number of sensors you want to import
    import numpy as np
    import sys

    meas_fn = f"{meas_dir}/MeasurementInfos.txt"
    measurements = np.loadtxt(meas_fn, delimiter=";", skiprows=1, dtype=str)

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

    meas_info = {'sens_num': int(measurements[meas_idx, 7]),
                 "gain": float(measurements[meas_idx, 4]),
                 "gen12": int(measurements[meas_idx, 3]),
                 "importgeometry": int(measurements[meas_idx, 5]),
                 "t_delay": float(measurements[meas_idx, 6]),
                 "trigger_ch": int(measurements[meas_idx, 10]),
                 "dataformat": measurements[meas_idx, 9],
                 "bads_idx": bads_idx,
                 "block_name": block_name,
                 "meas_dir": meas_dir,
                 "reorient_idx": reorient_idx
                 }

    sens_hold, con_pos = read_sensor_positions(meas_dir, block_name,
                                               meas_info["sens_num"])
    meas_info["con_pos"] = con_pos
    meas_info["sens_hold"] = sens_hold

    return meas_info


def import_sensor_pos_ori_occupied(sensorholder_path, name, subject_dir, gen12=1):
    import bioelmag.vector_functions as vfun
    import mne
    import numpy as np

    if gen12 == 2:
        sensorholder_file = sensorholder_path + name + '_sensor_pos_ori_occupied_v2.txt'
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


def import_sensor_pos_ori_all(sensorholder_path, name, subject_dir, gen12=1):
    import megtools.vector_functions as vfun
    import mne

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

        surf = mne.read_surface(subject_dir + name + "/surf/" + "lh.white", read_metadata=True)
        holders[:, 0:3] = holders[:, 0:3] - (surf[2]['cras'] / 1000.0)

    return holders
