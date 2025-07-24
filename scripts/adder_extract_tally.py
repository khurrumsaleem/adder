#!/usr/bin/env python3
import argparse
from pathlib import Path
from collections import defaultdict, OrderedDict

import h5py
import numpy as np

import adder


def get_data(h5_file, tally_id):
    h5_in = h5py.File(h5_file, "r")

    # We need to process the case_#/operation_#/step_# trees and get the data
    # And figure out the right ordering
    case_ids = []
    op_ids = []
    step_ids = []
    grp_names = []

    # Get the sorted order of the cases
    case_ids = sorted([int(k[len("case_"):])
                        for k in h5_in.keys() if k.startswith("case_")])
    # Now iterate through in this sorted order
    for case_id in case_ids:
        case_grp = h5_in["case_" + str(case_id)]

        # Get the sorted order of the operations
        op_ids = sorted([int(k[len("operation_"):])
                         for k in case_grp.keys()
                         if k.startswith("operation_")])
        # Now iterate through in this sorted order
        for op_id in op_ids:
            op_grp = case_grp["operation_" + str(op_id)]
            # Get the sorted order of the steps
            step_ids = sorted([int(k[len("step_"):])
                               for k in op_grp.keys()
                               if k.startswith("step_")])

            # Now we can create the group names
            for step_id in step_ids:
                grp_name = "case_{}/operation_{}/step_{}".format(case_id,
                                                                 op_id,
                                                                 step_id)
                grp_names.append(grp_name)

    # Initialize data and progress through the order
    case_labels = []
    operation_labels = []
    step_idxs = []
    times = []
    tally_id_l = []
    inp_spec = []
    type = []
    material_names = []
    universe_names = []
    tally_matrix = []
    tally_matrix_err = []
    facet_ids =[]

    for grp_name in grp_names:
        grp = h5_in[grp_name]
        tallies_grp = grp['user_tallies']
        if tally_id in tallies_grp.keys():
          tally = adder.Tally.from_hdf5(tallies_grp, tally_id)
          case_labels.append(grp.attrs["case_label"].decode())
          operation_labels.append(grp.attrs["operation_label"].decode())
          step_idxs.append(int(grp.attrs["step_idx"]))
          times.append(float(grp.attrs["time"]))
          tally_id_l.append(tally.id)
          type.append(tally.type)
          inp_spec.append(tally.tally_block)
          material_names.append(tally.material_names)
          universe_names.append(tally.universe_names)
          tally_matrix.append(tally.tally_matrix)
          tally_matrix_err.append(tally.tally_matrix_err)
          facet_ids.append(tally.facet_ids)


    results = OrderedDict()
    results['case_label'] = case_labels
    results['operation_label'] = operation_labels
    results['step_idx'] = step_idxs
    results['times'] = times
    results['id'] = tally_id_l
    results['input specification'] = inp_spec
    results['type'] = type
    results['material names'] = material_names
    results['universe names'] = universe_names
    results['facet ids'] = facet_ids
    results['tally matrix'] = tally_matrix
    results['tally matrix err'] = tally_matrix_err

    return results


def make_csv(fname, results):
    # Print header
    with open(fname, "w") as file:
        # Write the header
        header = "# data \\ time [d]\n"
        file.write(header)

        # First print labels
        for key in ["case_label", "operation_label", "step_idx"]:
            labels = results[key]
            label_row = "{}, ".format(key)
            for l in labels:
                label_row += "{}, ".format(l)
            file.write(label_row + "\n")

        # Then the times
        row = "times, "
        values = results["times"]
        for v in values:
            row += "{:1.6E}, ".format(v)
        file.write(row + "\n")

        # Everything else aside from isotopics and what was already printed
        # Then the times
        row = "type, "
        values = results["type"]
        for v in values:
            row += "{}, ".format(v)
        file.write(row + "\n")

        for key in ["material names", "universe names"]:
            values = results[key]

            label_row = "{}, ".format(key)
            n_col = len(values)
            n_row = len(values[0])
            max_length = max(len(lst) for lst in values)

            adjusted_data = [
                lst if len(lst) == max_length else ["no "+key[-6]+"s"] * max_length
                for lst in values
            ]
            matrix = np.array(adjusted_data)
            for l in range(n_row):
                for k in range(n_col):
                    label_row += "{},".format(matrix[k, l])
                if l != n_row - 1:
                    label_row += "\n,"
            file.write(label_row + "\n")

        row = "facet ids, "
        values = results["facet ids"]
        n_col = len(values)
        n_row = len(values[0])
        # if cases with tallies not present in geom, [0] default
        # must be adjusted with the length for cases present in geom.
        max_length = max(len(lst) for lst in values)

        adjusted_data = [
            lst if len(lst) == max_length else [np.int32(0)] * max_length
            for lst in values
        ]
        matrix = np.array(adjusted_data)
        for l in range(n_row):
            for k in range(n_col):
                row += "{:d},".format(matrix[k, l])
            if l != n_row - 1:
                row += "\n,"
        file.write(row + "\n")


        row = "tally results, "
        values = results["tally matrix"]
        n_col = len(values)
        # found the maximum dimension
        dim_flatten = max(np.shape(values[i].flatten())[0] for i in range(n_col))
        # Flatten and adjust list of arrays in once cycle
        values = [values[i].flatten() if np.shape(values[i].flatten())[0] == dim_flatten
                  else np.zeros(dim_flatten, dtype=values[i].dtype) for i in range(n_col)]

        n_row = len(values[0])
        matrix = np.array(values)
        for l in range(n_row):
            for k in range(n_col):
                row += "{:1.6E},".format(matrix[k, l])
            row += "\n,"
        file.write(row + "\n")

        row = "tally results err, "
        values = results["tally matrix err"]
        n_col = len(values)
        # found the maximum dimension
        dim_flatten = max(np.shape(values[i].flatten())[0] for i in range(n_col))
        # Flatten and adjust list of arrays in once cycle
        values = [values[i].flatten() if np.shape(values[i].flatten())[0] == dim_flatten
                  else np.zeros(dim_flatten, dtype=values[i].dtype) for i in range(n_col)]
        n_row = len(values[0])
        matrix = np.array(values)
        for l in range(n_row):
            for k in range(n_col):
                row += "{:1.6E},".format(matrix[k, l])
            row += "\n,"
        file.write(row + "\n")


description = """
This script can be used to parse an ADDER HDF5 output file to create a
comma-separated-value file containing the time-valued tally results with errors, in addition
to other tally attributes

Example usage:
adder_extract_tally.py in.h5 out.csv mat_name

"""


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass


parser = argparse.ArgumentParser(
    description=description,
    formatter_class=CustomFormatter
)
parser.add_argument('results_file', action='store', type=str,
                    help='The HDF5 file to read from')
parser.add_argument('csv_file', action='store', type=str,
                    help='The CSV file to create and write to')
parser.add_argument('tally_id', action='store', type=str,
                    help='The tally id to process')
args = parser.parse_args()

# Get the results file
if not Path(args.results_file).exists():
    msg = "{} does not exist!".format(args.results_file)
    raise ValueError(msg)
results_file = args.results_file

# Get and create the destination path if needed
csv_file = Path(args.csv_file)
csv_path = csv_file.parent
csv_path.mkdir(parents=True, exist_ok=True)

# Process the library
print("Processing ADDER Output: {}".format(results_file))

# Get the data
results = get_data(results_file, args.tally_id)

# Write the data
make_csv(str(csv_file), results)
