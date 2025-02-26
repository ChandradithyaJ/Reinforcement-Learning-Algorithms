import pathlib
import numpy as np
import nnabla_rl.utils.files as files
from nnabla_rl.writer import Writer
import pickle

class FileWriter(Writer):
    def __init__(self, outdir, file_prefix, fmt="%.3f"):
        super(FileWriter, self).__init__()
        if isinstance(outdir, str):
            outdir = pathlib.Path(outdir)
        self._outdir = outdir
        files.create_dir_if_not_exist(outdir=outdir)
        self._file_prefix = file_prefix
        self._fmt = fmt
        self._rewards = [] 

    def write_scalar(self, iteration_num, scalar):
        outfile = self._outdir / (self._file_prefix + "_scalar.tsv")

        len_scalar = len(scalar.values())
        out_scalar = {}
        out_scalar["iteration"] = iteration_num
        out_scalar.update(scalar)

        self._create_file_if_not_exists(outfile, out_scalar.keys())

        with open(outfile, "a") as f:
            np.savetxt(f, [list(out_scalar.values())], fmt=["%i"] + [self._fmt] * len_scalar, delimiter="\t")

    def write_histogram(self, iteration_num, histogram):
        outfile = self._outdir / (self._file_prefix + "_histogram.pkl")

        self._rewards.append(np.mean(histogram['returns']))
        pickle.dump(self._rewards, open(outfile, 'wb'))

    def write_image(self, iteration_num, image):
        pass

    def _create_file_if_not_exists(self, outfile, header_keys):
        if not outfile.exists():
            outfile.touch()
            self._write_file_header(outfile, header_keys)

    def _write_file_header(self, filepath, keys):
        with open(filepath, "w+") as f:
            np.savetxt(f, [list(keys)], fmt="%s", delimiter="\t")