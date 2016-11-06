import os
import struct
from PIL import Image


def read_record_ETL8B2(f):
    s = f.read(512)
    r = struct.unpack('>2H4s504s', s)
    i1 = Image.frombytes('1', (64, 63), r[3], 'raw')
    return r + (i1,)

dstdir = 'imgs'
filename = 'ETL8B/ETL8B2C1'
small_labels = [29, 61, 63, 65]

if not os.path.exists(dstdir):
    os.mkdir(dstdir)

idx = 0

with open(filename, 'rb') as f:
    for id_category in range(0, 75):
        # 小さい"つやゆよ"はスキップする
        if id_category in small_labels:
            continue

        subdir = "{:02d}".format(idx)
        if not os.path.exists(os.path.join(dstdir, subdir)):
            os.mkdir(os.path.join(dstdir, subdir))

        f.seek((id_category * 160 + 1) * 512)
        for i in range(160):
            new_img = Image.new('1', (64, 64))
            r = read_record_ETL8B2(f)
            new_img.paste(r[-1], (0, 0))

            iI = Image.eval(new_img, lambda x: not x)
            fn = 'ETL8B2_{:03d}_{:03d}.png'.format(idx, i)
            fn = os.path.join(dstdir, subdir, fn)
            iI.save(fn, 'PNG')

        idx += 1
