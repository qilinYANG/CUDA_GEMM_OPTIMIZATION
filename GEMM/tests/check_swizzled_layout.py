"""CPU model of swizzle, ldmatrix register ownership and MMA output mapping.
Not a CUDA execution test. Run: python3 GEMM/tests/check_swizzled_layout.py
"""
from pathlib import Path
import random

source = (Path(__file__).resolve().parents[1] / 'gemm_tensor_core.cu').read_text()
assert 'return r * stride + (((c / 8) ^ (r & 7)) * 8) + c % 8;' in source

def offset(r, c, stride):
    return r * stride + ((c // 8) ^ (r & 7)) * 8 + c % 8

# Bijection, vector contiguity and eight-row ldmatrix bank groups.
for stride in (64, 128):
    addresses = [offset(r, c, stride) for r in range(128) for c in range(stride)]
    assert sorted(addresses) == list(range(128 * stride))
    for r in range(128):
        for c in range(0, stride, 8):
            start = offset(r, c, stride)
            assert start * 2 % 16 == 0
            assert [offset(r, c + j, stride) for j in range(8)] == list(range(start, start + 8))
    for r0 in range(0, 128, 8):
        for c in range(0, stride, 8):
            banks = [(offset(r, c, stride) // 2 + word) % 32
                     for r in range(r0, r0 + 8) for word in range(4)]
            assert sorted(banks) == list(range(32))
    # Eight copy lanes cover a complete aligned 128-byte panel (in permutation).
    for r in range(128):
        for c0 in range(0, stride, 64):
            panel = sorted(offset(r, c, stride) for c in range(c0, c0 + 64))
            assert panel == list(range(panel[0], panel[0] + 64))
            assert panel[0] * 2 % 128 == 0

# Emulate each 8x8 ldmatrix result using the row addresses supplied by lanes.
# Transpose changes the matrix element assignment, not the source layout.
def ldmatrix(shared, stride, row, col, lane, transpose=False):
    regs = []
    for matrix in range(4):
        provider = matrix * 8
        r0 = row + provider % 16
        c0 = col + (provider // 16) * 8
        r, c = lane // 4, (lane % 4) * 2
        vals = []
        for e in range(2):
            rr, cc = (c + e, r) if transpose else (r, c + e)
            vals.append(shared[offset(r0 + rr, c0 + cc, stride)])
        regs.append(vals)
    return regs

rng = random.Random(42)
for bk in (16, 48, 64):
    stride = (bk + 63) // 64 * 64
    aa = [[rng.randrange(-3, 4) for _ in range(bk)] for _ in range(32)]
    bb = [[rng.randrange(-3, 4) for _ in range(32)] for _ in range(bk)]
    sa, sb = [None] * (32 * stride), [None] * (bk * 128)
    for r in range(32):
        for c in range(bk): sa[offset(r, c, stride)] = aa[r][c]
    for r in range(bk):
        for c in range(32): sb[offset(r, c, 128)] = bb[r][c]
    out = [[0] * 32 for _ in range(32)]
    for kk in range(0, bk, 16):
        for mi in range(2):
            # Recover operand matrices from documented MMA register ownership.
            ar = [[None] * 16 for _ in range(16)]
            for lane in range(32):
                regs = ldmatrix(sa, stride, mi * 16, kk, lane)
                for reg in range(4):
                    for e in range(2):
                        r = lane // 4 + (reg % 2) * 8
                        c = (lane % 4) * 2 + (reg // 2) * 8 + e
                        ar[r][c] = regs[reg][e]
            assert ar == [row[kk:kk+16] for row in aa[mi*16:mi*16+16]]
            for nj in range(4):
                br = [[None] * 8 for _ in range(16)]
                for lane in range(32):
                    regs = ldmatrix(sb, 128, kk, (nj // 2) * 16, lane, True)
                    for reg in range(2):
                        for e in range(2):
                            r = (lane % 4) * 2 + reg * 8 + e
                            c = lane // 4
                            br[r][c] = regs[(nj % 2)*2+reg][e]
                assert br == [row[nj*8:nj*8+8] for row in bb[kk:kk+16]]
                for lane in range(32):
                    for reg in range(4):
                        r = lane // 4 + (reg // 2) * 8
                        c = (lane % 4) * 2 + reg % 2
                        out[mi*16+r][nj*8+c] += sum(ar[r][k] * br[k][c] for k in range(16))
    ref = [[sum(aa[r][k]*bb[k][c] for k in range(bk)) for c in range(32)] for r in range(32)]
    assert out == ref

# Grid/warp/lane output ownership for partial blocks and partial warp tiles.
for m, n in ((16,16), (32,32), (80,144), (128,128), (144,272)):
    seen = set()
    for by in range((m+127)//128):
        for bx in range((n+127)//128):
            for warp in range(16):
                for lane in range(32):
                    for i in range(2):
                        for j in range(4):
                            for h in range(2):
                                r = by*128 + (warp//4)*32 + i*16 + lane//4 + h*8
                                c = bx*128 + (warp%4)*32 + j*8 + (lane%4)*2
                                if r < m and c+1 < n:
                                    assert (r*n+c)*4 % 8 == 0
                                    for e in range(2):
                                        assert (r,c+e) not in seen
                                        seen.add((r,c+e))
    assert len(seen) == m*n
print('PASS: swizzle bijection, copy alignment, modeled bank groups, ldmatrix/MMA register mapping, numeric tile results and boundary output ownership')
