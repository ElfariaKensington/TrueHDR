"""Parse ArtCNN mpv compute shaders (GLSL/ArtCNN_*.glsl, Artoriuz/ArtCNN) into spec-family layers.

Each '//!DESC ... (Conv2D...)' pass is one 3x3 (or kxk) conv. Feature maps with >4 channels are
stored by mpv as 2x2-packed RGBA textures; the store offset of resultK (output_base + ivec2(a, b))
identifies the channel group, and the consuming pass's texelFetch offset maps it back.
A pass whose input is the sum of two saved textures encodes the long skip: the spec family
expresses it as `skip` on the layer that produced the first texture.

GLSL M4(a0..a15) is column-major: column j holds the weights from input channel j, so
w[4K+i][4g+j][dy][dx] = a[4j+i] for 'resultK += M4(a) * inp_g_dx_dy'.
Returns layers in PyTorch layout w[cout][cin][k][k] (float32), plus the d2s check.
"""
import re
import numpy as np

NUM = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?'


def _nums(s):
    return [float(x) for x in re.findall(NUM, s)]


def parse_artcnn(path):
    src = open(path).read()
    passes = re.split(r'(?=^//!DESC )', src, flags=re.M)[1:]
    convs = []
    saw_d2s = False
    for p in passes:
        desc = p.splitlines()[0][len('//!DESC '):]
        binds = re.findall(r'^//!BIND (\S+)', p, flags=re.M)
        save = re.findall(r'^//!SAVE (\S+)', p, flags=re.M)
        if 'Depth-To-Space' in desc:
            # verify the PixelShuffle ordering: channel index = i0.y * 2 + i0.x
            assert re.search(r'\[i0\.y \* 2 \+ i0\.x\]', p), 'unexpected depth-to-space ordering'
            saw_d2s = True
            continue
        if 'Conv2D' not in desc:
            raise ValueError('unknown pass: ' + desc)
        k = int(re.search(r'ksize = ivec2\((\d+), (\d+)\)', p).group(1))
        m = re.search(r'shared (F|V4) inp\[(\d+)\]', p)
        scalar_in = m.group(1) == 'F'
        n_in_groups = int(m.group(2))
        # input group -> (texture list, offset)
        in_groups = {}
        for g, rhs in re.findall(r'inp\[(\d+)\]\[y\]\[x\] = (?:V4|F)\((.*)\);', p):
            texs = re.findall(r'(\w+)_mul \* texelFetch\(\1_raw, input_base \+ ivec2\((\d+), (\d+)\)', rhs)
            in_groups[int(g)] = [(t, (int(a), int(b))) for t, a, b in texs]
        assert len(in_groups) == n_in_groups
        results = {}
        for r, body in re.findall(r'V4 result(\d+) = V4\(([^)]*)\);', p):
            results[int(r)] = _nums(body)
        n_out_groups = len(results)
        cin = 1 if scalar_in else 4 * n_in_groups
        cout = 4 * n_out_groups
        w = np.zeros((cout, cin, k, k), dtype=np.float64)
        b = np.zeros(cout, dtype=np.float64)
        for r, bb in results.items():
            assert len(bb) == 4
            b[4 * r:4 * r + 4] = bb
        cnt = 0
        for r, kind, body, g, dx, dy in re.findall(
                r'result(\d+) \+= (M4|V4)\(([^)]*)\) \* inp_(\d+)_(\d+)_(\d+);', p):
            r, g, dx, dy = int(r), int(g), int(dx), int(dy)
            a = _nums(body)
            if kind == 'V4':
                assert scalar_in and len(a) == 4
                w[4 * r:4 * r + 4, 0, dy, dx] += a
            else:
                assert len(a) == 16
                for j in range(4):
                    for i in range(4):
                        w[4 * r + i, 4 * g + j, dy, dx] += a[4 * j + i]
            cnt += 1
        expect = n_out_groups * k * k * (1 if scalar_in else n_in_groups)
        assert cnt == expect, (desc, cnt, expect)
        # activation from the store expression
        stores = re.findall(r'imageStore\(out_image, output_base \+ ivec2\((\d+), (\d+)\), (.*)\);', p)
        relu = [('max(result' in s) for _, _, s in stores]
        assert all(relu) or not any(relu)
        out_offsets = {}
        for a, bb, s in stores:
            rr = int(re.search(r'result(\d+)', s).group(1))
            out_offsets[(int(a), int(bb))] = rr
        if not stores:  # single 4-channel output written once
            st = re.findall(r'imageStore\(out_image, output_base, (.*)\);', p)
            relu = ['max(result' in s for s in st]
            out_offsets[(0, 0)] = 0
        convs.append({'desc': desc, 'binds': binds, 'save': save[0] if save else None, 'k': k,
                      'cin': cin, 'cout': cout, 'act': 'relu' if relu[0] else 'none', 'w': w, 'b': b,
                      'in_groups': in_groups, 'out_offsets': out_offsets})
    assert saw_d2s
    # Resolve channel-group permutations between producer and consumer and the skip.
    by_save = {c['save']: i for i, c in enumerate(convs)}
    layers = []
    for i, c in enumerate(convs):
        w = c['w']
        skip = None
        if i > 0:
            texs0 = c['in_groups'][0]
            prod = by_save[texs0[0][0]]
            assert prod == i - 1, 'non-sequential input'
            if len(texs0) == 2:  # input = layer[i-1] + layer[s]
                s = by_save[texs0[1][0]]
                assert layers[i - 1].get('skip') is None
                layers[i - 1]['skip'] = s
            # permute input channels: consumer group g reads producer group out_offsets[offset]
            if c['cin'] > 1:
                perm = []
                for g in range(c['cin'] // 4):
                    off = c['in_groups'][g][0][1]
                    pg = convs[prod]['out_offsets'][off]
                    perm.append(pg)
                # consumer channel 4g+j == producer channel 4*perm[g]+j
                wn = np.zeros_like(w)
                for g, pg in enumerate(perm):
                    wn[:, 4 * pg:4 * pg + 4] = w[:, 4 * g:4 * g + 4]
                w = wn
        layers.append({'k': c['k'], 'cin': c['cin'], 'cout': c['cout'], 'act': c['act'],
                       'w': w.astype(np.float32), 'b': c['b'].astype(np.float32)})
        if skip is not None:
            layers[-1]['skip'] = skip
    # producer output groups are stored in result order; offsets only matter via the permutation above,
    # so channel index == 4*resultK + lane for every layer.
    assert layers[-1]['cout'] == 4
    return layers


if __name__ == '__main__':
    import sys
    for p in sys.argv[1:]:
        ls = parse_artcnn(p)
        print(p, 'MAC/px', sum(l['k'] ** 2 * l['cin'] * l['cout'] for l in ls))
        for l in ls:
            print('  k%d %d->%d %s skip=%s |w|max=%.3f' % (l['k'], l['cin'], l['cout'], l['act'],
                                                         l.get('skip'), np.abs(l['w']).max()))
