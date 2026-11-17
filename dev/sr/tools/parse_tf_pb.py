"""Parse Saafke/FSRCNN_Tensorflow frozen graphs (.pb) into spec-family layers.

No TensorFlow needed: a minimal protobuf wire-format reader extracts GraphDef nodes,
then the graph is walked from the input placeholder following the FSRCNN topology
(Conv2D -> add(bias) -> PReLU ... -> Conv2D(1x1, 4 ch) -> DepthToSpace -> BiasAdd).
The final per-image scalar bias (added after depth-to-space) is folded into the
last conv's 4 output biases, which is exact.

Returns layers as numpy float32 arrays in PyTorch layout w[cout][cin][k][k].
"""
import struct
import numpy as np


def _varint(buf, i):
    r = 0
    s = 0
    while True:
        b = buf[i]
        i += 1
        r |= (b & 0x7F) << s
        s += 7
        if b < 0x80:
            return r, i


def _fields(buf):
    """Yield (field_number, wire_type, value) for a message buffer."""
    i = 0
    n = len(buf)
    while i < n:
        key, i = _varint(buf, i)
        fn, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(buf, i)
        elif wt == 1:
            v = buf[i:i + 8]
            i += 8
        elif wt == 2:
            ln, i = _varint(buf, i)
            v = buf[i:i + ln]
            i += ln
        elif wt == 5:
            v = buf[i:i + 4]
            i += 4
        else:
            raise ValueError('unsupported wire type %d' % wt)
        yield fn, wt, v


DT_FLOAT = 1


def _tensor(buf):
    dtype = None
    dims = []
    content = None
    fvals = []
    for fn, wt, v in _fields(buf):
        if fn == 1:
            dtype = v
        elif fn == 2:  # TensorShapeProto
            for f2, _, v2 in _fields(v):
                if f2 == 2:  # Dim
                    size = 0
                    for f3, _, v3 in _fields(v2):
                        if f3 == 1:
                            size = v3
                    dims.append(size)
        elif fn == 4:
            content = bytes(v)
        elif fn == 5:
            if wt == 2:
                fvals.extend(struct.unpack('<%df' % (len(v) // 4), bytes(v)))
            else:
                fvals.append(struct.unpack('<f', bytes(v))[0])
    if dtype != DT_FLOAT:
        return None
    count = int(np.prod(dims)) if dims else 1
    if content is not None:
        arr = np.frombuffer(content, dtype='<f4').copy()
    else:
        arr = np.array(fvals, dtype=np.float32)
        if arr.size == 1 and count > 1:
            arr = np.full(count, arr[0], dtype=np.float32)
    return arr.reshape(dims) if dims else arr.reshape(())


def read_graph(path):
    data = open(path, 'rb').read()
    nodes = {}
    order = []
    for fn, _, v in _fields(data):
        if fn != 1:
            continue
        name = op = None
        inputs = []
        attrs = {}
        for f2, _, v2 in _fields(v):
            if f2 == 1:
                name = bytes(v2).decode()
            elif f2 == 2:
                op = bytes(v2).decode()
            elif f2 == 3:
                inputs.append(bytes(v2).decode())
            elif f2 == 5:
                key = val = None
                for f3, _, v3 in _fields(v2):
                    if f3 == 1:
                        key = bytes(v3).decode()
                    elif f3 == 2:
                        val = v3
                attrs[key] = val
        node = {'name': name, 'op': op, 'inputs': inputs, 'attrs': attrs}
        if op == 'Const' and 'value' in attrs:
            for f4, _, v4 in _fields(attrs['value']):
                if f4 == 8:
                    node['tensor'] = _tensor(v4)
        nodes[name] = node
        order.append(name)
    return nodes, order


def _consumers(nodes, name):
    return [n for n in nodes.values() if any(i.split(':')[0].lstrip('^') == name for i in n['inputs'])]


def _const(nodes, name):
    n = nodes[name.split(':')[0]]
    if n['op'] == 'Identity':
        return _const(nodes, n['inputs'][0])
    assert n['op'] == 'Const', (name, n['op'])
    return n['tensor']


def _other_input(node, name):
    ins = [i for i in node['inputs'] if i.split(':')[0] != name]
    assert len(ins) == 1, node
    return ins[0]


def _follow_prelu(nodes, x):
    """Match prelu(x) = relu(x) + alpha * (x - abs(x)) * 0.5; return (out_name, alpha)."""
    cons = _consumers(nodes, x)
    ops = sorted(c['op'] for c in cons)
    assert ops == ['Abs', 'Relu', 'Sub'], (x, ops)
    relu = next(c for c in cons if c['op'] == 'Relu')
    sub = next(c for c in cons if c['op'] == 'Sub')
    (mul1,) = _consumers(nodes, sub['name'])
    assert mul1['op'] == 'Mul'
    alpha = None
    other = _other_input(mul1, sub['name'])
    # mul(alpha, sub) then mul(.., 0.5)  -- either order of the two Muls is accepted
    t = _const(nodes, other)
    (mul2,) = _consumers(nodes, mul1['name'])
    assert mul2['op'] == 'Mul'
    t2 = _const(nodes, _other_input(mul2, mul1['name']))
    if t.ndim == 1 and t.size > 1 or (t.ndim == 1 and abs(float(t2) - 0.5) < 1e-7):
        alpha, half = t, t2
    else:
        alpha, half = t2, t
    assert abs(float(np.asarray(half).reshape(-1)[0]) - 0.5) < 1e-7, half
    (add,) = [c for c in _consumers(nodes, mul2['name'])]
    assert add['op'] in ('Add', 'AddV2') and relu['name'] in [i.split(':')[0] for i in add['inputs']], add
    return add['name'], np.asarray(alpha, dtype=np.float32).reshape(-1)


def parse_fsrcnn(path):
    nodes, _ = read_graph(path)
    start = next(n for n in nodes.values() if n['op'] in ('IteratorGetNext', 'Placeholder')
                 and any(c['op'] == 'Conv2D' for c in _consumers(nodes, n['name'])))
    cur = start['name']
    layers = []
    while True:
        cons = _consumers(nodes, cur)
        conv = [c for c in cons if c['op'] == 'Conv2D']
        if not conv:
            break
        (conv,) = conv
        w = _const(nodes, conv['inputs'][1])  # HWIO
        k, k2, cin, cout = w.shape
        assert k == k2 and k % 2 == 1
        wt = np.transpose(w, (3, 2, 0, 1)).astype(np.float32)  # -> [cout][cin][kh][kw]
        (nxt,) = _consumers(nodes, conv['name'])
        if nxt['op'] in ('Add', 'AddV2', 'BiasAdd'):
            b = _const(nodes, _other_input(nxt, conv['name'])).reshape(-1).astype(np.float32)
            out, alpha = _follow_prelu(nodes, nxt['name'])
            layers.append({'k': k, 'cin': cin, 'cout': cout, 'act': 'prelu', 'w': wt, 'b': b, 'alpha': alpha})
            cur = out
        elif nxt['op'] == 'DepthToSpace':
            assert cout == 4, cout
            (badd,) = _consumers(nodes, nxt['name'])
            assert badd['op'] == 'BiasAdd'
            b = _const(nodes, _other_input(badd, nxt['name'])).reshape(-1).astype(np.float32)
            assert b.size == 1
            layers.append({'k': k, 'cin': cin, 'cout': cout, 'act': 'none', 'w': wt,
                           'b': np.full(4, b[0], dtype=np.float32)})
            break
        else:
            raise ValueError('unexpected op after conv: %s' % nxt['op'])
    return layers


if __name__ == '__main__':
    import sys
    for p in sys.argv[1:]:
        ls = parse_fsrcnn(p)
        print(p)
        for l in ls:
            print('  k%d %d->%d %s |w|max=%.3f' % (l['k'], l['cin'], l['cout'], l['act'], np.abs(l['w']).max()))
