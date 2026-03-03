def print_log(filename, encoding):
    try:
        with open(filename, 'r', encoding=encoding, errors='ignore') as f:
            for line in f:
                print(line.rstrip())
        return True
    except Exception:
        return False

if not print_log('debug_shapes_flushed.log', 'utf-16'):
    if not print_log('debug_shapes_flushed.log', 'utf-8'):
        print_log('debug_shapes_flushed.log', 'mbcs')
