import sys
sys.stdout.reconfigure(encoding='utf-8')
import json

nb = json.load(open('Online_News_Popularity_KNN.ipynb', encoding='utf-8'))
cells = nb['cells']
print(f"Tong so cell: {len(cells)}")
print("=" * 70)

for i, c in enumerate(cells):
    src = ''.join(c['source'])
    if c['cell_type'] == 'markdown':
        if '## BƯỚC' in src:
            heading = [l for l in src.split('\n') if '## BƯỚC' in l][0].strip()
            print(f"\n[MARKDOWN] {heading}")
        elif src.strip().startswith('#'):
            print(f"[HEADER]   {src.strip().split(chr(10))[0][:60]}")
    elif c['cell_type'] == 'code':
        lines = [l.strip() for l in src.split('\n') if l.strip()]
        first = lines[0][:65] if lines else '(empty)'
        outputs = c.get('outputs', [])
        has_stdout = any(o.get('output_type') == 'stream' for o in outputs)
        has_plot   = any(o.get('output_type') == 'display_data' for o in outputs)
        tag_out  = "OUTPUT" if has_stdout else "no-output"
        tag_plot = "| PLOT" if has_plot else ""
        print(f"  [CODE]   {first}...")
        print(f"           -> {tag_out} {tag_plot}")
