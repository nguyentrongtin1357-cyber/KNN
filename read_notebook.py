import sys, json
sys.stdout.reconfigure(encoding='utf-8')

nb = json.load(open('Online_News_Popularity_KNN.ipynb', encoding='utf-8'))

for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    ctype = cell['cell_type']
    print(f"\n{'='*70}")
    print(f"CELL #{i+1} [{ctype.upper()}]")
    print('='*70)
    print(src[:3000])
    if ctype == 'code':
        outputs = cell.get('outputs', [])
        for o in outputs:
            if o.get('output_type') == 'stream':
                txt = ''.join(o.get('text', []))
                print(f"\n--- OUTPUT ---")
                print(txt[:3000])
            elif o.get('output_type') == 'display_data':
                print(f"\n--- [PLOT/IMAGE] ---")
